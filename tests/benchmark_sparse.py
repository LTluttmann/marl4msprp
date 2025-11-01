import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from torch_scatter import scatter_add, scatter_max
from torch.nn.modules.transformer import _get_activation_fn
from einops import rearrange


torch.manual_seed(0)
device = "cuda:0"



class MixedScoreFF(nn.Module):
    def __init__(self, embed_dim, num_heads=8, bias=True) -> None:
        super().__init__()

        ms_hidden_dim = embed_dim // num_heads

        # in initialization, account for the fact that we basically only have two input features
        mix1_init = math.sqrt(1/2)
        mix2_init = math.sqrt(1/ms_hidden_dim)

        self.lin1 = nn.Linear(2 * num_heads, ms_hidden_dim * num_heads, bias=bias)
        self.lin2 = nn.Linear(ms_hidden_dim * num_heads, num_heads, bias=bias)
        self.activation = _get_activation_fn("relu")
        
        nn.init.uniform_(self.lin1.weight, a=-mix1_init, b=mix1_init)
        nn.init.uniform_(self.lin2.weight, a=-mix2_init, b=mix2_init)
        if bias:
            nn.init.zeros_(self.lin1.bias)
            nn.init.zeros_(self.lin2.bias)


    def forward(self, dot_product_score, cost_mat_score):
        # dot_product_score shape: (batch, head_num, row_cnt, col_cnt)
        # cost_mat_score shape: (batch, head_num, row_cnt, col_cnt)

        # shape: (batch, head_num, row_cnt, col_cnt, num_scores)
        scores = torch.stack((dot_product_score, cost_mat_score.expand_as(dot_product_score)), dim=-1)
        # shape: (batch, row_cnt, col_cnt, num_heads*num_scores)
        scores = rearrange(scores, "b h ... s -> b ... (h s)")

        mixed_scores = self.lin2(self.activation(self.lin1(scores)))

        # shape: (batch, row_cnt, head_num, col_cnt)
        mixed_scores = rearrange(mixed_scores, "b ... h -> b h ...")

        return mixed_scores




# --- Baseline: Dense Masked Cross-Attention ---
class DenseMaskedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ms_scores = MixedScoreFF(embed_dim, num_heads)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, shelf_embs, product_embs, mask):
        B, S, D = shelf_embs.shape
        _, P, _ = product_embs.shape
        H = self.num_heads
        d_h = self.head_dim

        Q = self.W_q(shelf_embs).view(B, H, S, d_h)
        K = self.W_k(product_embs).view(B, H, P, d_h)
        V = self.W_v(product_embs).view(B, H, P, d_h)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_h ** 0.5)
        attn_scores = self.ms_scores(attn_scores, mask.unsqueeze(1))
        
        attn_mask = mask.eq(0).unsqueeze(1)  # [B, 1, S, P]
        attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, V)  # [B, H, S, d_h]
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_o(out)

# --- Sparse Attention (using torch_scatter) ---
class SparseCrossAttentionScatter(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ms_scores = MixedScoreFF(embed_dim, num_heads)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _edges_from_mask(mask):
        nz = (mask > 0).nonzero(as_tuple=False)
        b, s, p = nz.unbind(1)
        w = mask[b, s, p]
        return b, s, p, w

    def forward(self, shelf_embs, product_embs, mask):
        B, S, D = shelf_embs.shape
        _, P, _ = product_embs.shape
        H, d_h = self.num_heads, self.head_dim

        Q = self.W_q(shelf_embs).view(B, S, H, d_h)
        K = self.W_k(product_embs).view(B, P, H, d_h)
        V = self.W_v(product_embs).view(B, P, H, d_h)

        b_idx, s_idx, p_idx, w = self._edges_from_mask(mask)
        q = Q[b_idx, s_idx]
        k = K[b_idx, p_idx]
        v = V[b_idx, p_idx]

        logits = (q * k).sum(-1) / (d_h ** 0.5)
        logits = self.ms_scores(logits, w.unsqueeze(1))

        group_ids = b_idx * S + s_idx
        num_groups = B * S
        # max_per_group, _ = scatter_max(logits, group_ids, dim=0, dim_size=num_groups)
        logits_norm = logits #  - max_per_group[group_ids]
        exp_logits = torch.exp(logits_norm)
        denom = scatter_add(exp_logits, group_ids, dim=0, dim_size=num_groups)
        attn = exp_logits / (denom[group_ids] + 1e-9)
        attn = self.dropout(attn)

        weighted_v = attn.unsqueeze(-1) * v
        out = scatter_add(weighted_v, group_ids, dim=0, dim_size=num_groups)
        out = out.view(B, S, H, d_h).permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.W_o(out)

# --- Benchmark Function ---
def benchmark(layer, *inputs, n_warmup=10, n_runs=50):
    torch.cuda.synchronize()
    for _ in range(n_warmup): layer(*inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_runs): layer(*inputs)
    torch.cuda.synchronize()
    t1 = time.time()
    mem = torch.cuda.max_memory_allocated() / 1024**2
    return (t1 - t0) / n_runs * 1000, mem  # ms, MB

# --- Setup ---
B, S, P, D, H = 100, 50, 1000, 256, 8
density = 0.02

shelf = torch.randn(B, S, D, device=device)
prod  = torch.randn(B, P, D, device=device)
mask  = (torch.rand(B, S, P, device=device) < density).float()

dense_layer  = DenseMaskedCrossAttention(D, H).to(device)
sparse_layer = SparseCrossAttentionScatter(D, H).to(device)

# --- Benchmark ---
dense_time, dense_mem = benchmark(dense_layer, shelf, prod, mask)
sparse_time, sparse_mem = benchmark(sparse_layer, shelf, prod, mask)

print(f"Dense attention:  {dense_time:.2f} ms / {dense_mem:.1f} MB")
print(f"Sparse attention: {sparse_time:.2f} ms / {sparse_mem:.1f} MB")
print(f"Sparsity: {density*100:.1f}%  |  Speedup: {dense_time/sparse_time:.2f}x  |  Mem ratio: {dense_mem/sparse_mem:.2f}x")
