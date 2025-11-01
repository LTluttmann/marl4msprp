import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, shelf_embs, product_embs, supply, attn_mask=None):
        """
        shelf_embs: [B, S, D]
        product_embs: [B, P, D]
        supply: [B, S, P] binary or real mask (0 = absent)
        """
        device = shelf_embs.device
        B, S, D = shelf_embs.shape
        _, P, _ = product_embs.shape
        H = self.num_heads
        d_h = self.head_dim

        # Project to multi-head
        Q = self.W_q(shelf_embs).view(B, S, H, d_h)
        K = self.W_k(product_embs).view(B, P, H, d_h)
        V = self.W_v(product_embs).view(B, P, H, d_h)

        # Get all valid (b, s, p) pairs
        b_idx, s_idx, p_idx = torch.nonzero(supply > 0, as_tuple=True)
        n_edges = b_idx.numel()

        # Gather only relevant Q, K, V
        q = Q[b_idx, s_idx]                # [E, H, d_h]
        k = K[b_idx, p_idx]                # [E, H, d_h]
        v = V[b_idx, p_idx]                # [E, H, d_h]

        # Compute scaled dot-product attention per (b,s)
        attn_logits = (q * k).sum(-1) / (d_h ** 0.5)  # [E, H]

        # We need softmax over all p per (b,s)
        key = torch.stack([b_idx, s_idx], dim=1)      # [E, 2]
        unique_bs, inv = torch.unique(key, dim=0, return_inverse=True)
        num_edges_per_bs = torch.bincount(inv)

        # Compute softmax within each (b,s) group
        attn_exp = torch.exp(attn_logits - attn_logits.amax(0, keepdim=True)[inv])
        denom = torch.zeros((len(unique_bs), H), device=device)
        denom.index_add_(0, inv, attn_exp)
        attn_weights = attn_exp / (denom[inv] + 1e-9)  # [E, H]

        # Apply dropout and weight values
        attn_weights = self.dropout(attn_weights)
        weighted_v = attn_weights.unsqueeze(-1) * v  # [E, H, d_h]

        # Aggregate results back to shelf embeddings
        out_flat = torch.zeros((B, S, H, d_h), device=device)
        out_flat.index_add_(1, s_idx + b_idx * S, weighted_v.view(-1, H, d_h))  # (conceptually per (b,s))

        # Because we added across batch·shelf index, reshape
        out_flat = out_flat.view(B, S, H, d_h)
        out = out_flat.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        return self.W_o(out)
