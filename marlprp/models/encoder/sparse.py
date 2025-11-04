
import torch
import torch.nn as nn

from einops import rearrange

from .mixed_scores import MixedScoreFF
from marlprp.models.policy_args import TransformerParams


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, num_groups: int = None) -> torch.Tensor:
    size = list(src.size())
    size[dim] = num_groups
    out = src.new_zeros(*size)   
    out.index_add_(dim, index, src)      
    return out


class SparseCrossAttention(nn.Module):

    def __init__(self, params: TransformerParams):
        super().__init__()
        self.ms_scores_layer = MixedScoreFF(params)
        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_o = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(params.dropout)

    @staticmethod
    def _edges_from_weight_matrix(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, p = (weights > 0).detach().nonzero(as_tuple=True)
        w = weights[b, s, p]
        return b, s, p, w

    def _determine_weights_and_combine(self, logits: torch.Tensor, values: torch.Tensor, group_ids, num_groups):
        logits = logits.exp()
        denom = scatter_add(logits, group_ids, dim=0, num_groups=num_groups)
        attn = logits / (denom[group_ids] + 1e-9)
        attn = self.dropout(attn)

        weighted_v = torch.einsum('bh,bhd->bhd', attn, values)
        out = scatter_add(weighted_v, group_ids, dim=0, num_groups=num_groups)
        return out
        

    def forward(self, row_emb, col_emb, cost_mat = None, attn_mask: torch.Tensor = None):
        assert cost_mat is not None, "SparseCrossAttention requires a cost matrix"
        bs, row_cnt, emb_dim = row_emb.shape

        b_idx, s_idx, p_idx, w = self._edges_from_weight_matrix(cost_mat)

        row_emb = row_emb[b_idx, s_idx]
        col_emb = col_emb[b_idx, p_idx]

        q = self.W_q(row_emb).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(col_emb).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(col_emb).view(-1, self.num_heads, self.head_dim)

        logits = torch.einsum('bhd,bhd->bh', q, k) / (self.head_dim ** 0.5)
        logits = self.ms_scores_layer(logits, w.unsqueeze(1))

        group_ids = b_idx * row_cnt + s_idx
        num_groups = bs * row_cnt
        # max_per_group, _ = scatter_max(logits, group_ids, dim=0, dim_size=num_groups)
        # logits = logits #  - max_per_group[group_ids]
        heads = self._determine_weights_and_combine(logits, v, group_ids.detach(), num_groups)
        return self.W_o(rearrange(heads, "(b s) h d -> b s (h d)", b=bs, s=row_cnt, h=self.num_heads))



class EfficientSparseCrossAttention(SparseCrossAttention):
    def __init__(self, params: TransformerParams):
        super(SparseCrossAttention, self).__init__()

        embed_dim = params.embed_dim
        self.tanh_clip = params.ms_scores_tanh_clip
        self.temp = params.ms_scores_softmax_temp
        self.num_heads = params.num_heads
        self.head_dim = params.qkv_dim

        self.dropout = nn.Dropout(params.dropout)

        self.Wqv1 = nn.Linear(embed_dim, 2 * embed_dim, bias=params.bias)
        self.Wkv2 = nn.Linear(embed_dim, 2 * embed_dim, bias=params.bias)

        self.ms_scores_layer = MixedScoreFF(params)

        self.out_proj1 = nn.Linear(embed_dim, embed_dim, bias=params.bias)
        self.out_proj2 = nn.Linear(embed_dim, embed_dim, bias=params.bias)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, attn_mask: torch.Tensor = None, cost_mat: torch.Tensor = None):
        assert cost_mat is not None, "EfficientSparseCrossAttention requires a cost matrix"
    
        bs = x1.size(0)
        row_cnt = x1.size(-2)
        col_cnt = x2.size(-2)
        embed_dim = x1.size(-1)

        # Project query, key, value
        q, v1 = rearrange(
            self.Wqv1(x1), "b s (two h d) -> two b s h d", two=2, h=self.num_heads
        ).unbind(dim=0)

        # Project query, key, value
        k, v2 = rearrange(
            self.Wkv2(x2), "b s (two h d) -> two b s h d", two=2, h=self.num_heads
        ).unbind(dim=0)

        b_idx, s_idx, p_idx, w = self._edges_from_weight_matrix(cost_mat)
        q = q[b_idx, s_idx].contiguous()
        k = k[b_idx, p_idx].contiguous()
        v1 = v1[b_idx, s_idx].contiguous()
        v2 = v2[b_idx, p_idx].contiguous()

        # (bs * num_edges, num_heads)
        logits = (q * k).sum(-1) / (self.head_dim ** 0.5)
        logits = self.ms_scores_layer(logits, w.unsqueeze(1))

        row_group_ids = b_idx * row_cnt + s_idx
        num_row_groups = bs * row_cnt

        col_group_ids = b_idx * col_cnt + p_idx
        num_col_groups = bs * col_cnt

        row_heads = self._determine_weights_and_combine(logits, v2, row_group_ids.detach(), num_row_groups)
        h1 = self.out_proj1(rearrange(row_heads, "(b s) h d -> b s (h d)", b=bs, s=row_cnt, h=self.num_heads))

        col_heads = self._determine_weights_and_combine(logits, v1, col_group_ids.detach(), num_col_groups)
        h2 = self.out_proj2(rearrange(col_heads, "(b s) h d -> b s (h d)", b=bs, s=col_cnt, h=self.num_heads))

        return h1, h2