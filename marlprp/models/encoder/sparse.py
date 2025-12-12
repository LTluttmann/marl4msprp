
import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
from einops import rearrange

from .utils import _scatter_add
from .mixed_scores import MixedScoreFF, apply_weights_and_combine
from marlprp.models.policy_args import TransformerParams


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
        attn = scatter_softmax(logits, group_ids, dim=0, dim_size=num_groups)
        # attn = self._scatter_softmax(logits, group_ids, num_groups)
        attn = self.dropout(attn)
        weighted_v = torch.einsum('bh,bhd->bhd', attn, values)
        del attn
        out = _scatter_add(weighted_v, group_ids, dim=0, dim_size=num_groups)
        return out
        

    def forward(self, row_emb, col_emb, cost_mat, attn_mask: torch.Tensor = None):
        assert cost_mat is not None, "SparseCrossAttention requires a cost matrix"
        bs, row_cnt, emb_dim = row_emb.shape

        b_idx, s_idx, p_idx, w = self._edges_from_weight_matrix(cost_mat)

        row_emb = row_emb[b_idx, s_idx]
        col_emb = col_emb[b_idx, p_idx]

        q = self.W_q(row_emb).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(col_emb).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(col_emb).view(-1, self.num_heads, self.head_dim)
        del row_emb, col_emb   # free memory early

        logits = torch.einsum('bhd,bhd->bh', q, k) / (self.head_dim ** 0.5)
        del q, k  # both are no longer needed

        logits = self.ms_scores_layer(logits, w.unsqueeze(1))

        group_ids = b_idx * row_cnt + s_idx
        num_groups = bs * row_cnt
        heads = self._determine_weights_and_combine(logits, v, group_ids, num_groups)
        del logits
        return self.W_o(rearrange(heads, "(b s) h d -> b s (h d)", b=bs, s=row_cnt, h=self.num_heads))


class SemiSparseCrossAttention(nn.Module):

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
        self.tanh_clip = params.ms_scores_tanh_clip or 0
        self.temp = params.ms_scores_softmax_temp
        self.beta = nn.Parameter(torch.full((1,self.num_heads,1,1), fill_value=float(-self.tanh_clip)), requires_grad=True)

    @staticmethod
    def _edges_from_weight_matrix(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, p = (weights > 0).detach().nonzero(as_tuple=True)
        w = weights[b, s, p]
        return b, s, p, w

    def forward(self, row_emb, col_emb, cost_mat, attn_mask: torch.Tensor = None):
        assert cost_mat is not None, "SparseCrossAttention requires a cost matrix"
        bs, row_cnt, emb_dim = row_emb.shape
        col_cnt = col_emb.size(1)

        b_idx, s_idx, p_idx, w = self._edges_from_weight_matrix(cost_mat)
        # we need to full v
        v = rearrange(self.W_v(col_emb), "b s (h d) -> b h s d", h=self.num_heads)

        row_emb = row_emb[b_idx, s_idx]
        col_emb = col_emb[b_idx, p_idx]

        q = self.W_q(row_emb).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(col_emb).view(-1, self.num_heads, self.head_dim)
        del row_emb, col_emb   # free memory early

        logits = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        del q, k  # both are no longer needed

        logits = self.ms_scores_layer(logits, w.unsqueeze(1))

        if self.tanh_clip > 0:
            logits = torch.tanh(logits) * self.tanh_clip

        # fill attn matrix with logits
        scores = self.beta.expand(bs, self.num_heads, row_cnt, col_cnt).contiguous()
        scores[b_idx, :, s_idx, p_idx] = logits

        heads = apply_weights_and_combine(scores, v, attn_mask)
        return self.W_o(heads)



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


    def forward(self, row_emb: torch.Tensor, col_emb: torch.Tensor, cost_mat: torch.Tensor, attn_mask: torch.Tensor = None):
        assert cost_mat is not None, "EfficientSparseCrossAttention requires a cost matrix"
    
        bs = row_emb.size(0)
        row_cnt = row_emb.size(-2)
        col_cnt = col_emb.size(-2)

        b_idx, s_idx, p_idx, w = self._edges_from_weight_matrix(cost_mat)
        row_emb = row_emb[b_idx, s_idx]
        col_emb = col_emb[b_idx, p_idx]
        # Project query, key, value
        q, v1 = rearrange(
            self.Wqv1(row_emb), "be (two h d) -> two be h d", two=2, h=self.num_heads, d=self.head_dim
        ).unbind(dim=0)

        # Project query, key, value
        k, v2 = rearrange(
            self.Wkv2(col_emb), "be (two h d) -> two be h d", two=2, h=self.num_heads, d=self.head_dim
        ).unbind(dim=0)

        # (bs * num_edges, num_heads)
        logits = torch.einsum('bhd,bhd->bh', q, k) / (self.head_dim ** 0.5)
        del q, k  # both are no longer needed

        l1, l2 = self.ms_scores_layer(logits, w.unsqueeze(1))
        del w, logits

        row_group_ids = b_idx * row_cnt + s_idx
        num_row_groups = bs * row_cnt

        col_group_ids = b_idx * col_cnt + p_idx
        num_col_groups = bs * col_cnt

        row_emb = self._determine_weights_and_combine(l1, v2, row_group_ids, num_row_groups)
        row_emb = self.out_proj1(rearrange(row_emb, "(b s) h d -> b s (h d)", b=bs, s=row_cnt, h=self.num_heads))

        col_emb = self._determine_weights_and_combine(l2, v1, col_group_ids, num_col_groups)
        col_emb = self.out_proj2(rearrange(col_emb, "(b s) h d -> b s (h d)", b=bs, s=col_cnt, h=self.num_heads))

        return row_emb, col_emb