import math
import torch
import torch.nn as nn

from einops import rearrange
from marlprp.models.policy_args import TransformerParams

from .sparse import SparseCrossAttention
from .mixed_scores import apply_weights_and_combine, MixedScoreFF


class EfficientMixedScoreMultiHeadAttentionLayer(nn.Module):
    def __init__(self, model_params: TransformerParams):
        super().__init__()

        embed_dim = model_params.embed_dim
        self.tanh_clip = model_params.ms_scores_tanh_clip
        self.temp = model_params.ms_scores_softmax_temp
        self.num_heads = model_params.num_heads
        self.qkv_dim = model_params.qkv_dim
        self.norm_factor = 1 / math.sqrt(self.qkv_dim)
        self.dropout = model_params.dropout

        self.Wqv1 = nn.Linear(embed_dim, 2 * embed_dim, bias=model_params.bias)
        self.Wkv2 = nn.Linear(embed_dim, 2 * embed_dim, bias=model_params.bias)

        self.mixed_scores_layer = MixedScoreFF(model_params)

        self.out_proj1 = nn.Linear(embed_dim, embed_dim, bias=model_params.bias)
        self.out_proj2 = nn.Linear(embed_dim, embed_dim, bias=model_params.bias)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, cost_mat: torch.Tensor, attn_mask: torch.Tensor = None):
        batch_size = x1.size(0)
        row_cnt = x1.size(-2)
        col_cnt = x2.size(-2)

        # Project query, key, value
        q, v1 = rearrange(
            self.Wqv1(x1), "b s (two h d) -> two b h s d", two=2, h=self.num_heads
        ).unbind(dim=0)

        # Project query, key, value
        k, v2 = rearrange(
            self.Wkv2(x2), "b s (two h d) -> two b h s d", two=2, h=self.num_heads
        ).unbind(dim=0)

        # shape: (batch, num_heads, row_cnt, col_cnt)
        logits = self.norm_factor * torch.matmul(q, k.transpose(-2, -1))
        del q,k

        # shape: (batch, num_heads, row_cnt, col_cnt)
        cost_mat = (
            cost_mat.view(batch_size, 1, row_cnt, col_cnt)
            .expand(batch_size, self.num_heads, row_cnt, col_cnt)
            .contiguous()
        )
        l1, l2 = self.mixed_scores_layer(logits, cost_mat)
        del logits

        if attn_mask is not None:
            mask1 = attn_mask.view(batch_size, 1, row_cnt, col_cnt).expand_as(l1).contiguous()
            mask2 = mask1.clone().transpose(-2, -1)
        else:
            mask1, mask2 = None, None

        h1 = self.out_proj1(
            apply_weights_and_combine(l1, v2, mask=mask1, temperature=self.temp, tanh_clipping=self.tanh_clip, dropout=self.dropout)
        )
        h2 = self.out_proj2(
            apply_weights_and_combine(l2, v1, mask=mask2, temperature=self.temp, tanh_clipping=self.tanh_clip, dropout=self.dropout)
        )

        return h1, h2
    


class MixedScoreMultiHeadAttention(nn.Module):
    def __init__(self, model_params: TransformerParams):
        super().__init__()

        embed_dim = model_params.embed_dim
        num_heads = model_params.num_heads
        qkv_dim = model_params.qkv_dim

        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.norm_factor = 1 / math.sqrt(qkv_dim)
        self.dropout = model_params.dropout

        self.Wq = nn.Linear(embed_dim, num_heads * qkv_dim, bias=model_params.bias)
        self.Wk = nn.Linear(embed_dim, num_heads * qkv_dim, bias=model_params.bias)
        self.Wv = nn.Linear(embed_dim, num_heads * qkv_dim, bias=model_params.bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=model_params.bias)

        self.mixed_scores_layer = MixedScoreFF(model_params)


    def forward(self, row_emb: torch.Tensor, col_emb: torch.Tensor, cost_mat: torch.Tensor, attn_mask: torch.Tensor = None):

        # q shape: (batch, head_num, row_cnt, qkv_dim)
        q = rearrange(self.Wq(row_emb), "b s (h d) -> b h s d", h=self.num_heads)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)
        k = rearrange(self.Wk(col_emb), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.Wv(col_emb), "b s (h d) -> b h s d", h=self.num_heads)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        # shape: (batch, head_num, row_cnt, col_cnt)
        logits = self.norm_factor * torch.matmul(q, k.transpose(2, 3))
        del q,k
        # shape: (batch, num_heads, row_cnt, col_cnt)
        cost_mat = (
            cost_mat.unsqueeze(1)
            .expand(batch_size, self.num_heads, row_cnt, col_cnt)
            .contiguous()
        )
        logits = self.mixed_scores_layer(logits, cost_mat)

        if attn_mask is not None:
            attn_mask = attn_mask.view(batch_size, 1, row_cnt, col_cnt).expand_as(logits).contiguous()

        out = self.out_proj(apply_weights_and_combine(logits, v, mask=attn_mask, dropout=self.dropout))
        return out


class MixedScoreMultiHeadAttentionLayer(nn.Module):

    def __init__(self, model_params: TransformerParams):
        super().__init__()
        if model_params.ms_sparse_attn:
            self.row_encoding_block = SparseCrossAttention(model_params)
            self.col_encoding_block = SparseCrossAttention(model_params)
        else:   
            self.row_encoding_block = MixedScoreMultiHeadAttention(model_params)
            self.col_encoding_block = MixedScoreMultiHeadAttention(model_params)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, attn_mask: torch.Tensor = None, cost_mat: torch.Tensor = None):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        if cost_mat is not None:
            cost_mat_t = cost_mat.transpose(2, 1).contiguous()
        else:
            cost_mat_t = None

        if attn_mask is not None:
            attn_mask_t = attn_mask.transpose(2, 1).contiguous()
        else:
            attn_mask_t = None
        x1_out = self.row_encoding_block(x1, x2, cost_mat=cost_mat, attn_mask=attn_mask)
        x2_out = self.col_encoding_block(
            x2, x1, cost_mat_t, attn_mask=attn_mask_t
        )

        return x1_out, x2_out