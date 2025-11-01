import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.transformer import _get_activation_fn

from marlprp.models.policy_args import TransformerParams

from .sparse import SparseCrossAttention


def apply_weights_and_combine(
        logits: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor = None, 
        scale: bool = False,
        tanh_clipping: float = 0.,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
    logits = logits.clone()
    # scale to avoid numerical underflow
    if scale:
        logits = logits / (logits.std() + 1e-6)

    # tanh clipping to avoid explosions
    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    if mask is not None:
        mask = mask.clone()
        # Identify positions where everything is masked
        all_masked = mask.all(-1, keepdim=True)
        # For normal masked positions, set logits to -inf
        logits = logits.masked_fill(mask, float("-inf"))
        # shape: (batch, num_heads, row_cnt, col_cnt)
        weights = torch.softmax(logits / temperature, dim=-1)
        # Zero out weights where everything was masked
        weights = weights.masked_fill(all_masked.expand_as(weights), 0.0)
    else:
        # shape: (batch, num_heads, row_cnt, col_cnt)
        weights = torch.softmax(logits, dim=-1)
    weights = F.dropout(weights, p=dropout)
    # shape: (batch, num_heads, row_cnt, qkv_dim)
    out = torch.matmul(weights, v)
    # shape: (batch, row_cnt, num_heads, qkv_dim)
    out = rearrange(out, "b h s d -> b s (h d)")
    return out
      


class MixedScoreFF(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        num_heads = params.num_heads
        ms_hidden_dim = params.ms_hidden_dim
        scores_dims = params.cost_mat_dims + 1
        # in initialization, account for the fact that we basically only have two input features
        mix1_init = math.sqrt(1/scores_dims)
        mix2_init = math.sqrt(1/ms_hidden_dim)

        self.lin1 = nn.Linear(scores_dims * num_heads, ms_hidden_dim * num_heads, bias=params.bias)
        self.lin2 = nn.Linear(ms_hidden_dim * num_heads, num_heads, bias=params.bias)
        self.activation = _get_activation_fn(params.activation)
        
        nn.init.uniform_(self.lin1.weight, a=-mix1_init, b=mix1_init)
        nn.init.uniform_(self.lin2.weight, a=-mix2_init, b=mix2_init)
        if params.bias:
            nn.init.zeros_(self.lin1.bias)
            nn.init.zeros_(self.lin2.bias)
        self.chunk_ms_scores_batch = params.chunk_ms_scores_batch or 0
        # basically adds a head dimension to cost matrix
        self.alpha = nn.Parameter(torch.ones(1, params.num_heads, 1, 1, 1), requires_grad=True)


    def forward(self, dot_product_score, cost_mat_score):
        # dot_product_score shape: (batch, head_num, row_cnt, col_cnt)
        # cost_mat_score shape: (batch, head_num, row_cnt, col_cnt)

        # shape: (batch, head_num, row_cnt, col_cnt, num_scores)
        scores = torch.cat((dot_product_score, self.alpha * cost_mat_score), dim=-1)
        # shape: (batch, row_cnt, col_cnt, num_heads*num_scores)
        scores = rearrange(scores, "b h r c s -> b r c (h s)")

        # in large batches, this is very memory heavy, so optinally split the batch before the mlp
        if self.chunk_ms_scores_batch > 1 and not torch.is_grad_enabled():
            # only use this in inference mode
            chunks = scores.chunk(self.chunk_ms_scores_batch, dim=0)
            outputs = []
            for scores_chunk in chunks:
                out_chunk = self.lin2(self.activation(self.lin1(scores_chunk)))
                outputs.append(out_chunk)
            mixed_scores = torch.cat(outputs, dim=0)
        else:
            mixed_scores = self.lin2(self.activation(self.lin1(scores)))

        # shape: (batch, row_cnt, head_num, col_cnt)
        mixed_scores = rearrange(mixed_scores, "b r c h -> b h r c")

        return mixed_scores


class SimpleScoreMixer(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1, params.num_heads, 1, 1, 1), requires_grad=True)

    def forward(self, dot_product_score, cost_mat_score):
        return (dot_product_score + self.alpha * cost_mat_score).squeeze(-1)


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


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, attn_mask: torch.Tensor = None, cost_mat: torch.Tensor = None):
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
        dot = self.norm_factor * torch.matmul(q, k.transpose(-2, -1))
        
        if cost_mat is not None:
            dot = dot.unsqueeze(-1)
            # shape: (batch, num_heads, row_cnt, col_cnt, n_edge_feats)
            cost_mat_score = (
                cost_mat.view(batch_size, 1, row_cnt, col_cnt, -1)
                .expand(batch_size, self.num_heads, row_cnt, col_cnt, -1)
                .contiguous()
            )
            dot = self.mixed_scores_layer(dot, cost_mat_score)

        if attn_mask is not None:
            mask1 = attn_mask.view(batch_size, 1, row_cnt, col_cnt).expand_as(dot).contiguous()
            mask2 = mask1.clone().transpose(-2, -1)
        else:
            mask1, mask2 = None, None

        h1 = self.out_proj1(
            apply_weights_and_combine(dot, v2, mask=mask1, temperature=self.temp, tanh_clipping=self.tanh_clip, dropout=self.dropout)
        )
        h2 = self.out_proj2(
            apply_weights_and_combine(dot.transpose(-2, -1), v1, mask=mask2, temperature=self.temp, tanh_clipping=self.tanh_clip, dropout=self.dropout)
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


    def forward(self, row_emb, col_emb, cost_mat, attn_mask = None):

        # q shape: (batch, head_num, row_cnt, qkv_dim)
        q = rearrange(self.Wq(row_emb), "b s (h d) -> b h s d", h=self.num_heads)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)
        k = rearrange(self.Wk(col_emb), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.Wv(col_emb), "b s (h d) -> b h s d", h=self.num_heads)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        # shape: (batch, head_num, row_cnt, col_cnt)
        dot_product_score = self.norm_factor * torch.matmul(q, k.transpose(2, 3))
        
        if cost_mat is not None:
            dot_product_score = dot_product_score.unsqueeze(-1)
            # shape: (batch, num_heads, row_cnt, col_cnt, n_edge_feats)
            cost_mat_score = (
                cost_mat.unsqueeze(1)
                .expand(batch_size, self.num_heads, row_cnt, col_cnt, -1)
                .contiguous()
            )
            dot_product_score = self.mixed_scores_layer(dot_product_score, cost_mat_score)

        if attn_mask is not None:
            attn_mask = attn_mask.view(batch_size, 1, row_cnt, col_cnt).expand_as(dot_product_score).contiguous()

        out = self.out_proj(apply_weights_and_combine(dot_product_score, v, mask=attn_mask, dropout=self.dropout))
        return out


class MixedScoreMultiHeadAttentionLayer(nn.Module):

    def __init__(self, model_params: TransformerParams):
        super().__init__()
        if model_params.ms_sparse_attn:
            self.row_encoding_block = SparseCrossAttention(model_params.embed_dim, model_params.num_heads, dropout=model_params.dropout)
            self.col_encoding_block = SparseCrossAttention(model_params.embed_dim, model_params.num_heads, dropout=model_params.dropout)
        else:   
            self.row_encoding_block = MixedScoreMultiHeadAttention(model_params)
            self.col_encoding_block = MixedScoreMultiHeadAttention(model_params)

    def forward(self, x1, x2, attn_mask = None, cost_mat = None):
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
        x1_out = self.row_encoding_block(x1, x2, cost_mat, attn_mask=attn_mask)
        x2_out = self.col_encoding_block(
            x2, x1, cost_mat_t, attn_mask=attn_mask_t
        )

        return x1_out, x2_out