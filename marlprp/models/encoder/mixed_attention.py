import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn
import math
from einops import rearrange
from marlprp.models.policy_args import TransformerParams


def apply_weights_and_combine(
        logits: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor = None, 
        scale: bool = False,
        tanh_clipping: float = 0.
    ):
    logits = logits.clone()
    # scale to avoid numerical underflow
    if scale:
        logits = logits / (logits.std() + 1e-5)

    # tanh clipping to avoid explosions
    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    if mask is not None:
        mask = mask.clone()
        # hack to avoid nans
        all_masked = mask.all(-1, keepdim=True).expand_as(mask)
        mask[all_masked] = False
        logits = logits.masked_fill(mask, float("-inf"))

    # shape: (batch, num_heads, row_cnt, col_cnt)
    weights = torch.softmax(logits, dim=-1)

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
        ms_output_dim = num_heads if not params.ms_split_heads else 2 * num_heads
        self.split_heads = params.ms_split_heads
        # in initialization, account for the fact that we basically only have two input features
        mix1_init = math.sqrt(1/2)
        mix2_init = math.sqrt(1/ms_hidden_dim)

        self.lin1 = nn.Linear(2 * num_heads, num_heads * ms_hidden_dim, bias=False)
        self.lin2 = nn.Linear(num_heads * ms_hidden_dim, ms_output_dim, bias=False)

        self.activation = _get_activation_fn(params.activation)
        nn.init.uniform_(self.lin1.weight, a=-mix1_init, b=mix1_init)
        nn.init.uniform_(self.lin2.weight, a=-mix2_init, b=mix2_init)

    def forward(self, dot_product_score, cost_mat_score):
        # dot_product_score shape: (batch, head_num, row_cnt, col_cnt)
        # cost_mat_score shape: (batch, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=-1)
        two_scores = rearrange(two_scores, "b h r c s -> b r c (h s)")
        # shape: (batch, row_cnt, col_cnt, 2 * num_heads)
        ms = self.lin2(self.activation(self.lin1(two_scores)))
        if self.split_heads:
            # shape: (batch, row_cnt, head_num, col_cnt)
            mixed_scores = rearrange(ms, "b r c (h two) -> b h r c two", two=2)
            ms1, ms2 = mixed_scores.chunk(2, dim=-1)
            ms1, ms2 = ms1.squeeze(-1), ms2.squeeze(-1).transpose(-2, -1)
        else:
            mixed_scores = rearrange(ms, "b r c h -> b h r c")
            ms1, ms2 = mixed_scores, mixed_scores.transpose(-2, -1)
        return ms1, ms2


class EfficientMixedScoreMultiHeadAttentionLayer(nn.Module):
    def __init__(self, model_params: TransformerParams):
        super().__init__()

        embed_dim = model_params.embed_dim
        qkv_dim = model_params.qkv_dim

        self.num_heads = model_params.num_heads
        self.qkv_dim = qkv_dim
        self.norm_factor = 1 / math.sqrt(qkv_dim)

        self.Wqv1 = nn.Linear(embed_dim, 2 * embed_dim, bias=model_params.bias)
        self.Wkv2 = nn.Linear(embed_dim, 2 * embed_dim, bias=model_params.bias)

        self.mixed_scores_layer = MixedScoreFF(model_params)

        self.out_proj1 = nn.Linear(embed_dim, embed_dim, bias=model_params.bias)
        self.out_proj2 = nn.Linear(embed_dim, embed_dim, bias=model_params.bias)

    def forward(self, x1, x2, attn_mask = None, cost_mat = None):
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
            # shape: (batch, num_heads, row_cnt, col_cnt)
            cost_mat_score = cost_mat[:, None, :, :].expand_as(dot)
            ms1, ms2 = self.mixed_scores_layer(dot, cost_mat_score)

        if attn_mask is not None:
            mask1 = attn_mask.view(batch_size, 1, row_cnt, col_cnt).expand_as(ms1)
            mask2 = mask1.transpose(-2, -1)
        else:
            mask1, mask2 = None, None

        h1 = self.out_proj1(apply_weights_and_combine(ms1, v2, mask=mask1))
        h2 = self.out_proj2(apply_weights_and_combine(ms2, v1, mask=mask2))

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

        self.Wq = nn.Linear(embed_dim, num_heads * qkv_dim, bias=model_params.bias)
        self.Wk = nn.Linear(embed_dim, num_heads * qkv_dim, bias=model_params.bias)
        self.Wv = nn.Linear(embed_dim, num_heads * qkv_dim, bias=model_params.bias)

        self.out_proj1 = nn.Linear(embed_dim, embed_dim, bias=model_params.bias)
        self.out_proj2 = nn.Linear(embed_dim, embed_dim, bias=model_params.bias)

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

        # shape: (batch, head_num, row_cnt, col_cnt)
        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, self.num_heads, row_cnt, col_cnt)

        mixed_scores, _ = self.mixed_scores_layer(dot_product_score, cost_mat_score)

        out = self.out_proj1(apply_weights_and_combine(mixed_scores, v, mask=attn_mask))
        return out


class MixedScoreMultiHeadAttentionLayer(nn.Module):

    def __init__(self, model_params: TransformerParams):
        super().__init__()
        self.row_encoding_block = MixedScoreMultiHeadAttention(model_params)
        self.col_encoding_block = MixedScoreMultiHeadAttention(model_params)

    def forward(self, x1, x2, attn_mask = None, cost_mat = None):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        x1_out = self.row_encoding_block(x1, x2, cost_mat)
        x2_out = self.col_encoding_block(x2, x1, cost_mat.transpose(1, 2))

        return x1_out, x2_out