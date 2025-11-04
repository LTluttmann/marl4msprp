import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.modules.transformer import _get_activation_fn

from marlprp.models.policy_args import TransformerParams


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
        self.num_heads = params.num_heads
        ms_hidden_dim = params.ms_hidden_dim

        # in initialization, account for the fact that we basically only have two input features
        mix1_init = math.sqrt(1/2)
        mix2_init = math.sqrt(1/ms_hidden_dim)

        self.lin1 = nn.Linear(2 * self.num_heads, ms_hidden_dim * self.num_heads, bias=params.bias)
        self.lin2 = nn.Linear(ms_hidden_dim * self.num_heads, self.num_heads, bias=params.bias)
        self.activation = _get_activation_fn(params.activation)
        
        nn.init.uniform_(self.lin1.weight, a=-mix1_init, b=mix1_init)
        nn.init.uniform_(self.lin2.weight, a=-mix2_init, b=mix2_init)
        if params.bias:
            nn.init.zeros_(self.lin1.bias)
            nn.init.zeros_(self.lin2.bias)

        self.chunk_ms_scores_batch = params.chunk_ms_scores_batch or 0
        # basically adds a head dimension to cost matrix
        self.alpha = nn.Parameter(torch.ones(params.num_heads), requires_grad=True)


    def forward(self, dot_product_score: torch.Tensor, cost_mat_score: torch.Tensor):

        alpha = self.alpha.view(1, self.num_heads, *[1] * (dot_product_score.dim() - 2))

        scores = torch.stack((dot_product_score, alpha * cost_mat_score), dim=-1)

        scores = rearrange(scores, "b h ... s -> b ... (h s)")

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
        mixed_scores = rearrange(mixed_scores, "b ... h -> b h ...")

        return mixed_scores