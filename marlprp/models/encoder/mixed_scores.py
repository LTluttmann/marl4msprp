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
        self.split_heads = params.param_sharing

        ms_output_dim = self.num_heads if not self.split_heads else 2 * self.num_heads
        ms_hidden_dim = params.ms_hidden_dim

        mix1_init = math.sqrt(1 / 2)
        mix2_init = math.sqrt(1 / ms_hidden_dim)

        self.lin1 = nn.Linear(2 * self.num_heads, ms_hidden_dim * self.num_heads, bias=params.bias)
        self.lin2 = nn.Linear(ms_hidden_dim * self.num_heads, ms_output_dim, bias=params.bias)
        self.activation = _get_activation_fn(params.activation)
        if hasattr(self.activation, "inplace"):
            self.activation.inplace = True

        nn.init.uniform_(self.lin1.weight, a=-mix1_init, b=mix1_init)
        nn.init.uniform_(self.lin2.weight, a=-mix2_init, b=mix2_init)
        if params.bias:
            nn.init.zeros_(self.lin1.bias)
            nn.init.zeros_(self.lin2.bias)

        self.chunk_ms_scores_batch = params.chunk_ms_scores_batch or 0
        self.alpha = nn.Parameter(torch.ones(params.num_heads), requires_grad=True)

        def _ff_block(x):
            x = F.linear(x, self.lin1.weight, self.lin1.bias)
            x = self.activation(x)
            x = F.linear(x, self.lin2.weight, self.lin2.bias)
            return x
        
        self.ff_block = _ff_block


    def forward(self, dot_product_score: torch.Tensor, cost_mat_score: torch.Tensor):
        alpha = self.alpha.view(1, self.num_heads, *[1] * (dot_product_score.dim() - 2))
        scores = torch.stack((dot_product_score, alpha * cost_mat_score), dim=-1)
        scores = rearrange(scores, "b h ... s -> b ... (h s)")


        if self.chunk_ms_scores_batch > 1 and not torch.is_grad_enabled():
            chunks = scores.chunk(self.chunk_ms_scores_batch, dim=0)
            outputs = []
            for sc in chunks:
                outputs.append(self.ff_block(sc))
                del sc
            scores = torch.cat(outputs, dim=0)
            del outputs
        else:
            scores = self.ff_block(scores)

        if self.split_heads:
            # shape: (batch, row_cnt, head_num, col_cnt)
            scores = rearrange(scores, "b ... (h two) -> b h ... two", two=2)
            ms1, ms2 = scores.chunk(2, dim=-1)
            ms1, ms2 = ms1.squeeze(-1), ms2.squeeze(-1)
            del scores
            if ms1.dim() < 4:
                return ms1, ms2
            else:
                return ms1, ms2.transpose(-2, -1).contiguous()
        
        return scores

        