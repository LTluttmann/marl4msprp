import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange
from marlprp.utils.config import ModelParams
from marlprp.models.nn.misc import Normalization


class FFN(nn.Module):
    def __init__(self, model_params: ModelParams) -> None:
        super().__init__()

        embed_dim = model_params.embed_dim
        feed_forward_hidden = model_params.feed_forward_hidden

        self.ops = nn.ModuleDict(
            {
                "norm1": Normalization(model_params),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                ),
                "norm2": Normalization(model_params),
            }
        )

    def forward(self, x, x_old):

        x = self.ops["norm1"](x_old + x)
        x = self.ops["norm2"](x + self.ops["ffn"](x))

        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_params: ModelParams, sdp_module: nn.Module) -> None:
        super().__init__()
        self.sdp_layer = sdp_module(model_params)
        self.ffn_layer = FFN(model_params)

    def forward(self, x, *args, **kwargs):
        h = self.sdp_layer(x, *args, **kwargs)

        h = self.ffn_layer(h, x)

        return h



class BaseMultiHeadSelfAttention(nn.Module):
    def __init__(self, model_params: ModelParams) -> None:
        super().__init__()
        self.embed_dim = model_params.embed_dim
        self.attention_dropout = model_params.dropout
        self.num_heads = model_params.num_heads
        self.head_dim = model_params.qkv_dim

        self.Wqkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    @staticmethod
    def _expand_pad_to_attn_mask(mask):
        new_shape = list(mask.shape)
        new_shape.insert(-1, mask.size(-1))
        mask = mask.unsqueeze(-2).expand(*new_shape)
        return mask
    
    def _expand_head_dim(self, mask: torch.Tensor):
        new_shape = list(mask.shape)
        new_shape.insert(1, self.num_heads)
        return mask.unsqueeze(1).expand(*new_shape)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("MHSA logic not implemented")



class MultiHeadSelfAttention(BaseMultiHeadSelfAttention):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(self, model_params: ModelParams) -> None:
        super().__init__(model_params)


    def forward(self, x, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        # Optionally prepare mask 
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = self._expand_pad_to_attn_mask(attn_mask)
            attn_mask = self._expand_head_dim(attn_mask)
            

        # Scaled dot product attention
        out = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))