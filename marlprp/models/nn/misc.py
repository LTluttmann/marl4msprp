import torch
import torch.nn as nn
from tensordict import TensorDict
from einops import rearrange
from typing import List
import math

from marlprp.models.policy_args import TransformerParams


class MHAWaitOperationEncoder(nn.Module):
    def __init__(self, input_size: int, params: TransformerParams) -> None:
        super().__init__()

        if params.is_multiagent_policy and params.stepwise_encoding:
            # in multiagent settings, waiting is a valid operation and we want expressive embeddings for it
            self.dummy = nn.Parameter(torch.randn(1, 1, input_size), requires_grad=True)
            self.num_heads = params.num_heads
            self.encoder = nn.MultiheadAttention(
                embed_dim=input_size, num_heads=params.num_heads, batch_first=True
            )
            # self.encoder = None

        else:
            # in single agent settings, waiting is only allowed when the instance is completed. Thus 
            # it does not require a learnable embedding
            self.dummy = nn.Parameter(torch.randn(1, 1, input_size), requires_grad=False)
            self.encoder = None


    def forward(self, ops_emb: torch.Tensor, td: TensorDict):
        bs, n_jobs, n_ops, emb_dim = ops_emb.shape
        # (bs, ops)
        dummy = self.dummy.expand(bs, 1, -1)  
        if self.encoder is not None:
            attn_mask = ~rearrange(td["op_scheduled"] + td["pad_mask"], "b j o -> b 1 (j o)")
            attn_mask = torch.logical_or(td["done"][..., None], attn_mask)
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            dummy, _ = self.encoder(
                query=dummy, 
                key=rearrange(ops_emb, "b j o e -> b (j o) e"), 
                value=rearrange(ops_emb, "b j o e -> b (j o) e"), 
                attn_mask=~attn_mask  # True means: not attend
            )
        # (bs, 1, emb)
        return dummy


class PositionalEncodingWithOffset(nn.Module):

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, offsets: torch.Tensor = None):
        """
        Positional Encoding with per-head offsets.
        :param x: sequence of embeddings (bs, num_heads, seq_len, d_model)
        :param offsets: per-head sequence offsets (bs, num_heads)
        """
        batch_size, num_heads, length, embed_dim = x.shape
        device = x.device

        if embed_dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(embed_dim)
            )

        # Create a position matrix for each batch instance and head
        position = torch.arange(0, length, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        position = position.expand(batch_size, num_heads, length)  # Shape: (bs, num_heads, seq_len)
        
        # Apply offsets and clamp at 0, using offsets of shape (bs, num_heads)
        if offsets is not None:
            position = (position - offsets.unsqueeze(-1)).clamp(min=0)  # Shape: (bs, num_heads, seq_len)

        # Initialize the positional encoding tensor
        pe = torch.zeros(batch_size, num_heads, length, embed_dim, device=device)

        # Compute the div_term only once (shared across batch and heads)
        div_term = torch.exp(
            (
                torch.arange(0, embed_dim, 2, device=device, dtype=torch.float)
                * -(math.log(10000.0) / embed_dim)
            )
        )

        # Apply positional encoding to even (sin) and odd (cos) indices
        pe[:, :, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)

        # Apply dropout and return the output
        return self.dropout(x + pe)
    

class Normalization(nn.Module):
    def __init__(self, embed_dim: int, normalization: str):
        super().__init__()

        normalizer_class = {
            "batch": nn.BatchNorm1d, 
            "instance": nn.InstanceNorm1d
        }.get(normalization)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_neurons: List[int] = [64, 32],
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
        input_norm: str = "None",
        output_norm: str = "None",
        bias: bool = True
    ):
        super(MLP, self).__init__()

        assert input_norm in ["Batch", "Layer", "None"]
        assert output_norm in ["Batch", "Layer", "None"]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.lins = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            self.lins.append(nn.Linear(in_dim, out_dim, bias=bias))

        self.input_norm = self._get_norm_layer(input_norm, input_dim)
        self.output_norm = self._get_norm_layer(output_norm, output_dim)

    def forward(self, xs):
        xs = self.input_norm(xs)
        for i, lin in enumerate(self.lins[:-1]):
            xs = lin(xs)
            xs = self.hidden_act(xs)
        xs = self.lins[-1](xs)
        xs = self.out_act(xs)
        xs = self.output_norm(xs)
        return xs

    @staticmethod
    def _get_norm_layer(norm_method, dim):
        if norm_method == "Batch":
            in_norm = nn.BatchNorm1d(dim)
        elif norm_method == "Layer":
            in_norm = nn.LayerNorm(dim)
        elif norm_method == "None":
            in_norm = nn.Identity()  # kinda placeholder
        else:
            raise RuntimeError(
                "Not implemented normalization layer type {}".format(norm_method)
            )
        return in_norm

    def _get_act(self, is_last):
        return self.out_act if is_last else self.hidden_act
