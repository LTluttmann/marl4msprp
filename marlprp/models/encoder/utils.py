import abc
import torch
import torch.nn as nn

from torch import Tensor
from typing import TypedDict
from tensordict import TensorDict
from marlprp.models.policy_args import TransformerParams


class BaseEncoder(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> TensorDict:
        pass

class MatNetEncoderOutput(TypedDict):
    shelf: Tensor
    sku: Tensor

class ETEncoderOutput(TypedDict):
    shelf: Tensor
    sku: Tensor
    agent: Tensor



class MockTransformer(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.params = params
        self.mlp = nn.Sequential(
            nn.Linear(params.embed_dim, params.feed_forward_hidden),
            nn.GELU(),
            nn.Linear(params.feed_forward_hidden, params.embed_dim),
        )

    def forward(self, x, **kwargs):
        return self.mlp(x)
    



def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
    size = list(src.size())
    size[dim] = dim_size
    out = src.new_zeros(*size)   
    out.index_add_(dim, index, src)      
    return out


def _scatter_softmax(logits, group_ids, num_groups):
    from torch_scatter import scatter_max
    max_per_group = scatter_max(logits, group_ids, dim=0, dim_size=num_groups)
    if torch.is_grad_enabled():
        logits = torch.exp(logits - max_per_group[group_ids])
    else:
        logits.sub_(max_per_group[group_ids])
        logits.exp_()
    del max_per_group
    denom = _scatter_add(logits, group_ids, dim=0, dim_size=num_groups)
    attn = logits / (denom[group_ids] + 1e-9)
    del denom
    return attn