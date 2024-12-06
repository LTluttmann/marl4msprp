import logging
from typing import Union
import torch.nn as nn
import torch
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor


log = logging.getLogger(__name__)

class NormByConstant(nn.Module):
    """torch module to apply a constant norm factor on an input sequence"""
    def __init__(self, const, static_size: int = 0) -> None:
        super().__init__()
        self.static_size = static_size
        self.const = const
        
    def forward(self, x):
        x[...,self.static_size:] /= self.const
        return x

def feature_normalize(data, dim=None):
    keepdim = dim is not None
    mean_feat = torch.mean(data, dim=dim, keepdim=keepdim)
    std_feat = torch.std(data, dim=dim, keepdim=keepdim)
    return (data - mean_feat) / (std_feat + 1e-5)


def get_inner_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def set_decode_type(model, decode_type, *args, **kwargs):
    mode = "training" if model.training else "eval"
    select_best = mode == "eval"
    kwargs["select_best"] = select_best
    log.info(f"Set decoding strategy to {decode_type} in {mode} mode")
    model.set_decode_type(decode_type, *args, **kwargs)


def _batchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def batchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def _unbatchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Undoes batchify operation for Tensordicts as well"""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.rearrange(x, '(r b) ... -> b r ...', r=repeats)` but ~2x faster and supports TensorDicts
    Repeats unbatchify operation `n` times as specified by each shape element
    If shape is a tuple, iterates over each element and unbatchifies that many times to match the tuple shape.

    Example:
    >>> x.shape: [a*b*c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a, b, c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x

def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)


def sample_n_random_actions(td: TensorDict, n: int):
    """Helper function to sample n random actions from available actions. If
    number of valid actions is less then n, we sample with replacement from the
    valid actions
    """
    action_mask = td["action_mask"]
    # check whether to use replacement or not
    n_valid_actions = torch.sum(action_mask[:, 1:], 1).min()
    if n_valid_actions < n:
        replace = True
    else:
        replace = False
    ps = torch.rand((action_mask.shape))
    ps[~action_mask] = -torch.inf
    ps = torch.softmax(ps, dim=1)
    selected = torch.multinomial(ps, n, replacement=replace).squeeze(1)
    selected = rearrange(selected, "b n -> (n b)")
    return selected.to(td.device)
