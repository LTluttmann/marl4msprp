import torch
from dataclasses import dataclass
from typing import Optional, Callable, Union, Type

from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    ListStorage,
    TensorDictReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    ReplayBuffer
)

Numeric = Union[int, float]


def make_replay_buffer(
        buffer_size: int, 
        device="cpu", 
        priority_key: str = None,
        priority_alpha: float = 0.2,
        priority_beta: float = 0.1
    ) -> ReplayBuffer:
    
    if device == "cpu":
        storage = LazyMemmapStorage(buffer_size, device="cpu")
        prefetch = 5
    else:
        storage = ListStorage(buffer_size)
        prefetch = None

    if priority_key is None:
        rb = TensorDictReplayBuffer(
            storage=storage,
            # batch_size=batch_size,
            # sampler=SamplerWithoutReplacement(drop_last=True, shuffle=True),
            pin_memory=False,  # TODO use pin_memory https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
            prefetch=prefetch,
        )

    else:
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=priority_alpha, # how much to use prioritization
            beta=priority_beta, # how much to correct for bias through importance sampling
            storage=storage,
            #batch_size=batch_size,
            priority_key=priority_key,
            pin_memory=False,
            prefetch=prefetch,
        )

    return rb


# def one_sided_independent_t_test(candidate_rewards, ref_rewards):
#     # Step 1: gather first and second order statistics
#     a_mean = torch.mean(candidate_rewards).item()
#     a_var = torch.var(candidate_rewards, unbiased=True)  # unbiased for sample var

#     b_mean = torch.mean(ref_rewards).item()
#     b_var = torch.var(ref_rewards, unbiased=True) # unbiased for sample var

#     # Step 2: Calculate the t-statistic for a one-sample t-test
#     # 2.1 determine degrees of freedom
#     na = candidate_rewards.size(0)
#     nb = ref_rewards.size(0)
#     df = na + nb - 2
#     # 2.2 numerator
#     dm = a_mean - b_mean
#     # 2.3 pooled variance and denominator
#     pooled_v = (((na-1) * a_var) + ((nb-1) * b_var)) / df
#     denom = torch.sqrt(pooled_v) * math.sqrt(1/na + 1/nb)
#     # t-statistic formula
#     t_stat = dm / denom

#     # Step 3: determine p value
#     cdf = stats.t.cdf(t_stat.cpu().numpy(), df=df)
#     p_value = 1 - cdf
#     return p_value


class RewardScaler:
    """This class calculates the running mean and variance of a stepwise observed
    quantity, like the RL reward / advantage using the Welford online algorithm.
    The mean and variance are either used to standardize the input (scale='norm') or
    to scale it (scale='scale').

    Args:
        scale: None | 'scale' | 'mean': specifies how to transform the input; defaults to None
    """

    def __init__(self, scale: str = None):
        self.scale = scale
        self.count = 0
        self.mean = 0
        self.M2 = 0

    def __call__(self, scores: torch.Tensor):
        if self.scale is None:
            return scores
        elif isinstance(self.scale, int):
            return scores / self.scale
        # Score scaling
        self.update(scores)
        tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
        std = (self.M2 / (self.count - 1)).float().sqrt()
        score_scaling_factor = std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
        if self.scale == "norm":
            scores = (scores - self.mean.to(**tensor_to_kwargs)) / score_scaling_factor
        elif self.scale == "scale":
            scores /= score_scaling_factor
        else:
            raise ValueError("unknown scaling operation requested: %s" % self.scale)
        return scores

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        batch = batch.reshape(-1)
        self.count += len(batch)

        # newvalues - oldMean
        delta = batch - self.mean
        self.mean += (delta / self.count).sum()
        # newvalues - newMeant
        delta2 = batch - self.mean
        self.M2 += (delta * delta2).sum()



@dataclass
class NumericParameter:
    _val: Numeric
    update_coef: Optional[float] = None
    min: Optional[Numeric] = None
    max: Optional[Numeric] = None
    dtype: Type | Callable = None

    def __post_init__(self):
        self.min = self.min or float("-inf")
        self.max = self.max or float("inf")

    @property
    def val(self):
        if self._val is None:
            return None
        val = min(max(self._val, self.min), self.max)
        if self.dtype is not None:
            val = self.dtype(val)
        return val
    
    @val.setter
    def val(self, value):
        self._val = value

    def update(self):
        if self.update_coef is not None:
            self.val = self._val * self.update_coef
        return self