import torch
import numpy as np

from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Callable, Union, Type, List

from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    ReplayBuffer
)

Numeric = Union[int, float]


def make_replay_buffer(
        buffer_size: int, 
        device="cpu", 
        priority_key: str = None,
        prefetch: int = None,
        priority_alpha: float = 0.2,
        priority_beta: float = 0.1,
        pin_memory: bool = True
    ) -> ReplayBuffer:
    
    if device == "cpu":
        storage = LazyMemmapStorage(buffer_size, device="cpu")
    else:
        storage = LazyTensorStorage(buffer_size, device=device)

    if priority_key is None:
        rb = TensorDictReplayBuffer(
            storage=storage,
            pin_memory=pin_memory,
            prefetch=prefetch,
        )

    else:
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=priority_alpha, # how much to use prioritization
            beta=priority_beta, # how much to correct for bias through importance sampling
            storage=storage,
            priority_key=priority_key,
            pin_memory=pin_memory,
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


class GradNormAccumulator:
    """
    Accumulates gradients from multiple batches, each potentially representing a
    different difficulty level in a curriculum learning setup.

    This class ensures each accumulated batch has an equal influence on the final
    gradient by normalizing the gradient of each batch to a unit norm before
    summation. The final gradient assigned to the model is the average of these
    normalized gradients.

    Args:
        model (torch.nn.Module): The model whose gradients are to be accumulated.
        min_norm (float): A minimum norm value to prevent division by zero or
                          by very small numbers. Gradients with a norm less than
                          this will be scaled by `1.0 / min_norm`. Defaults to 1.0.
    """
    def __init__(self, model: torch.nn.Module, min_norm: float = 1.0):
        self.model = model
        self.min_norm = min_norm
        # Stores the list of normalized gradient tensors for each parameter.
        # The outer list corresponds to model parameters, the inner list
        # to accumulated gradients from different instance_ids.
        self._grad_buffers: List[List[torch.Tensor]] = [
            [] for _ in self.model.parameters()
        ]
        self._num_accumulations = 0

    def _get_grad_norm(self) -> float:
        """Calculates the total L2 norm of the current gradients in the model."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _normalize_and_store(self):
        """Normalizes current model grads and stores them in the buffer."""
        norm = self._get_grad_norm()
        # Prevent division by a very small number
        scale = 1.0 / max(norm, self.min_norm)

        for i, p in enumerate(self.model.parameters()):
            if p.grad is not None:
                # Normalize and store a detached clone
                norm_grad = p.grad.data * scale
                self._grad_buffers[i].append(norm_grad.detach().clone())

    def accumulate(self):
        """
        Normalizes the current gradients on the model, stores them, and then
        clears the model's gradients to prepare for the next backward pass.
        """
        self._normalize_and_store()
        self._num_accumulations += 1
        
        # Important: clear gradients after saving for the next batch
        self.model.zero_grad()

    def average_and_assign(self):
        """
        Averages the accumulated gradients and assigns them back to the model's
        .grad attribute. Resets the accumulator for the next cycle.
        """
        if self._num_accumulations == 0:
            return

        for i, p in enumerate(self.model.parameters()):
            # Only process parameters that have gradients
            if not self._grad_buffers[i]:
                continue
            
            # Stack and average the normalized gradients for this parameter
            avg_grad = torch.stack(self._grad_buffers[i], dim=0).mean(dim=0)
            
            if p.grad is None:
                p.grad = avg_grad
            else:
                p.grad.data.copy_(avg_grad)

        # Reset for the next accumulation cycle
        self.reset()

    def reset(self):
        """Clears all accumulated gradients and resets the counter."""
        self._grad_buffers = [[] for _ in self.model.parameters()]
        self._num_accumulations = 0
        
    def __len__(self) -> int:
        """Returns the number of gradients accumulated so far."""
        return self._num_accumulations


def environment_distribution(env_sizes, current_epoch, epochs_until_uniform, temperature=1.0):
    """
    Compute a probability distribution over environments, shifting from size-based to uniform.

    Args:
        env_sizes (list or np.array): List of environment sizes.
        current_epoch (int): Current epoch number.
        max_epochs (int): Total number of training epochs.
        temperature (float): Controls initial preference for small environments.
        beta (float): Controls transition smoothness.

    Returns:
        np.array: Probability distribution over environments.
    """
    env_sizes = np.array(env_sizes)

    # Step 1: Compute initial probabilities favoring small environments
    initial_probs = np.exp(-env_sizes / temperature)
    initial_probs /= initial_probs.sum()  # Normalize to form a distribution

    # Step 2: Compute uniform distribution
    uniform_probs = np.ones_like(env_sizes) / len(env_sizes)

    # Step 3: Compute scheduling coefficient (alpha)
    if current_epoch < epochs_until_uniform:
        alpha = np.linspace(0, 1, epochs_until_uniform)[current_epoch]
    else:
        alpha = 1

    # Step 4: Interpolate between the two distributions
    final_probs = (1 - alpha) * initial_probs + alpha * uniform_probs

    # Ensure normalization (to account for numerical issues)
    final_probs /= final_probs.sum()

    return final_probs


def get_environment_distribution_for_current_epoch(env_sizes, current_epoch, epochs_until_uniform, **kwargs):
    if current_epoch < epochs_until_uniform:
        all_dists = generate_env_curriculum(env_sizes, epochs_until_uniform, **kwargs)
        return all_dists[current_epoch]
    else:
        return np.ones_like(env_sizes) / len(env_sizes)
    

lru_cache(maxsize=1)
def generate_env_curriculum(
    env_sizes: list[int], 
    steps_until_uniform: int,     
    uniformity_threshold: float = 0.3, # Wie gleichmäßig am Ende (P_best/P_worst - 1)
    dominance_ratio_threshold: float = 200.0, # Wie dominant die kleinste Größe am Start sein soll (P_min/P_second_min)
    curve_factor: float = 2.0 # Form der Temperaturkurve
):
    env_sizes = np.array(env_sizes)
    sizes_scaled = (env_sizes - env_sizes.min()) / (env_sizes.max() - env_sizes.min())

    scores = -sizes_scaled.astype("float")
    S_min_val = sizes_scaled[0]
    S_second_min_val = sizes_scaled[1]

    score_diff_for_Tmin = S_second_min_val - S_min_val

    log_dominance_factor = np.log(dominance_ratio_threshold)
    if log_dominance_factor <= 1e-9: # Sicherheit gegen Division durch Null
        log_dominance_factor = 1e-9
    T_min_auto = score_diff_for_Tmin / log_dominance_factor

    # --- Automatische Bestimmung von T_max ---
    S_max_val = sizes_scaled[-1]
    score_diff_for_Tmax = S_max_val - S_min_val

    log_uniformity_factor = np.log(1 + uniformity_threshold)
    if log_uniformity_factor <= 1e-9:
        log_uniformity_factor = 1e-9

    T_max_auto = score_diff_for_Tmax / log_uniformity_factor

    temps = T_min_auto + (T_max_auto - T_min_auto) * (np.linspace(0, 1, steps_until_uniform) ** curve_factor)

    exp_scaled_scores = np.exp(scores[None] / temps[:, None])

    probs = exp_scaled_scores / exp_scaled_scores.sum(1, keepdims=True)
    return probs


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
