import math
import torch
from typing import Optional
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler


## Taken from https://github.com/huggingface/transformers/blob/0a4e8e2855927eb08d92631f58eb3ec96a8f6d96/src/transformers/optimization.py#L335
class WarmupScheduler(_LRScheduler):
    def __init__(
        self,
        after_scheduler: _LRScheduler,
        start_value: float,
        duration: int,
        end_value: Optional[float] = None,
        last_epoch: int = -1
    ):
        optimizer = after_scheduler.optimizer
        if end_value is None:
            # Use the initial optimizer lr as the warmup end value
            end_value = optimizer.param_groups[0]['lr']

        self.after_scheduler = after_scheduler
        self.warmup_start_value = start_value
        self.warmup_end_value = end_value
        self.warmup_duration = duration
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_duration:
            # Linear warmup calculation
            warmup_factor = (self.last_epoch + 1) / self.warmup_duration
            lr = self.warmup_start_value + warmup_factor * (self.warmup_end_value - self.warmup_start_value)
            return [lr for _ in self.base_lrs]
        else:
            if not self.finished_warmup:
                # Align the after_scheduler’s base_lrs to warmup_end_value
                self.after_scheduler.base_lrs = [self.warmup_end_value for _ in self.base_lrs]
                self.finished_warmup = True
            return self.after_scheduler.get_last_lr()

    def step(self, epoch: Optional[int] = None):
        if self.last_epoch < self.warmup_duration:
            super().step(epoch)
        else:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_duration)
            self.last_epoch += 1



def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    current_step = current_step + 1
    if current_step <= num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)

def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: Optional[float] = None,
    min_lr_rate: Optional[float] = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# from RL4CO
def get_pytorch_lr_schedulers():
    """Get all learning rate schedulers from `torch.optim.lr_scheduler`"""
    return torch.optim.lr_scheduler.__all__


def create_scheduler(
    optimizer: Optimizer, scheduler_name: str, scheduler_kwargs: dict = None, warmup: dict = None
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create scheduler for optimizer. If `scheduler_name` is not found, raise ValueError."""
    scheduler_kwargs = scheduler_kwargs or {}
    if scheduler_name in get_pytorch_lr_schedulers():
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
    elif scheduler_name == "cosine":
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, **scheduler_kwargs)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found.") 

    if warmup is not None:
        lr_scheduler = WarmupScheduler(
            lr_scheduler, 
            **warmup
        )
    return lr_scheduler