import torch.nn as nn
from dataclasses import dataclass, field
from typing import Literal, Union
from rl4co.utils import pylogger

from marlprp.utils.config import ModelParams

log = pylogger.get_pylogger(__name__)


@dataclass(kw_only=True)
class PPOParams(ModelParams):
    algorithm: str = "ppo"
    ppo_epochs: int = 1
    mini_batch_size: int = None
    rollout_batch_size: int = None
    buffer_storage_device: Literal["gpu", "cpu"] = "gpu"
    buffer_size: int = 100_000
    reward_scale: str = "scale"
    stepwise_encoding: bool = True
    stepwise_reward: bool = True
    ref_policy_update_steps: int = 1
    entropy_coef: float = 0.01
    vf_loss_coef: float = 0.5

    def __post_init__(self):
        self.policy.use_critic = True
        return super().__post_init__()


@dataclass(kw_only=True)
class SelfLabelingParameters(ModelParams):
    algorithm: str = "sl"
    sl_epochs: int = 1
    num_batches: int = None
    mini_batch_size: Union[float, int] = 1000
    rollout_batch_size: Union[float, int] = 0.25
    buffer_storage_device: Literal["gpu", "cpu"] = "gpu"
    buffer_kwargs: dict = field(default_factory= lambda: {
        "priority_alpha": 0.5,
        "priority_beta": 0.8
    })
    priority_key: str = None
    num_starts: int = 100
    buffer_size: int = 100_000
    stepwise_encoding: bool = True
    # decoding
    sl_temperature: float = 1.0
    entropy_coef: float = 0. # 0.05
    eval_multistep: bool = False
    eval_per_agent: bool = True
    loss: str = "ce"
    listnet_alpha: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.eval_multistep = self.eval_multistep and self.policy.is_multiagent_policy
        self.eval_per_agent = self.eval_per_agent and self.eval_multistep
        self.policy.eval_multistep = self.eval_multistep
        self.policy.eval_per_agent = self.eval_per_agent
        if self.eval_multistep and self.eval_per_agent:
            # softmax is calculated based on agents#
            # thus we need ce loss since this assumes 
            # probs to sum up to one
            self.loss = "ce"
        elif self.eval_multistep and not self.eval_per_agent:
            # softmax is calculated over the entire action space. Technically,
            # we have multiple "correct labels", thus we should use a list ranking
            # loss here
            self.loss = "list"
        else:
            # we evaluate one action per time, both losses are identical
            self.loss = "ce"


@dataclass(kw_only=True)
class ContrastivePolicyLearningParameters(ModelParams):
    algorithm: str = "cpl"
    sl_epochs: int = 1
    num_batches: int = None
    mini_batch_size: Union[float, int] = 1000
    rollout_batch_size: Union[float, int] = 0.25
    buffer_storage_device: Literal["gpu", "cpu"] = "gpu"
    buffer_kwargs: dict = field(default_factory= lambda: {
        "priority_alpha": 0.5,
        "priority_beta": 0.8
    })
    priority_key: str = None
    num_starts: int = 100
    buffer_size: int = 100_000
    stepwise_encoding: bool = True
    # decoding
    sl_temperature: float = 1.0
    entropy_coef: float = 0.05


@dataclass(kw_only=True)
class ReinforceParams(ModelParams):
    algorithm: str = "reinforce"
    bl_warmup_epochs: int = 1
    baseline: Literal["critic", "rollout", "exponential", "pomo"] = "rollout"
    exp_beta: float = 0.8
    num_starts: int = None
    reward_scale: str = None

    def __post_init__(self):
        if self.baseline == "critic" and self.reward_scale is None:
            log.warning(
                """Using a critic baseline in REINFORCE with unscaled rewards.
                Consider scaling them by passing `reward_scale=scale` to the model"""
            )
        return super().__post_init__()