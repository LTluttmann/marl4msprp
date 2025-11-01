from typing import Literal
from dataclasses import dataclass

from marlprp.utils.logger import get_lightning_logger
from marlprp.utils.config import ModelParams, ModelWithReplayBufferParams, MAX_BATCH_SIZE


log = get_lightning_logger(__name__)


@dataclass(kw_only=True)
class PPOParams(ModelWithReplayBufferParams):
    algorithm: str = "ppo"
    reward_scale: str = "scale"
    stepwise_reward: bool = True
    ref_policy_update_steps: int = 1
    entropy_coef: float = 0.01
    vf_loss_coef: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.policy.use_critic = True


@dataclass(kw_only=True)
class SelfLabelingParameters(ModelWithReplayBufferParams):
    algorithm: str = "sl"
    num_starts: int = 100
    entropy_coef: float = 0. # 0.05
    loss: str = None
    listnet_alpha: float = 0.0
    lookback_intervals: int = None
    always_clear_buffer: bool = False
    update_after_every_batch: bool = False
    penalty_coef: float = 0.005
    
    def __post_init__(self):
        super().__post_init__()

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