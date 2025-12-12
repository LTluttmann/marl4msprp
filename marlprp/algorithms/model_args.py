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
    entropy_coef: float = 0.0
    loss: str = "ce"
    listnet_alpha: float = 0.0
    num_starts: int = 128
    lookback_intervals: int = None
    always_clear_buffer: bool = True
    update_after_every_batch: bool = True
    use_advantage_weights: bool = False
    ref_policy_warmup: int = 0
    penalty_coef: float = 0.005

    def __post_init__(self):
        super().__post_init__()

        if self.listnet_alpha != 0:
            self.loss = "listnet_weighted"
        if self.num_starts is not None and self.ref_model_num_starts is None and self.ref_model_num_decoding_samples is None:
            # ref_model_num_starts not set properly but num_starts is used instead
            self.ref_model_num_decoding_samples = self.num_starts
        if self.ref_model_num_augment is None:
            # if not set, use num_augment from environment
            self.ref_model_num_augment = self.policy.env.num_augment
        self.ref_model_select_best = False  # handle selection logic in model
        self.ref_model_num_strategies = getattr(self.policy, "num_strategies", None)
        assert not self.ref_model_decode_type == "greedy", "Greedy decoding is not available in Self-Labeling"
        assert self.ref_model_decoding.num_samples > 1, "We need multiple samples for Self-Labeling"



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