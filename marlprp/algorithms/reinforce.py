import torch
import torch.nn as nn
from torch.optim import Adam

from marlprp.algorithms.baselines import Baseline, WarmupBaseline
from marlprp.algorithms.base import LearningAlgorithm
from marlprp.algorithms.utils import RewardScaler
from marlprp.algorithms.model_args import ReinforceParams
from marlprp.env.env import MSPRPEnv
from marlprp.utils.config import (
    TrainingParams, 
    ValidationParams, 
    TestParams, 
    save_config_to_dict
)


class REINFORCE(LearningAlgorithm):

    name = "reinforce"

    def __init__(
        self, 
        env: MSPRPEnv,
        policy: nn.Module,
        model_params: ReinforceParams,
        train_params: TrainingParams,
        val_params: ValidationParams,
        test_params: TestParams
    ) -> None:
        
        super().__init__(
            env=env,
            policy=policy,
            model_params=model_params,
            train_params=train_params,
            val_params=val_params,
            test_params=test_params
        )

        self.model_params = model_params
        self.baseline = self.get_baseline()
        self.scaler = RewardScaler(scale=model_params.reward_scale)

    def get_baseline(self) -> Baseline:
        bl_warmup_epochs = self.model_params.bl_warmup_epochs
        self.pylogger.info("initialize baseline: %s" % self.model_params.baseline)
        baseline = Baseline.initialize(self.model_params.baseline, model_params=self.model_params)

        if bl_warmup_epochs > 0:
            self.pylogger.info("...and warming it up for %s epochs" % bl_warmup_epochs)
            baseline = WarmupBaseline(baseline, self.model_params)

        self.pylogger.info("Setup baseline")
        baseline.setup(self.policy)
        
        return baseline
    
    def _get_optimizer(self):

        optimizer_kwargs = save_config_to_dict(self.train_params.optimizer_kwargs)
        policy_lr = optimizer_kwargs.pop("policy_lr")
        baseline_lr = optimizer_kwargs.pop("baseline_lr", None)
        baseline_lr = baseline_lr or policy_lr

        policy_params = [{'params': self.policy.parameters(), 'lr': policy_lr}]
        baseline_params = (
            [{'params': self.baseline.get_learnable_parameters(), 'lr': baseline_lr}] 
            if len(self.baseline.get_learnable_parameters()) > 0 else []
        )
        optimizer = Adam(policy_params + baseline_params, **optimizer_kwargs)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        td = self.env.reset(batch)
        # Full forward pass through the dataset
        td_out = self.policy(td, self.env)
        reward = td_out["reward"]

        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = self.baseline.eval(batch, reward, self.env)
        advantage = self.scaler(reward - bl_val)
        policy_loss = -torch.mean(advantage * td_out["log_likelihood"])
        loss = policy_loss + bl_loss
        self.log("train/reward", reward.mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        return {
            "loss": loss,
            "reward": reward,
            "policy_loss": policy_loss,
            "bl_loss": bl_loss
        }
           
    def on_validation_epoch_end(self):
        val_reward = self._get_global_validation_reward()
        self.baseline.epoch_callback(self.policy, val_reward)