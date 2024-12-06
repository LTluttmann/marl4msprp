import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch.utilities.rank_zero import rank_zero_only

from marlprp.env.env import MSPRPEnv
from marlprp.algorithms.model_args import PPOParams
from marlprp.algorithms.base import ManualOptLearningAlgorithm
from marlprp.algorithms.utils import RewardScaler, make_replay_buffer
from marlprp.utils.config import TrainingParams, ValidationParams, TestParams


class PPO(ManualOptLearningAlgorithm):

    name = "ppo"

    def __init__(
        self, 
        env: MSPRPEnv,
        policy: nn.Module,
        model_params: PPOParams,
        train_params: TrainingParams,
        val_params: ValidationParams,
        test_params: TestParams
    ) -> None:

        env.stepwise_reward = model_params.stepwise_reward

        super().__init__(
            env=env,
            policy=policy,
            model_params=model_params,
            train_params=train_params,
            val_params=val_params,
            test_params=test_params
        )

        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.set_decode_type(
            decode_type="sampling", 
            tanh_clipping=train_params.decoding["tanh_clipping"],
            top_p=train_params.decoding["top_p"],
        )

        rollout_batch_size = model_params.rollout_batch_size or train_params.batch_size
        if isinstance(rollout_batch_size, float):
            rollout_batch_size = int(rollout_batch_size * train_params.batch_size)
        self.rollout_batch_size = min(train_params.batch_size, rollout_batch_size)

        mini_batch_size = model_params.mini_batch_size or train_params.batch_size
        if isinstance(mini_batch_size, float):
            mini_batch_size = int(mini_batch_size * train_params.batch_size)
        
        self.rb = make_replay_buffer(
            buffer_size=model_params.buffer_size, 
            batch_size=mini_batch_size, 
            device=model_params.buffer_storage_device, 
            priority_key=None,
        )

        self.ppo_epochs = model_params.ppo_epochs
        self.max_grad_norm = train_params.max_grad_norm
        self.use_stepwise_reward = model_params.stepwise_reward
        self.scaler = RewardScaler(scale=model_params.reward_scale)
        self.ref_policy_update_steps = model_params.ref_policy_update_steps
        self.vf_loss_coef = model_params.vf_loss_coef
        self.entropy_coef = model_params.entropy_coef

    def update(self, device):
        outs = []
        # PPO inner epoch
        for _ in range(self.ppo_epochs):
            for sub_td in self.rb:
                sub_td = sub_td.to(device)
                previous_reward = sub_td["reward"]
                previous_logp = sub_td["logprobs"]

                logprobs, value_pred, entropy = self.policy.evaluate(sub_td, self.env)

                ratios = torch.exp(logprobs - previous_logp)
                if ratios.dim() == 1:
                    ratios = ratios.unsqueeze(1)

                advantages = previous_reward - value_pred.detach()

                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - 0.2,
                        1 + 0.2,
                    )
                    * advantages
                )
                policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
                # compute value function loss
                value_loss = F.mse_loss(value_pred, previous_reward) # , reduction='none')
                # compute total loss
                loss = policy_action_loss + self.vf_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.max_grad_norm is not None:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.max_grad_norm,
                        gradient_clip_algorithm="norm",
                    )

                opt.step()

                out = {
                    "loss": loss.detach(),
                    "surrogate_loss": policy_action_loss.detach(),
                    "value_loss": value_loss.detach(),
                }

                outs.append(out)

        # Copy new weights into old policy:
        outs = {k: torch.stack([dic[k] for dic in outs], dim=0) for k in outs[0]}
        return outs

    def training_step(self, batch, batch_idx):
        
        orig_td = self.env.reset(batch)
        bs = orig_td.size(0)
        device = orig_td.device

        # data gathering loop
        for i in range(0, bs, self.rollout_batch_size):

            next_td = orig_td[i : i + self.rollout_batch_size]

            while not next_td["done"].all():

                with torch.no_grad():
                    td = self.policy_old.act(next_td, self.env, return_logp=True)

                # get next state
                next_td = self.env.step(td)["next"]

                if self.use_stepwise_reward:
                    # get reward of action
                    reward = next_td["step_reward"]
                    reward = self.scaler(reward)
                    # add reward to prior state
                    td.set("reward", reward)
                    # add tensordict with action, logprobs and reward information to buffer
                    self.rb.extend(td)

                else:
                    raise NotImplementedError("PPO requires stepwise rewards in the current state")

        total_reward = self.env.get_reward(next_td)
        out = self.update(device)
        self.rb.empty()
        torch.cuda.empty_cache()
        out["reward"] = total_reward
        self.log("train/reward", total_reward.mean(), on_step=True, prog_bar=True, on_epoch=True, sync_dist=True)
        return out

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if batch_idx % self.ref_policy_update_steps == 0:
            self.policy_old.load_state_dict(self.policy.state_dict())

