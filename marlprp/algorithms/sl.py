import gc
import copy
from functools import partial
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.distributed as dist
from tensordict import TensorDict

from marlprp.env.env import MSPRPEnv
from marlprp.utils.ops import batchify, unbatchify
from marlprp.algorithms.model_args import SelfLabelingParameters
from marlprp.algorithms.base import LearningAlgorithmWithReplayBuffer
from marlprp.algorithms.losses import ce_loss, simple_listnet_loss, calc_adv_weights
from marlprp.utils.config import TrainingParams, ValidationParams, TestParams


class SelfLabeling(LearningAlgorithmWithReplayBuffer):

    name = "sl"

    def __init__(
        self, 
        env: MSPRPEnv,
        policy: nn.Module,
        model_params: SelfLabelingParameters,
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

        self.model_params: SelfLabelingParameters
        # for lookback / revert operations save the current state
        if self.model_params.lookback_intervals:
            self.old_tgt_policy_state = copy.deepcopy(self.policy.state_dict())

        self.num_starts = self.setup_parameter(model_params.num_starts, dtype=int)
        self.entropy_coef = self.setup_parameter(model_params.entropy_coef)
        self.lookback_intervals = self.setup_parameter(
            model_params.lookback_intervals,
            dtype=lambda x: int(x) or float("inf")
        )
       
        self.best_rewards = {instance: float("-inf") for instance in self.rbs.keys()}

        if model_params.loss == "ce":
            self.loss_fn = ce_loss
        else:
            self.loss_fn = simple_listnet_loss

        self.always_clear_buffer = model_params.always_clear_buffer
        self.update_after_every_batch = model_params.update_after_every_batch

    def _update(self):
        losses = []

        rb_distribution = torch.tensor([len(x) for x in self.rb_ensamble.storage], device=self.device)
        rb_distribution = torch.where(rb_distribution == 0, -torch.inf, rb_distribution)
        rb_distribution = torch.softmax(rb_distribution, dim=0)
        self.rb_ensamble.sampler.p = rb_distribution

        for _ in range(self.num_training_batches):

            sub_td = self.rb_ensamble.sample(self.mini_batch_size)[0]
            sub_td = sub_td.to(self.device)

            weight = sub_td.get("_weight", None)

            logp, _, entropy, mask = self.policy.evaluate(sub_td, self.env)

            loss = self.loss_fn(logp, entropy, mask=mask, entropy_coef=self.entropy_coef)

            # (bs)
            adv_weights = sub_td.get("adv_weights", None)
            if adv_weights is not None:
                loss = loss * adv_weights

            # aggregate loss: bs -> 1
            if weight is not None:
                loss = (loss * weight).sum() / weight.sum()
            else:
                loss = loss.mean()

            self.manual_opt_step(loss)
            losses.append(loss.detach())

        loss = torch.stack(losses, dim=0).mean()
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.always_clear_buffer:
            self.clear_buffer()

    def training_step(self, batch: TensorDict, batch_idx: int):
        batch, instance_id = batch
        orig_state = self.env.reset(batch)
        bs = orig_state.size(0)
        if isinstance(self.rollout_batch_size, (dict, DictConfig)):
            rollout_batch_size = self.rollout_batch_size.get(instance_id, self.train_batch_size)
        else:
            rollout_batch_size = self.rollout_batch_size
        train_rewards = []
        reward_std = []
        # data gathering loop
        for i in range(0, bs, rollout_batch_size):

            next_state = orig_state[i : i + rollout_batch_size]

            next_state = batchify(next_state, self.num_starts)
            state_stack = []
            steps = 0
            while not next_state.done.all():

                with torch.no_grad():
                    td = self.policy_old.act(next_state, self.env, return_logp=False)
                    next_state = td.pop("next")

                if not self.model_params.eval_multistep and td["action"].dim() == 2:
                    action = td["action"].clone()
                    pad_mask = td["state"].agent_pad_mask.clone()
                    aug_bs, n_actions = action.shape
                    td = td.unsqueeze(1).expand(aug_bs, n_actions)
                    td["action"] = action.unsqueeze(-1)
                    td["mask"] = torch.logical_or(td["state"].done, pad_mask)
                    steps += n_actions
                else:
                    td["mask"] = td["state"].done.clone()
                    td = td.unsqueeze(1)
                    steps += 1

                # add tensordict to buffer
                state_stack.append(td)

            # (bs * #samples, #steps)
            state_stack = torch.cat(state_stack, dim=1)
            # (bs, #samples, #steps)
            states_unbs = unbatchify(state_stack, self.num_starts)
            # (bs * #samples)
            rewards = self.env.get_reward(next_state, mode="train")
            # (bs, #samples)
            rewards = unbatchify(rewards, self.num_starts)
            train_rewards.append(rewards.mean(1))
            reward_std.append(rewards.std(1) / (-rewards.mean(1) + 1e-6))
            # (bs)
            best_rew, best_idx = rewards.max(dim=1)
            # (bs)
            best_states = states_unbs.gather(
                1, best_idx[:, None, None].expand(-1, 1, steps)
            ).squeeze(1)

            # flatten so that every step is an experience
            best_states = best_states.flatten()
            # filter out steps where the instance is already in terminal state. There is nothing to learn from
            best_states = best_states[~best_states["mask"]]
            # save to memory
            self.rbs[instance_id].extend(best_states)
            # self.log("train/rb_size", len(self.rb), on_step=True, sync_dist=True)

        reward_std = torch.cat(reward_std, 0).mean()
        train_reward = torch.cat(train_rewards, 0).mean()
        self.log("train/reward_std", reward_std, on_epoch=True, sync_dist=True)
        self.log("train/reward_mean", train_reward, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.trainer.is_last_batch or self.update_after_every_batch:
            self._update()

        dummy_loss = torch.tensor(0.0, device=self.device)  # dummy loss
        return {"loss": dummy_loss, "reward": train_reward}

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        
        improved = False
        epoch_averages = super().on_validation_epoch_end()
        if self.improved_at_least_half(epoch_averages):
            result_string = "\n".join([f"{idx}: {round(-self.best_rewards[idx], 2)} -> {round(-reward, 2)}" for idx, reward in epoch_averages.items()])
            self.pylogger.info(
                f"Improved val reward: \n{result_string} \nUpdating Reference Policy..."
            )
            improved = True
            self.best_rewards.update(epoch_averages)
            self.policy_old.load_state_dict(copy.deepcopy(self.policy.state_dict()))

        if self.always_clear_buffer or improved:
            self.clear_buffer()

        if ((self.current_epoch + 1) % self.lookback_intervals == 0) and (self.current_epoch + self.lookback_intervals < self.trainer.max_epochs):
            self.policy.load_state_dict(copy.deepcopy(self.old_tgt_policy_state))
            self.old_tgt_policy_state = copy.deepcopy(self.policy_old.state_dict())

    def clear_buffer(self):
        self.pylogger.info(f"Emptying replay buffer of size {self.rb_size}")
        for rb in self.rbs.values():
            rb.empty()
        gc.collect()
        torch.cuda.empty_cache()

    def state_dict(self, *args, **kwargs):
        # Get the original state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Augment with metadata
        state_dict["best_rewards"] = self.best_rewards
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Extract metadata if it exists
        self.best_rewards = state_dict.pop("best_rewards", self.best_rewards)
        # Load the remaining state_dict
        super().load_state_dict(state_dict, *args, **kwargs)

    def is_pareto_efficient(self, epoch_averages):
        not_worse = [epoch_reward >= self.best_rewards[instance] for instance, epoch_reward in epoch_averages.items()]
        better = [epoch_reward > self.best_rewards[instance] for instance, epoch_reward in epoch_averages.items()]
        return all(not_worse) and any(better)

    def improved_at_least_half(self, epoch_averages):
        better = [epoch_reward > self.best_rewards[instance] for instance, epoch_reward in epoch_averages.items()]
        return sum(better) / len(epoch_averages) > .5
   
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     super().on_train_batch_end(outputs, batch, batch_idx)
    #     if self.model_params.update_after_every_batch:
    #         self.pylogger.info(f"Emptying replay buffer of size {len(self.rb)}")
    #         self.rb.empty()
    #         torch.cuda.empty_cache()