import copy
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from tensordict import TensorDict

from marlprp.env.env import MSPRPEnv
from marlprp.utils.ops import batchify, unbatchify
from marlprp.algorithms.model_args import SelfLabelingParameters
from marlprp.algorithms.base import LearningAlgorithmWithReplayBuffer
from marlprp.algorithms.losses import ce_loss, listnet_loss, calc_adv_weights
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
       
        self.best_reward = float("-inf")
        if model_params.loss == "ce":
            self.loss_fn = ce_loss
        else:
            self.loss_fn = partial(listnet_loss, alpha=model_params.listnet_alpha)

    def _update(self, device, batch_idx: int):
        losses = []
        for sub_td in self.experience_sampler:

            sub_td = sub_td.to(device)
            weight = sub_td.get("_weight", None)

            logp, _, entropy, mask = self.policy.evaluate(sub_td, self.env)

            loss = self.loss_fn(logp, entropy, mask=mask, entropy_coef=self.entropy_coef)

            if self.model_params.priority_key is not None:
                sub_td.set(self.model_params.priority_key, loss.detach().clone())
                self.rb.update_tensordict_priority(sub_td)

            # (bs)
            adv_weights = sub_td.get("adv_weights", None)
            if adv_weights is not None:
                loss = loss * adv_weights

            # aggregate loss: bs -> 1
            if weight is not None:
                loss = (loss * weight).sum() / weight.sum()
            else:
                loss = loss.mean()

            self.manual_opt_step(loss, batch_idx)
            losses.append(loss.detach())

        return torch.stack(losses, dim=0).mean()

    def training_step(self, batch: TensorDict, batch_idx: int):
        
        orig_state = self.env.reset(batch)
        bs = orig_state.size(0)
        device = orig_state.device
        train_rewards = []
        # data gathering loop
        for i in range(0, bs, self.rollout_batch_size):

            next_state = orig_state[i : i + self.rollout_batch_size]
            next_state = batchify(next_state, self.num_starts)
            state_stack = []
            steps = 0
            while not next_state.done.all():

                with torch.no_grad():
                    td = self.policy_old.act(next_state, self.env, return_logp=False)
                    next_state = td.pop("next")

                if not self.model_params.eval_multistep and td["action"].dim() == 2:
                    action = td["action"].clone()
                    aug_bs, n_actions = action.shape
                    td = td.unsqueeze(1).expand(aug_bs, n_actions)
                    td["action"] = action
                    steps += n_actions
                else:
                    td = td.unsqueeze(1)
                    steps += 1

                # add tensordict to buffer
                state_stack.append(td)

            # (bs * #samples, #steps)
            state_stack = torch.cat(state_stack, dim=1)
            # (bs, #samples, #steps)
            states_unbs = unbatchify(state_stack, self.num_starts)
            # (bs * #samples)
            rewards = self.env.get_reward(next_state)
            train_rewards.append(rewards.mean())
            # (bs, #samples)
            rewards = unbatchify(rewards, self.num_starts)
            # (bs)
            best_rew, best_idx = rewards.max(dim=1)
            # (bs)
            best_states = states_unbs.gather(
                1, best_idx[:, None, None].expand(-1, 1, steps)
            ).squeeze(1)
            if False:
                # add advantage
                advantage = best_rew - rewards.mean(1, keepdim=False)
                adv_weights = _calc_adv_weights(advantage)
                best_states["adv_weights"] = adv_weights.unsqueeze(1).expand(*best_states.shape)
            # flatten so that every step is an experience
            best_states = best_states.flatten()
            # filter out steps where the instance is already in terminal state. There is nothing to learn from
            best_states = best_states[~best_states["state"].done]
            # save to memory
            self.rb.extend(best_states)

        train_reward = torch.stack(train_rewards).mean()
        loss = self._update(device, batch_idx)        
        self.log("train/reward", train_reward, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "reward": train_reward}

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        
        tgt_policy_changed = False
        ref_policy_changed = False
        epoch_average = super().on_validation_epoch_end()
        # Check if we're using multiple GPUs (DDP is enabled)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Only rank 0 makes the decision whether to update the reference policy
        if self.global_rank == 0:

            if epoch_average > self.best_reward:
                self.pylogger.info(
                    f"Improved val reward {round(-self.best_reward, 2)} -> {round(-epoch_average, 2)}. " + \
                    "Updating Reference Policy..."
                )
                self.best_reward = epoch_average
                self.policy_old.load_state_dict(copy.deepcopy(self.policy.state_dict()))
                ref_policy_changed = True

            if (self.current_epoch + 1) % self.lookback_intervals == 0:
                self.policy.load_state_dict(self.old_tgt_policy_state)
                self.old_tgt_policy_state = copy.deepcopy(self.policy_old.state_dict())
                tgt_policy_changed = True

        # Synchronize the updated reference policy across all GPUs (only if DDP is enabled)
        if world_size > 1:
            self._sync_policies(ref_policy_changed, tgt_policy_changed)

    def _sync_policies(self, ref_policy_changed, tgt_policy_changed):
        # Synchronize the reference policy parameters across all GPUs
        if ref_policy_changed:
            for param in self.policy_old.parameters():
                dist.broadcast(param.data, src=0)
        # Synchronize the target policy parameters across all GPUs
        if tgt_policy_changed:
            for param in self.policy.parameters():
                dist.broadcast(param.data, src=0)

    def on_fit_start(self) -> None:
        # make sure to set the right reward structure
        self.env.stepwise_reward = False

    def state_dict(self, *args, **kwargs):
        # Get the original state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Augment with metadata
        state_dict["best_reward"] = self.best_reward
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Extract metadata if it exists
        self.best_reward = state_dict.pop("best_reward", self.best_reward)
        # Load the remaining state_dict
        super().load_state_dict(state_dict, *args, **kwargs)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.rb.empty()
        torch.cuda.empty_cache()

        # max_bs = max(self.rollout_batch_size * self.num_starts, self.mini_batch_size)
        # bs_in_op_block_attn = max_bs * self.env.get_problem_size(batch).num_jobs
        # if bs_in_op_block_attn > MAX_BATCH_SIZE and hasattr(self.policy.encoder, "use_block_attn"):
        #     self.policy.encoder.use_block_attn = False
        #     self.policy_old.encoder.use_block_attn = False

    # def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
    #     super().on_train_batch_end(outputs, batch, batch_idx)
    #     max_mem = torch.cuda.max_memory_allocated()
    #     mem_available = torch.cuda.mem_get_info()


