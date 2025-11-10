import gc
import copy
import torch
import torch.nn as nn

from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage

from marlprp.env.env import MultiAgentEnv
from marlprp.env.instance import MSPRPState
from marlprp.algorithms.model_args import SelfLabelingParameters
from marlprp.utils.ops import batchify, unbatchify, gather_by_index
from marlprp.algorithms.base import LearningAlgorithmWithReplayBuffer
from marlprp.algorithms.losses import ce_loss, simple_listnet_loss, calc_adv_weights
from marlprp.utils.config import TrainingParams, ValidationParams, TestParams


_float = torch.get_default_dtype()


class SelfLabeling(LearningAlgorithmWithReplayBuffer):

    name = "sl"

    def __init__(
        self, 
        env: MultiAgentEnv,
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

        self.num_starts = model_params.ref_model_decoding.num_samples
        self.penalty_coef = self.setup_parameter(model_params.penalty_coef)
        self.entropy_coef = self.setup_parameter(model_params.entropy_coef)
        self.lookback_intervals = self.setup_parameter(
            model_params.lookback_intervals,
            dtype=lambda x: int(x) or float("inf")
        )

        if model_params.loss == "ce":
            self.loss_fn = ce_loss
        else:
            self.loss_fn = simple_listnet_loss

        self.update_after_every_batch = model_params.update_after_every_batch
        if self.update_after_every_batch:
            # if we update after every batch, we always need to flush the buffer afterwards, otherwise
            # examples of earlier batches are seen more often during training then thoss of later batches
            self.always_clear_buffer = True
        else:
            self.always_clear_buffer = model_params.always_clear_buffer
        # if we do not set a fixed number of agents, we need to reinitialize the rb storage to avoid shape issues
        self.reinitialize_rb_storage = self.env.params.num_agents is None

    def _update_rb_sampler_probs(self):
        if len(self.rb_ensamble.storage) == 1:
            return
        rb_distribution = torch.tensor([len(x) for x in self.rb_ensamble.storage], device=self.device)
        rb_distribution = torch.where(rb_distribution == 0, -torch.inf, rb_distribution)
        rb_distribution = torch.softmax(rb_distribution / 1000, dim=0)
        self.rb_ensamble.sampler.p = rb_distribution

    def get_buffer_id(self, sub_td):
        return list(self.rbs.keys())[sub_td["index"]["buffer_ids"][0]]

    def _update(self):
        losses = []
        entropies = []
        self._update_rb_sampler_probs()
        for _ in range(self.num_training_batches):
            sub_td = self.rb_ensamble.sample().clone().to(self.device).squeeze(0)
            buffer_id = self.get_buffer_id(sub_td)
            # get logp of target policy for pseudo expert actions 
            logp, _, entropy, mask = self.policy.evaluate(sub_td, self.env)
            # (bs)
            loss = self.loss_fn(logp, entropy, mask=mask, entropy_coef=self.entropy_coef)
            # (bs)
            adv_weights = sub_td.get("adv_weights", None)
            if adv_weights is not None:
                loss = loss * adv_weights
            # (bs)
            weight = sub_td.get("_weight", None)
            # aggregate loss: bs -> 1
            if weight is not None:
                loss = (loss * weight).sum() / weight.sum()
            else:
                loss = loss.mean()
            self.manual_opt_step(loss, buffer_id)
            losses.append(loss.detach())
            entropies.append(entropy.mean())

        loss = torch.stack(losses, dim=0).mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        entropy = torch.stack(entropies, dim=0).mean()
        self.log("train/entropy", entropy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.always_clear_buffer:
            self.clear_buffer(reinitialize=self.reinitialize_rb_storage)


    @torch.no_grad
    def _collect_experience(self, next_state: MSPRPState, instance_id: str):
        # init storage and perform rollout
        experience_buffer = LazyTensorStorage(self.env.max_num_steps, device=self.model_params.buffer_storage_device)
        done_states, rewards, experience_buffer = self.ref_policy(next_state, self.env, storage=experience_buffer)
        # (bs, #samples, #steps)
        experience_buffer = unbatchify(experience_buffer, self.num_starts)
        # (bs, #samples)
        rewards = unbatchify(rewards, self.num_starts).to(experience_buffer.device)
        steps = (~experience_buffer["state"].done).sum(-1).to(_float)
        # (bs); add step penalty and very small noise to randomize tie breaks
        best_idx = torch.argmax(rewards - self.penalty_coef * steps + torch.rand_like(rewards) * 1e-9, dim=1)
        best_reward = gather_by_index(rewards, best_idx)
        # store rollout results
        self.max_rewards.append(best_reward)
        self.train_rewards.append(rewards.mean(1))
        self.reward_std.append(rewards.std(1) / (-rewards.mean(1) + 1e-6))
        self.avg_steps.append(steps.mean())

        # (bs)
        best_trajectories = gather_by_index(experience_buffer, best_idx, dim=1)

        if self.model_params.use_advantage_weights:
            advantage = best_reward - rewards.mean(1, keepdim=False)
            adv_weights = calc_adv_weights(advantage, temp=2.5, weight_clip=10.)
            best_trajectories["adv_weights"] = adv_weights.unsqueeze(1).expand(*best_trajectories.shape)

        # flatten so that every step is an experience
        best_trajectories = best_trajectories.reshape(-1).contiguous()
        # filter out steps where the instance is already in terminal state. There is nothing to learn from
        best_trajectories = best_trajectories[~best_trajectories["state"].done]
        if not self.model_params.eval_multistep and self.ref_policy.params.is_multiagent_policy:
            # if we have a multi-agent policy (generating M action per step) but want to train the model only
            # on a single action, we need to flatten the agent index in the batch dimension 
            best_trajectories = self.flatten_multi_action_td(best_trajectories)
        if hasattr(self.env, "augment_states"):
            best_trajectories = self.env.augment_states(best_trajectories)
        # save to buffer
        self.rbs[instance_id].extend(best_trajectories)


    def training_step(self, batch: TensorDict, batch_idx: int):
        batch, instance_id = batch
        orig_state = self.env.reset(batch)
        bs = orig_state.size(0)

        if isinstance(self.rollout_batch_size, (dict, DictConfig)):
            rollout_batch_size = self.rollout_batch_size.get(instance_id)
        else:
            rollout_batch_size = self.rollout_batch_size

        # data gathering loop
        for i in range(0, bs, rollout_batch_size):
            next_state = orig_state[i : i + rollout_batch_size]
            self._collect_experience(next_state, instance_id)

        if self.trainer.is_last_batch or self.update_after_every_batch:
            self._update()


    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        
        _, improved = super().on_validation_epoch_end()

        if improved:
            self.pylogger.info("Updating Reference Policy...")
            self.ref_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

            if not self.always_clear_buffer:
                self.clear_buffer(reinitialize=self.reinitialize_rb_storage)

        if (
            ((self.current_epoch + 1) % self.lookback_intervals == 0) and 
            (self.current_epoch + self.lookback_intervals < self.trainer.max_epochs)
        ):
            self.policy.load_state_dict(copy.deepcopy(self.old_tgt_policy_state))
            self.old_tgt_policy_state = copy.deepcopy(self.ref_policy.state_dict())

        if self.model_params.ref_policy_warmup == (self.current_epoch + 1):
            self._reset_target_policy()
  
    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self.avg_steps = []
        self.reward_std = []
        self.train_rewards = []
        self.max_rewards = []
        if "ffsp" in self.env.name:
            self.n_skip_tok = []

    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        avg_steps = torch.stack(self.avg_steps, 0).mean()
        reward_std = torch.cat(self.reward_std, 0).mean()
        train_reward = torch.cat(self.train_rewards, 0).mean()
        max_rewards = torch.cat(self.max_rewards, 0).mean()
        self.log("train/avg_steps", avg_steps, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_std", reward_std, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_mean", train_reward, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/max_rewards", max_rewards, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if "ffsp" in self.env.name:
            n_skip_tok = torch.stack(self.n_skip_tok, 0).mean()
            self.log("train/n_skip_tok", n_skip_tok, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def _reset_target_policy(self):
        for layer in self.policy.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def flatten_multi_action_td(self, td: TensorDict):
        """This function flattens out the 'action dimension' in multi agent settings. This means,
        every action from a multi agent policy will be stored as a seperate experience in the
        resulting tensordict."""
        if not (td["action"]["idx"].dim() == 2 and td["action"]["idx"].size(1) > 1):
            # tensordict is already flat
            return td
        assert td.dim() == 1
        flat_td = td.clone()
        bs, n_actions = td["action"]["idx"].shape
        action: TensorDict = flat_td["action"].clone()
        action.batch_size = (bs, n_actions)
        # (n_actions, bs)
        flat_td = flat_td.unsqueeze(0).expand(n_actions, bs).contiguous()
        flat_td["action"] = action.permute(1, 0).contiguous()
        flat_td = flat_td.view(bs * n_actions)
        return flat_td
    