import copy
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist

from marlprp.env.env import MSPRPEnv
from marlprp.utils.config import TrainingParams, ValidationParams, TestParams
from marlprp.algorithms.base import ManualOptLearningAlgorithm
from marlprp.algorithms.utils import (
    SampleGenerator, 
    make_replay_buffer, 
)
from marlprp.utils.ops import batchify, unbatchify
from marlprp.algorithms.model_args import SelfLabelingParameters



def ce_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        **kwargs
    ):

    bs = logp.size(0)
    if mask is not None:
        logp[mask] = 0
        entropy[mask] = 0
        denom = mask.view(bs, -1).logical_not().sum(1) + 1e-6

    # add entropy penalty 
    loss = torch.clamp(-logp - entropy_coef * entropy, min=0)

    if mask is not None:
        loss = loss.view(bs, -1).sum(1) / denom
    else:
        loss = loss.view(bs, -1).mean(1)

    return loss


def listnet_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        alpha: float = 0.2,
        **kwargs
    ):
    """ListNet inspired loss. This loss assumes that the logps are ordered corresponding to the
    order the machinens sampled actions during experience collection. The loss enforces a precendence
    by weighting machines that sampled first stronger. This makes intuitive sense, because these agents
    disrupted the sampling space of succeeding machines
    """
    bs = logp.size(0)
    # (bs, num_actions)
    logp = logp.view(bs, -1)
    
    y_true = torch.ones_like(logp)
    ranks = y_true.cumsum(-1)
    weights = torch.exp(-alpha * (ranks - 1))
    if mask is not None:
        # TODO is this sufficient?
        weights[mask] = 0
        entropy[mask] = 0  # for masked entries, simply add no penalty
    # (bs, num_actions)
    target_dist = (y_true * weights) / weights.sum(dim=-1, keepdims=True)
    # (bs, actions)
    ce_loss = -torch.mul(target_dist, logp)
    loss = torch.clamp(ce_loss - entropy_coef * entropy, min=0)
        
    loss = loss.sum(-1)

    return loss



class SelfLabeling(ManualOptLearningAlgorithm):

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

        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.set_decode_type(
            decode_type="sampling", 
            tanh_clipping=train_params.decoding["tanh_clipping"],
            top_p=train_params.decoding["top_p"],
            temperature=model_params.ref_model_temp
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
            priority_key=model_params.priority_key,
            **model_params.buffer_kwargs
        )

        self.priority_key = model_params.priority_key
        self.num_starts = model_params.num_starts
        self.max_grad_norm = train_params.max_grad_norm
        self.entropy_coef = model_params.entropy_coef
        self.best_reward = float("-inf")
        self.experience_sampler = SampleGenerator(
            rb=self.rb, 
            num_iter=model_params.inner_epochs, 
            num_samples=model_params.num_batches
        )
        self.eval_multistep = model_params.eval_multistep

        if model_params.loss == "ce":
            self.loss_fn = partial(
                ce_loss, 
                entropy_coef=model_params.entropy_coef,
            )
        else:
            self.loss_fn = partial(
                listnet_loss, 
                entropy_coef=model_params.entropy_coef,
                alpha=model_params.listnet_alpha
            )

    def _update(self, device):
        losses = []
        for sub_td in self.experience_sampler:

            sub_td = sub_td.to(device)
            weight = sub_td.get("_weight", None)

            logp, _, entropy, mask = self.policy.evaluate(sub_td, self.env)

            loss = self.loss_fn(logp, entropy, mask=mask)

            if self.priority_key is not None:
                sub_td.set(self.priority_key, loss.detach().clone())
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
            losses.append(loss.detach())

        return torch.stack(losses, dim=0).mean()

    def training_step(self, batch, batch_idx):
        
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

                if not self.eval_multistep and td["action"].dim() == 2:
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
            # add advantage
            advantage = best_rew - rewards.mean(1, keepdim=False)
            if False:
                adv_weights = _calc_adv_weights(advantage)
                best_states["adv_weights"] = adv_weights.unsqueeze(1).expand(*best_states.shape)
            # flatten so that every step is an experience
            best_states = best_states.flatten()
            # filter out steps where the instance is already in terminal state. There is nothing to learn from
            best_states = best_states[~best_states["state"].done]
            # save to memory
            self.rb.extend(best_states)

        train_reward = torch.stack(train_rewards).mean()
        loss = self._update(device)        
        self.rb.empty()
        torch.cuda.empty_cache()
        self.log("train/reward", train_reward, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {
            "loss": loss, 
            "reward": train_reward
        }

    def on_validation_epoch_end(self):
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

        # Synchronize the updated reference policy across all GPUs (only if DDP is enabled)
        if world_size > 1:
            self._sync_reference_policy()
        
    def _sync_reference_policy(self):
        # Synchronize the reference policy parameters across all GPUs
        for param in self.policy_old.parameters():
            dist.broadcast(param.data, src=0)



def _calc_adv_weights(adv: torch.Tensor, temp: float = 1.0, weight_clip: float = 20.0):

    adv_mean = adv.mean()
    adv_std = adv.std()

    norm_adv = (adv - adv_mean) / (adv_std + 1e-5)

    weights = torch.exp(norm_adv / temp)

    weights = torch.minimum(weights, torch.full_like(weights, fill_value=weight_clip))
    return weights