import torch
import torch.nn as nn

from tensordict import TensorDict
from typing import Union, Tuple
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from marlprp.utils.utils import Registry
from marlprp.env.env import MultiAgentEnv
from marlprp.env.instance import MSPRPState
from marlprp.utils.config import DecodingConfig
from marlprp.models.policy_args import PolicyParams
from marlprp.models.decoder.base import BaseDecoder
from marlprp.models.encoder import MatNetEncoder, ETEncoder
from marlprp.models.decoder.multi_agent import HierarchicalMultiAgentDecoder, HierarchicalParcoDecoder
from marlprp.models.decoder.single_agent import HierarchicalSingleAgentDecoder, Hierarchical2dPtrDecoder


policy_registry = Registry()


class RoutingPolicy(nn.Module):

    def __init__(self, params: PolicyParams):
        super().__init__()  
        self.encoder: nn.Module = ...
        self.decoder: BaseDecoder = ...
        self.critic = None
        self.num_starts = None
        self.stepwise_encoding: bool = params.stepwise_encoding


    @classmethod
    def initialize(cls, params: PolicyParams):
        PolicyCls = policy_registry.get(params.policy)
        return PolicyCls(params)

    @property
    def mode(self):
        return "train" if self.training else "val"

    def _setup_storage(self, size, device = "cpu", **kwargs) -> LazyTensorStorage:
        return LazyTensorStorage(size, device=device, **kwargs)
    
    def forward(
        self, 
        problem: MSPRPState, 
        env: MultiAgentEnv, 
        return_logp: bool = False,
        return_trajectories: bool = False,
        storage: LazyTensorStorage = None,
        **storage_kwargs
    ) -> Union[Tuple[MSPRPState, torch.Tensor], Tuple[MSPRPState, torch.Tensor, TensorDict]]:
        """Function for a full Policy Rollout"""
        next_state = problem.clone()
        # pre-rollout hook  
        next_state = self.decoder.pre_rollout_hook(next_state, env)
        # encoding once
        embeddings = self.encoder(next_state)
        if not self.stepwise_encoding:
            # cache decoder computations if not re-encoding each step
            self.decoder.compute_cache(embeddings)

        # setup optional trajectory buffer
        if storage is None and (return_trajectories or return_logp):
            storage = self._setup_storage(env.max_num_steps, **storage_kwargs)

        # generation loop
        step = 0

        while not next_state.done.all():
            # autoregressive decoding
            step_dict = self.decoder(embeddings, next_state, env, return_logp=return_logp)
            next_state = step_dict.pop("next")
            if storage is not None:
                # step_dict["mask"] = step_dict["state"].done.clone()
                storage.set(slice(step, step+1), step_dict.unsqueeze(0).clone())

            if self.stepwise_encoding:
                # optionally, re-encode
                embeddings = self.encoder(next_state)
            # increment step
            step += 1

        # prepare return td
        reward = env.get_reward(next_state)
        # postprocessing
        done_td, reward, storage = self.decoder.post_rollout_hook(next_state, reward, storage)
        # return storage or not
        if storage is not None:
            return done_td, reward, storage
        return done_td, reward
    
    
    def act(self, td, env, **decoder_kwargs) -> TensorDict:
        embeddings = self.encoder(td)
        td = self.decoder(embeddings, td, env, **decoder_kwargs)
        return td
    
    
    def evaluate(self, td: TensorDict, env):
        state: MSPRPState = td["state"]
        actions: TensorDict = td["action"]
        action_masks: TensorDict = td["action_mask"]
        # Encoder: get encoder output and initial embeddings from initial state
        embeddings = self.encoder(state)
        # pred value via the value head
        if self.critic is not None:
            value_pred = self.critic(embeddings, state)
        else:
            value_pred = None

        action_logprobs, entropies, mask = self.decoder.get_logp_of_action(embeddings, state, actions, action_masks, env)

        return action_logprobs, value_pred, entropies, mask

    def set_decode_type(self, decoding_params: DecodingConfig):
        self.decoder._set_decode_strategy(decoding_params)


    @torch.no_grad()
    def generate(self, state: MSPRPState, env=None, **kwargs) -> TensorDict:
        is_training = self.training
        self.train(False)
        out = super().__call__(state, env, **kwargs)
        self.train(is_training)
        return out


@policy_registry.register(name="ham")
class SingleAgentPolicy(RoutingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.encoder = MatNetEncoder(model_params)
        self.decoder = HierarchicalSingleAgentDecoder(model_params)
        self.critic = None


@policy_registry.register(name="et")
class EquityTransformerPolicy(RoutingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.encoder = ETEncoder(model_params)
        self.decoder = HierarchicalSingleAgentDecoder(model_params)
        self.critic = None


@policy_registry.register(name="2dptr")
class TwoDPtrPolicy(RoutingPolicy):
    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.encoder = MatNetEncoder(model_params)
        self.decoder = Hierarchical2dPtrDecoder(model_params)
        self.critic = None



@policy_registry.register(name="parco")
class MultiAgentPolicy(RoutingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.encoder = MatNetEncoder(model_params)
        self.decoder = HierarchicalParcoDecoder(model_params)
        self.critic = None


@policy_registry.register(name="maham")
class MultiAgentPolicy(RoutingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.encoder = MatNetEncoder(model_params)
        self.decoder = HierarchicalMultiAgentDecoder(model_params)
        self.critic = None
