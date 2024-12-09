import torch
import torch.nn as nn
from tensordict import TensorDict
from rl4co.envs.common import RL4COEnvBase

from marlprp.utils.utils import Registry
from marlprp.env.env import MSPRPEnv
from marlprp.env.instance import MSPRPState
from marlprp.models.encoder import MatNetEncoder
from marlprp.models.policy_args import PolicyParams
from marlprp.models.decoder.base import BaseDecoder
from marlprp.models.decoder.multi_agent import HierarchicalMultiAgentDecoder


policy_registry = Registry()


class RoutingPolicy(nn.Module):

    def __init__(self, model_params: PolicyParams):
        super().__init__()  
        self.encoder: nn.Module = ...
        self.decoder: BaseDecoder = ...
        self.critic = None
        self.stepwise_encoding: bool = model_params.stepwise_encoding


    @classmethod
    def initialize(cls, params: PolicyParams):
        PolicyCls = policy_registry.get(params.policy)
        return PolicyCls(params)


    def forward(
            self, 
            td: TensorDict, 
            env: RL4COEnvBase, 
            return_actions: bool = False
        ):
        # encoding once
        embeddings = self.encoder(td)
        # pre decoding
        td, env, embeddings = self.decoder.pre_decoding_hook(td, env, embeddings)

        while not td["done"].all():
            # autoregressive decoding
            td = self.decoder(embeddings, td, env)
            td = env.step(td)["next"]
            if self.stepwise_encoding:
                embeddings = self.encoder(td)

        # gather all logps
        log_ps, actions, td, env = self.decoder.post_decoding_hook(td, env)
        # prepare return td
        td.update({
            "reward": env.get_reward(td),
            "log_likelihood": log_ps.sum(1),
        })

        if return_actions:
            td.set("actions", actions)

        return td

    def act(self, state: MSPRPState, env: MSPRPEnv, return_logp: bool = True):
        embeddings = self.encoder(state)
        state, env, embeddings = self.decoder.pre_decoding_hook(
            state, env, embeddings
        )
        td = self.decoder(embeddings, state, env, return_logp=return_logp)
        return td
    
    def evaluate(self, td: TensorDict, env):
        state: MSPRPState = td["state"]
        actions: TensorDict = td["action"]
        # Encoder: get encoder output and initial embeddings from initial state
        embeddings = self.encoder(state)

        # pred value via the value head
        if self.critic is not None:
            value_pred = self.critic(embeddings, state)
        else:
            value_pred = None
        # pre decoder / actor hook
        state, env, embeddings = self.decoder.pre_decoding_hook(
            state, env, embeddings
        )
        action_logprobs, entropies, mask = self.decoder.get_logp_of_action(embeddings, actions, state)

        return action_logprobs, value_pred, entropies, mask

    def set_decode_type(self, decode_type, **kwargs):
        self.decoder._set_decode_strategy(decode_type, **kwargs)


    @torch.no_grad()
    def generate(self, td, env=None, phase: str = "train", **kwargs) -> dict:
        is_training = self.training
        self.train(False)
        assert phase != "train", "dont use generate() in training mode"
        with torch.no_grad():
            out = super().__call__(td, env, phase=phase, **kwargs)
        self.train(is_training)
        return out



@policy_registry.register(name="maham")
class MatNetPolicy(RoutingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.encoder = MatNetEncoder(model_params)
        self.decoder = HierarchicalMultiAgentDecoder(model_params)
        self.critic = None
