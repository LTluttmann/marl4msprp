import abc
import torch
import torch.nn as nn
from typing import Tuple
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage

from marlprp.env.env import MultiAgentEnv
from marlprp.env.instance import MSPRPState
from marlprp.utils.config import DecodingConfig, PolicyParams
from marlprp.decoding.strategies import DecodingStrategy, get_decoding_strategy


class BaseDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, params: PolicyParams) -> None:
        super().__init__() 
        self.pointer: "BasePointer" = ...
        self.dec_strategy: DecodingStrategy = None
        self.stepwise_encoding = params.stepwise_encoding

    def pre_rollout_hook(self, state: MSPRPState,  env: MultiAgentEnv) -> Tuple[MSPRPState, MultiAgentEnv]:
        # logic to be applied after first encoder forward pass
        state = self.dec_strategy.setup(state, env)
        return state

    def post_rollout_hook(
        self, 
        state: MSPRPState, 
        reward: torch.Tensor, 
        storage: LazyTensorStorage = None
    ):
        state, reward, storage = self.dec_strategy.post_decoding_hook(state, reward, storage)
        return state, reward, storage

    def _set_decode_strategy(self, decoding_params: DecodingConfig):
        self.dec_strategy = get_decoding_strategy(decoding_params)

    @abc.abstractmethod
    def forward(
        self, 
        embeddings: TensorDict, 
        state: MSPRPState, 
        env: MultiAgentEnv, 
        return_logp: bool = False
    ) -> TensorDict:
        pass

    @abc.abstractmethod
    def get_logp_of_action(
        self, 
        embeddings: TensorDict, 
        state: MSPRPState,
        action: torch.Tensor, 
        action_mask: torch.Tensor,
        env: MultiAgentEnv
    ):
        pass

    def compute_cache(self, embeddings: TensorDict):
        self.pointer.compute_cache(embeddings)



class BasePointer(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__() 

    @abc.abstractmethod
    def compute_cache(self, embeddings: TensorDict):
        pass

    @abc.abstractmethod
    def forward(self, embs: TensorDict,  td: TensorDict, **kwargs):
        pass