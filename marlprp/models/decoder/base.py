import abc
from torch import Tensor
import torch.nn as nn
from tensordict import TensorDict
from marlprp.utils.ops import batchify
from marlprp.env.instance import MSPRPState
from marlprp.utils.config import ModelParams
from marlprp.decoding.strategies import DecodingStrategy, get_decoding_strategy


class BaseDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, model_params: ModelParams) -> None:
        super().__init__() 
        self.dec_strategy: DecodingStrategy = None
        self.stepwise_encoding = model_params.stepwise_encoding

    def pre_decoding_hook(self, state: MSPRPState, env, embeddings: TensorDict):
        state, env, num_starts = self.dec_strategy.setup(state, env)
        if num_starts > 1:
            embeddings = batchify(embeddings, num_starts)
        return state, env, embeddings

    def post_decoding_hook(self, state: MSPRPState, env):
        logps, actions, state, env = self.dec_strategy.post_decoder_hook(state, env)
        return logps, actions, state, env

    def _set_decode_strategy(self, decode_type, **kwargs):
        self.dec_strategy = get_decoding_strategy(decode_type, **kwargs)

    @abc.abstractmethod
    def forward(self, embeddings: TensorDict, state: MSPRPState, env, return_logp: bool = False):
        pass

    @abc.abstractmethod
    def get_logp_of_action(self, embeddings, actions: TensorDict, masks: TensorDict, state: MSPRPState):
        pass


class BasePointer(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        super().__init__() 

    @property
    def device(self):
        return next(self.parameters()).device

    @abc.abstractmethod
    def forward(self, embs: TensorDict,  td: TensorDict, **kwargs):
        pass