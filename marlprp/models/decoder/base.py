import abc
import torch.nn as nn
from tensordict import TensorDict
from marlprp.utils.ops import batchify
from marlprp.env.env import MSPRPEnv
from marlprp.env.instance import MSPRPState
from marlprp.utils.config import ModelParams
from marlprp.decoding.strategies import DecodingStrategy, get_decoding_strategy


class BaseDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, model_params: ModelParams) -> None:
        super().__init__() 
        self.stepwise_encoding = model_params.stepwise_encoding
        self.dec_strategy: DecodingStrategy = None

    def pre_forward_hook(self, state: MSPRPState, embeddings: TensorDict):
        self.dec_strategy.setup()
        if self.dec_strategy.num_starts > 1:
            state = batchify(state, self.dec_strategy.num_starts)
            embeddings = batchify(embeddings, self.dec_strategy.num_starts)
        return state, embeddings

    def post_forward_hook(self, state: MSPRPState, env: MSPRPEnv):
        logps, actions, state = self.dec_strategy.post_decoder_hook(state, env)
        return logps, actions, state

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