import abc
from typing import Tuple, TypedDict

import torch
from torch.distributions import Categorical
from tensordict import TensorDict
from einops import rearrange

from marlprp.env.env import MSPRPEnv
from marlprp.models.policy_args import TransformerParams
from marlprp.utils.ops import gather_by_index
from marlprp.models.decoder.base import BaseDecoder

from .mlp import JobMachinePointer


__all__ = []

class SingleAgentAction(TypedDict):
    idx: torch.Tensor  # the flat (in case of combined action space) action
    job: torch.Tensor
    machine: torch.Tensor




class BaseSingleAgentDecoder(BaseDecoder):
    def __init__(self, pointer, params: TransformerParams) -> None:
        super().__init__(params)
        self.pointer = pointer

    def forward(
        self, 
        embeddings: TensorDict, 
        td: TensorDict, 
        env: MSPRPEnv, 
        return_logp: bool = False
    ):
        
        logits, mask = self.pointer(embeddings, td)
        logp, mask = self._logits_to_logp(logits, mask)
        td = self.dec_strategy.step(logp, mask, td)
        action = self._translate_action(td, env)
        if return_logp:
            logp = self.dec_strategy.logp["action"][-1]
            td["logprobs"] = logp
        # insert action td
        td.set("action", action)
        return td
    
    def get_logp_of_action(self, embeddings: TensorDict, td: TensorDict):
        logits, mask = self.pointer(embeddings, td)
        logp, _ = self._logits_to_logp(logits, mask)
        action_logp = gather_by_index(logp, td["action"]["idx"])
        dist_entropys = Categorical(logp.exp()).entropy()
        return action_logp, dist_entropys, None  # no mask due to padding in single agent settings
    
    @abc.abstractmethod
    def _logits_to_logp(self, logits, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    @abc.abstractmethod
    def _translate_action(self, td: TensorDict, env: MSPRPEnv) -> SingleAgentAction:
        pass
    

class JobMachineMLPDecoder(BaseSingleAgentDecoder):
    def __init__(self, params: TransformerParams) -> None:
        pointer = JobMachinePointer(params)
        super().__init__(pointer, params)

    def _logits_to_logp(self, logits, mask):
        mask = rearrange(mask, "b m j -> b (j m)")
        logits = rearrange(logits, "b m j -> b (j m)")
        logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
        return logp, mask

    def _translate_action(self, td: TensorDict, env: MSPRPEnv) -> SingleAgentAction:
        # translate and store action
        action = td["action"]
        selected_machine = action % env.num_mas
        selected_job = action // env.num_mas
        action = TensorDict(
            {
                "idx": action, 
                "job": selected_job, 
                "machine": selected_machine,
            },
            batch_size=td.batch_size
        )
        return action