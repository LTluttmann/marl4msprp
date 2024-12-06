import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from typing import Literal, Dict, Type
from tensordict import TensorDict
from rl4co.envs.common import RL4COEnvBase

from marlprp.models.critic import OperationsCritic
from marlprp.models.decoder.base import BaseDecoder
from marlprp.models.decoder import (
    MultiAgentAttnDecoder, 
    MultiAgentMLPDecoder, 
    MultiJobDecoder,
    JobMachineMLPDecoder,
    JobMLPDecoder
)
from marlprp.models.encoder import OperationsEncoder, MatNetEncoder
from marlprp.models.policy_args import PolicyParams
from marlprp.utils.utils import Registry


policy_registry = Registry()


class SchedulingPolicy(nn.Module):

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

    def act(self, td, env, return_logp = True):
        embeddings = self.encoder(td)
        td, env, embeddings = self.decoder.pre_decoding_hook(
            td, env, embeddings
        )
        td = self.decoder(embeddings, td, env, return_logp=return_logp)
        return td
    
    def evaluate(self, td, env):

        # Encoder: get encoder output and initial embeddings from initial state
        embeddings = self.encoder(td)

        # pred value via the value head
        if self.critic is not None:
            value_pred = self.critic(embeddings, td)
        else:
            value_pred = None
        # pre decoder / actor hook
        td, env, embeddings = self.decoder.pre_decoding_hook(
            td, env, embeddings
        )
        action_logprobs, entropies, mask = self.decoder.get_logp_of_action(embeddings, td)

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


@policy_registry.register(name="transformer")
class TransformerPolicy(SchedulingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.encoder = OperationsEncoder(model_params)
        self.decoder = JobMLPDecoder(model_params)
        self.critic = self._critic(model_params)

    def _critic(self, model_params: PolicyParams):
        if not model_params.use_critic:
            critic = None
        else:
            critic = OperationsCritic(model_params)
        return critic



@policy_registry.register(name="matnet")
class MatNetPolicy(SchedulingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.encoder = MatNetEncoder(model_params)
        self.decoder = JobMachineMLPDecoder(model_params)
        self.critic = None


@policy_registry.register(name="marlprp4js")
class marlprpPolicy4JS(SchedulingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.emb_dim = model_params.embed_dim
        self.encoder = OperationsEncoder(model_params)
        self.decoder = MultiJobDecoder(model_params)
        self.critic = None
    

@policy_registry.register(name="marlprp")
class marlprpPolicy(SchedulingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.emb_dim = model_params.embed_dim
        self.encoder = MatNetEncoder(model_params)
        self.decoder = MultiAgentAttnDecoder(model_params)
        self.critic = None


@policy_registry.register(name="marlprp_mlp")
class marlprpMLPPolicy(SchedulingPolicy):

    def __init__(self, model_params: PolicyParams):
        super().__init__(model_params)  
        self.emb_dim = model_params.embed_dim
        self.encoder = MatNetEncoder(model_params)
        self.decoder = MultiAgentMLPDecoder(model_params)
        self.critic = None
    


#######################################################
################ RANDOM POLICIES ######################
#######################################################


def random_jssp_policy(td: TensorDict):
    logits = torch.randn(td["action_mask"].shape)
    mask = td["action_mask"].clone()
    logits_masked = logits.masked_fill(~mask, -torch.inf)
    probs = F.softmax(logits_masked, dim=1)
    action = probs.multinomial(1).squeeze(1)
    td.set("action", action)
    return td


def random_multi_agent_jssp_policy(td: TensorDict):
    """Helper function to select a random action from available actions"""
    logits = torch.randn(td["action_mask"].shape)
    mask = td["action_mask"].clone()
    bs, num_mas, num_jobs_plus_one = mask.shape
    batch_idx = torch.arange(0, bs, device=td.device)

    temp = 1.0
    idle_machines = torch.arange(0, num_mas, device=td.device)[None,:].expand(bs, -1)

    # initialize action buffer
    actions = torch.zeros(size=(bs, num_mas), device=td.device, dtype=torch.long)
    while mask.any():

        logits_masked = logits.masked_fill(~mask, -torch.inf)
        logits_reshaped = rearrange(logits_masked, "b m j -> b (j m)") / temp
        logp = F.softmax(logits_reshaped, dim=1)

        action = logp.multinomial(1).squeeze(1)

        selected_machine = action % num_mas
        selected_job = action // num_mas
        actions[batch_idx, selected_machine] = selected_job

        idle_machines = (
            idle_machines[idle_machines!=selected_machine[:, None]]
            .view(bs, -1)
        )

        mask = mask.scatter(-1, selected_job.view(bs, 1, 1).expand(-1, num_mas, 1), False)
        mask[..., 0] = mask[..., 0].scatter(-1, idle_machines.view(bs, -1), True)
        mask = mask.scatter(-2, selected_machine.view(bs, 1, 1).expand(-1, 1, num_jobs_plus_one), False)

    # actions = torch.stack(actions, dim=1)
    td.set("action", actions)
    return td


def random_fjsp_policy(td: TensorDict):
    bs, nj, nm = td["action_mask"].shape
    # (bs, job, ma)
    job_ma_logits = torch.randn(td["action_mask"].shape)
    mask = td["action_mask"].clone()
    logits_masked = job_ma_logits.masked_fill(~mask, -torch.inf)
    logits_reshapde = rearrange(logits_masked, "b j m -> b (j m)")
    probs = F.softmax(logits_reshapde, dim=1)
    action = probs.multinomial(1).squeeze(1)
    job = action // nm
    ma = action % nm
    action = TensorDict({
        "job": job,
        "machine": ma
    })
    td.set("action", action)
    return td


def random_multi_agent_fjsp_policy(td: TensorDict):
    """Helper function to select a random action from available actions"""
    logits = torch.randn(td["action_mask"].shape)
    mask = td["action_mask"].clone()
    bs, num_mas, num_jobs_plus_one = mask.shape
    batch_idx = torch.arange(0, bs, device=td.device)

    temp = 1.0
    idle_machines = torch.arange(0, num_mas, device=td.device)[None,:].expand(bs, -1)

    # initialize action buffer
    actions = []
    while mask.any():

        logits_masked = logits.masked_fill(~mask, -torch.inf)
        logits_reshaped = rearrange(logits_masked, "b m j -> b (j m)") / temp
        logp = F.softmax(logits_reshaped, dim=1)

        action = logp.multinomial(1).squeeze(1)

        selected_machine = action % num_mas
        selected_job = action // num_mas
        actions.append(TensorDict(
            {
                "job": selected_job,
                "machine": selected_machine
            },
            batch_size=td.batch_size
        ))

        idle_machines = (
            idle_machines[idle_machines!=selected_machine[:, None]]
            .view(bs, -1)
        )

        mask = mask.scatter(-1, selected_job.view(bs, 1, 1).expand(-1, num_mas, 1), False)
        mask[..., 0] = mask[..., 0].scatter(-1, idle_machines.view(bs, -1), True)
        mask = mask.scatter(-2, selected_machine.view(bs, 1, 1).expand(-1, 1, num_jobs_plus_one), False)

    actions = torch.stack(actions, dim=1)
    td.set("action", actions)
    return td