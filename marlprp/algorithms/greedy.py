import os
import time
import torch
import pyrootutils
import numpy as np
from marlprp.utils.ops import batchify, unbatchify, gather_by_index

from marlprp.env.env import MSPRPEnv
from marlprp.utils.config import EnvParams
from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import MahamParams
from marlprp.utils.dataset import EnvLoader, read_luttmann
from marlprp.models.decoder.base import BaseDecoder
from marlprp.models.decoder.multi_agent import (
    HierarchicalMultiAgentDecoder, MultiAgentShelfDecoder, MultiAgentSkuDecoder
)
from marlprp.decoding.strategies import get_decoding_strategy


root = pyrootutils.find_root(__file__, indicator=".gitignore")
data_path = os.path.join(root, "data_test/luttmann/")
instance_paths = [os.path.join(data_path, x) for x in os.listdir(data_path)]



NUM_ITERS = 100


def random__shelf_pointer(embeddings, state: MSPRPState, attn_mask = None):
    agent_coordinates = gather_by_index(state.coordinates, state.current_location, dim=1)
    logits = -torch.cdist(agent_coordinates, state.coordinates)
    return logits


def random__sku_pointer(embeddings, state: MSPRPState, attn_mask = None):
    bs = state.size(0)
    num_agents = state.num_agents
    num_skus = state.num_skus
    return torch.rand((bs, num_agents, num_skus+1))
    

class RandomHierarchicalPolicy(HierarchicalMultiAgentDecoder):
    def __init__(self, model_params):
        BaseDecoder.__init__(self, model_params)
        self.shelf_decoder = MultiAgentShelfDecoder(model_params, pointer=random__shelf_pointer)
        self.sku_decoder = MultiAgentSkuDecoder(model_params, pointer=random__sku_pointer)
        self.dec_strategy = get_decoding_strategy("sampling")
        self.shelf_decoder.dec_strategy = self.dec_strategy
        self.sku_decoder.dec_strategy = self.dec_strategy


def get_random_baseline(instance_path):
    
    solution_path=os.path.join(instance_path, "td_data.pth")
    env_params = EnvParams(always_mask_depot=True, goal="min-max")
    env = MSPRPEnv(env_params)
    dl = EnvLoader(
        env, 
        batch_size=1, 
        path=solution_path,
        read_fn=read_luttmann
    )

    model_params = MahamParams(policy="random", env=env_params, decoder_attn_mask=True)
    random_policy = RandomHierarchicalPolicy(model_params)

    rewards = []
    for td in dl:
        state = env.reset(td)
        state = batchify(state, NUM_ITERS)
        i = 0
        while not state.done.all():

            with torch.no_grad():
                td = random_policy(None, state, env, return_logp=False)
                state = td.pop("next")
            i += 1

        reward = env.get_reward(state)

        reward_unbs = unbatchify(reward, NUM_ITERS)
        best_rew, best_idx = reward_unbs.max(dim=1)
        rewards.append(best_rew)

    return torch.stack(rewards).squeeze()

if __name__ == "__main__":
    torch.manual_seed(1234567)
    solutions = {}
    for instance_path in instance_paths:
        instance_name = instance_path.split("/")[-1]
        start_time = time.time()
        rewards = get_random_baseline(instance_path)
        duration = time.time() - start_time
        solutions[instance_name] = {
            "avg_reward": rewards.mean().item(),
            "avg_runtime": duration / rewards.numel()
        }

    print(solutions)
