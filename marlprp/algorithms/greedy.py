import torch

import os
import time
import pyrootutils
from tqdm import tqdm 
from functools import partial
from marlprp.env.env import MultiAgentEnv
from marlprp.utils.config import EnvParams
from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import MahamParams
from marlprp.utils.dataset import EnvLoader, read_luttmann
from marlprp.models.decoder.base import BaseDecoder
from marlprp.models.decoder.multi_agent import (
    HierarchicalMultiAgentDecoder, MultiAgentShelfDecoder, MultiAgentSkuDecoder
)
from marlprp.decoding.strategies import get_decoding_strategy
from marlprp.utils.ops import batchify, unbatchify, gather_by_index


root = pyrootutils.find_root(__file__, indicator=".gitignore")
data_path = os.path.join(root, "data_test/ood/")
instance_paths = [os.path.join(data_path, x) for x in os.listdir(data_path)]



NUM_ITERS = 200


def random__shelf_pointer(embeddings, state: MSPRPState, attn_mask = None):
    agent_coordinates = gather_by_index(state.coordinates, state.current_location, dim=1)
    # scale probs of visiting a node according to its negative distance
    logits = -torch.cdist(agent_coordinates, state.coordinates)
    # increase prob of longer tour to return to depot
    logits[..., :state.num_depots] += (state.tour_length - state.tour_length.max(1, keepdim=True).values)[..., None]
    return logits


def random__sku_pointer(embeddings, state: MSPRPState, attn_mask = None):
    bs = state.size(0)
    num_agents = state.num_agents
    num_skus = state.num_skus
    supply_at_agent_loc = state.supply_w_depot_and_dummy.gather(1, state.current_location[..., None].expand(bs, num_agents, num_skus+1))
    min_pickable = torch.minimum(supply_at_agent_loc, state.demand_w_dummy[:, None])
    min_pickable = torch.minimum(min_pickable, state.remaining_capacity[..., None])
    return min_pickable  # torch.rand((bs, num_agents, num_skus+1))
    

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
    env_params = EnvParams(num_agents=None, goal="min-max")
    env = MultiAgentEnv(env_params)
    dl = EnvLoader(
        env, 
        batch_size=1, 
        path=solution_path,
        read_fn=partial(read_luttmann, num_agents=None)
    )

    model_params = MahamParams(policy="random", env=env_params, decoder_attn_mask=True)
    random_policy = RandomHierarchicalPolicy(model_params)

    rewards = []
    for td, _ in tqdm(dl):
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
        print("Processing test files in folder: ", instance_path)
        instance_name = instance_path.split("/")[-1]
        start_time = time.time()
        rewards = get_random_baseline(instance_path)
        duration = time.time() - start_time
        # solutions[instance_name] = {
        #     "avg_reward": rewards.mean().item(),
        #     "avg_runtime": duration / rewards.numel()
        # }
        torch.save(
            obj={
                "avg_reward": rewards.mean().item(),
                "avg_runtime": duration / rewards.numel()
            },
            f=os.path.join(instance_path, "greedy.pth")
        )
    # print(solutions)
