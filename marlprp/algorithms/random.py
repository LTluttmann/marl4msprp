import os
import time
import torch
import pyrootutils
from functools import partial
from marlprp.utils.ops import batchify, unbatchify

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


root = pyrootutils.find_root(__file__, indicator=".gitignore")
data_path = os.path.join(root, "data_test/luttmann/")
instance_paths = [os.path.join(data_path, x) for x in os.listdir(data_path)]




NUM_ITERS = 100


def random__shelf_pointer(embeddings, state: MSPRPState, attn_mask = None):
    bs = state.size(0)
    num_agents = state.num_agents
    num_nodes = state.num_nodes
    return torch.rand((bs, num_agents, num_nodes))


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
    for td, _ in dl:
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



##ARCHIVE###

# def random_policy(state: MSPRPState, env: MSPRPEnv):
#     node_mask = env.get_node_mask(state)
#     bs, num_agents, num_nodes = node_mask.shape

#     # (bs, agents, shelves)
#     node_logits = torch.rand_like(node_mask.float())

#     actions = []
#     busy_agents = []
#     while not node_mask.all():
#         logits_masked = node_logits.masked_fill(node_mask, -torch.inf)
#         logits_reshaped = rearrange(logits_masked, "b a n -> b (n a)")
#         probs = F.softmax(logits_reshaped, dim=-1)
#         action = probs.multinomial(1).squeeze(1)

#         # bs
#         selected_agent = action % num_agents
#         selected_node = action // num_agents

#         # (bs, skus+1)
#         sku_mask = env.get_sku_mask(state, selected_node.unsqueeze(1)).squeeze(1)

#         sku_logits = torch.rand_like(sku_mask.float()).masked_fill(sku_mask, -torch.inf)
#         sku_probs = F.softmax(sku_logits, dim=-1)
#         selected_sku = sku_probs.multinomial(1).squeeze(1)

#         action = TensorDict(
#             {
#                 "agent": selected_agent,
#                 "shelf": selected_node,
#                 "sku": selected_sku
#             },
#             batch_size=state.batch_size
#         )
#         actions.append(action)
#         busy_agents.append(selected_agent)

#         # mask the selected node for all other agents
#         node_mask = node_mask.scatter(-1, selected_node.view(bs, 1, 1).expand(-1, num_agents, 1), True)
#         # mask all actions / nodes for the selected agent
#         node_mask = node_mask.scatter(-2, selected_agent.view(bs, 1, 1).expand(-1, 1, num_nodes), True)


#     actions = torch.stack(actions, dim=1)
#     return actions
