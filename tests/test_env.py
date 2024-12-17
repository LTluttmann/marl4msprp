import os
import torch
import pytest
import torch.nn.functional as F
from tensordict import TensorDict
from einops import rearrange
from marlprp.env.env import MSPRPEnv
from marlprp.utils.config import EnvParams
from marlprp.utils.dataset import EnvLoader

# @pytest.mark.parametrize("num_agents", [2])
# def test_env(num_agents):

#     params = EnvParams(
#         num_agents=num_agents, 
#         num_depots=2, 
#         num_shelves=5, 
#         num_skus=10, 
#         avg_loc_per_sku=2,
#     )

#     env = MSPRPEnv(params)

#     tc = env.reset(batch_size=2)

#     while not tc.done.all():
#         node_mask = env.get_node_mask(tc)
#         bs, num_agents, num_nodes = node_mask.shape

#         # (bs, agents, shelves)
#         node_logits = torch.rand_like(node_mask.float())

#         actions = []
#         while not node_mask.all():
#             logits_masked = node_logits.masked_fill(node_mask, -torch.inf)
#             logits_reshaped = rearrange(logits_masked, "b a n -> b (n a)")
#             probs = F.softmax(logits_reshaped, dim=-1)
#             action = probs.multinomial(1).squeeze(1)

#             # bs
#             selected_agent = action % num_agents
#             selected_node = action // num_agents

#             # (bs, skus+1)
#             sku_mask = env.get_sku_mask(tc, selected_node)

#             sku_logits = torch.rand_like(sku_mask.float()).masked_fill(sku_mask, -torch.inf)
#             sku_probs = F.softmax(sku_logits, dim=-1)
#             selected_sku = sku_probs.multinomial(1).squeeze(1) - 1

#             action = TensorDict(
#                 {
#                     "agent": selected_agent,
#                     "shelf": selected_node,
#                     "sku": selected_sku
#                 },
#                 batch_size=tc.batch_size
#             )
#             actions.append(action)

#             # mask the selected node for all other agents
#             node_mask = node_mask.scatter(-1, selected_node.view(bs, 1, 1).expand(-1, num_agents, 1), True)
#             # mask all actions / nodes for the selected agent
#             node_mask = node_mask.scatter(-2, selected_agent.view(bs, 1, 1).expand(-1, 1, num_nodes), True)


#         actions = torch.stack(actions, dim=1)
#         tc = env.step(action, tc)


#     reward = env.get_reward(tc)
#     print(reward)


@pytest.mark.parametrize(
    "instance_path", [
        "data_test/luttmann/10s-3i-20p/",
        "data_test/luttmann/10s-6i-20p/",
        "data_test/luttmann/10s-9i-20p/",
    ]
)
def test_w_lt_instances(instance_path):
    
    solution_path=os.path.join(instance_path, "solution.pth")
    env = MSPRPEnv(EnvParams())
    dl = EnvLoader(
        env, 
        batch_size=1, 
        path=solution_path,
        read_fn=lambda x: torch.load(x)
    )
    for td in dl:
        sol = td.pop("solution")[0]
        state = env.reset(td)
        orig_state = state.clone()
        i = 0
        while not state.done.item():
            next_shelf = sol["shelf"][i].view(1,1)
            next_sku = sol["sku"][i].view(1,1)
            agent = torch.tensor(0).view(1,1)
            action = TensorDict({
                "shelf": next_shelf.long(),
                "sku": next_sku.long(),
                "agent": agent.long(),
                "units": sol["units"][i].view(1,1)
            }, batch_size=[1, 1])
            state = env.step(action, state)
            i += 1
        assert torch.isclose(sol["length"], state.tour_length).item(), f"{sol['length']} vs {state.tour_length}"