import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from einops import rearrange
from marlprp.env.env import MSPRPEnv
from marlprp.utils.config import EnvParams


@pytest.mark.parametrize("num_agents", [2])
def test_env(num_agents):

    params = EnvParams(
        num_agents=num_agents, 
        num_depots=2, 
        num_shelves=5, 
        num_skus=10, 
        avg_loc_per_sku=2,
    )

    env = MSPRPEnv(params)

    tc = env.reset(batch_size=2)

    while not tc.done.all():
        node_mask = env.set_node_mask(tc)
        bs, num_agents, num_nodes = node_mask.shape

        # (bs, agents, shelves)
        node_logits = torch.rand_like(node_mask.float())

        actions = []
        while not node_mask.all():
            logits_masked = node_logits.masked_fill(node_mask, -torch.inf)
            logits_reshaped = rearrange(logits_masked, "b a n -> b (n a)")
            probs = F.softmax(logits_reshaped, dim=-1)
            action = probs.multinomial(1).squeeze(1)

            # bs
            selected_agent = action % num_agents
            selected_node = action // num_agents

            # (bs, skus+1)
            sku_mask = env.get_item_mask_from_node(tc, chosen_node=selected_node)

            sku_logits = torch.rand_like(sku_mask.float()).masked_fill(sku_mask, -torch.inf)
            sku_probs = F.softmax(sku_logits, dim=-1)
            selected_sku = sku_probs.multinomial(1).squeeze(1) - 1

            action = TensorDict(
                {
                    "agent": selected_agent,
                    "shelf": selected_node,
                    "sku": selected_sku
                },
                batch_size=tc.batch_size
            )
            actions.append(action)

            # mask the selected node for all other agents
            node_mask = node_mask.scatter(-1, selected_node.view(bs, 1, 1).expand(-1, num_agents, 1), True)
            # mask all actions / nodes for the selected agent
            node_mask = node_mask.scatter(-2, selected_agent.view(bs, 1, 1).expand(-1, 1, num_nodes), True)

        td = TensorDict(
            {
                "state": tc,
                "action": torch.stack(actions, dim=1)
            },
            batch_size=tc.batch_size
        )

        tc = env.step(td)


    reward = env.get_reward(tc)
    print(reward)