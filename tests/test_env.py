import os
import torch
import pytest
import torch.nn.functional as F
from tensordict import TensorDict
from einops import rearrange
from marlprp.env.env import MultiAgentEnv
from marlprp.utils.config import EnvParams
from marlprp.utils.dataset import EnvLoader, read_icaps_instances


@pytest.mark.parametrize(
    "instance_path", [
        "data_test/icaps/10s-3i-20p/",
        "data_test/icaps/10s-6i-20p/",
        "data_test/icaps/10s-9i-20p/",
    ]
)
def test_w_lt_instances(instance_path):
    
    solution_path=os.path.join(instance_path, "solution.pth")
    env = MultiAgentEnv(EnvParams())
    dl = EnvLoader(
        env, 
        batch_size=1, 
        path=solution_path,
        read_fn=read_icaps_instances
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
        assert torch.isclose(sol["length"], state.tour_length[0,0]).item(), f"{sol['length']} vs {state.tour_length[0,0]}"