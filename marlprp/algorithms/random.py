import os
import torch
import pytest
import torch.nn.functional as F
from tensordict import TensorDict
from einops import rearrange
from marlprp.env.env import MSPRPEnv
from marlprp.utils.config import EnvParams
from marlprp.utils.dataset import EnvLoader, read_luttmann



def random_policy(state):
    print(state)


def get_random_baseline(instance_path):
    
    solution_path=os.path.join(instance_path, "td_data.pth")
    env = MSPRPEnv(EnvParams())
    dl = EnvLoader(
        env, 
        batch_size=1, 
        path=solution_path,
        read_fn=read_luttmann
    )
    for td in dl:
        state = env.reset(td)
        i = 0
        while not state.done.item():
            action = random_policy(state)
            state = env.step(action, state)
            i += 1


if __name__ == "__main__":
    get_random_baseline("/home/laurin.luttmann/repos/marl4msprp/data_test/luttmann/10s-3i-20p")