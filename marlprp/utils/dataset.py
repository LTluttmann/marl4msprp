from typing import Union
from omegaconf import DictConfig
from functools import partial

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from marlprp.env.env import MultiAgentEnv



class SynetheticDataset(Dataset):
    def __init__(
            self, 
            env: MultiAgentEnv,
            batch_size: int, 
            num_samples: int, 
            drop_last: bool = False,
            **kwargs
        ) -> None:
        
        super().__init__()

        self.env = env
        self.num_samples = num_samples 
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.kwargs = kwargs

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, _):
        td = self.env.reset(batch_size=self.batch_size, **self.kwargs)
        return td
    
 

class DistributableSynetheticDataset(Dataset):
    def __init__(
            self, 
            env: MultiAgentEnv,
            num_samples: int, 
            **kwargs
        ) -> None:
        
        super().__init__()
        self.num_samples = num_samples 
        self.datasets = {g.id: g(batch_size=num_samples) for g in env.generators}

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index_tuple):
        dataset_idx, sample_idx = index_tuple
        return self.datasets[dataset_idx][sample_idx], dataset_idx
    
    @staticmethod
    def collate_fn(batch_and_idx):
        batch, dataset_idx = zip(*batch_and_idx)
        return torch.stack(batch, 0), dataset_idx[0]


class InstanceFilesDataset(Dataset):

    def __init__(
            self, 
            path: str, 
            read_fn: callable,
        ) -> None:

        super().__init__()
        self.instances = read_fn(path)
        self.num_samples = len(self.instances)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        td = self.instances[idx]
        return td

    def collate_fn(self, batch):
        td = torch.stack(batch, 0)
        return td, None



class SequentialSampler(Sampler[int]):

    def __init__(
            self, 
            data_source: DistributableSynetheticDataset,
            batch_size: int,
            dataset_distribution = None,
        ) -> None:

        self.batch_size = batch_size
        self.datasets = data_source.datasets
        self.dataset_indices = list(self.datasets.keys())
        self.dataset_distribution = dataset_distribution


    def __iter__(self):
        sampler_iter = iter(range(len(self)))
        while True:
            try:
                dataset_idx = np.random.choice(self.dataset_indices, p=self.dataset_distribution)
                batch = [next(sampler_iter) for _ in range(self.batch_size)]
                yield from [(dataset_idx, i) for i in batch]
            except StopIteration:
                break

    def __len__(self) -> int:
        return min([len(x) for x in self.datasets.values()]) 


    
class EnvLoader(DataLoader):
    def __init__(
        self, 
        env: MultiAgentEnv = None, 
        batch_size: int = 1,
        dataset_size: int = None,
        path: str = None,
        shuffle: bool = False,
        reload_every_n: int = 1,
        sampler = None,
        batch_sampler = None,
        mode: str = "train",
        dataset_distribution = None,
        **kwargs
    ) -> None:

        if path is not None:
            dataset = InstanceFilesDataset(path, **kwargs)
            super().__init__(
                dataset=dataset, 
                batch_size=batch_size, 
                sampler=sampler, 
                collate_fn=dataset.collate_fn,
                shuffle=False,
            )

        elif reload_every_n >= 1:
            assert env is not None
            dataset = DistributableSynetheticDataset(
                env=env,
                num_samples=dataset_size,
                **kwargs
            )
            if sampler is not None:
                raise ValueError("passing sampler externally not implemented yet")
            sampler = SequentialSampler(dataset, batch_size, dataset_distribution)
            super().__init__(
                dataset=dataset, 
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                batch_sampler=batch_sampler,
                shuffle=False,
            )

        else:
            assert env is not None
            dataset = SynetheticDataset(
                env=env,
                batch_size=batch_size, 
                num_samples=dataset_size, 
                **kwargs
            )
            super().__init__(
                dataset=dataset, 
                batch_size=None, 
                collate_fn=lambda x: x,
                sampler=sampler,
                batch_sampler=batch_sampler,
                shuffle=False,
            )
        

def get_file_dataloader(env, batch_size: int, file_dir: Union[dict, str] = None, num_agents = None):
    if file_dir is None:
        return {}
    
    if isinstance(file_dir, (dict, DictConfig)):
        dataloader = {
            file_name: EnvLoader(
                env=env,
                path=file_dir, 
                batch_size=batch_size,
                read_fn=partial(read_luttmann, num_agents=num_agents)
            )
            for file_name, file_dir in file_dir.items()
        }
    elif isinstance(file_dir, str):
        dataloader = EnvLoader(
            env=env,
            path=file_dir, 
            batch_size=batch_size,
            read_fn=partial(read_luttmann, num_agents=num_agents)
        )
    else:
        raise ValueError(f"Expected str or dict for param file_dir, got {file_dir}")
    return dataloader


def read_luttmann(path, num_agents):
    td = torch.load(path)
    bs = td.batch_size
    # NOTE below code does not work since num_agents in instances is fixed to 1 (e.g. through current_lcation)
    if num_agents is None:
        num_agents = torch.ceil(td["demand"].sum(-1, keepdim=True) / td["init_capacity"].max())
        max_num_agents = int(num_agents.max().item())
    else:
        max_num_agents = num_agents
    agent_pad_mask = num_agents < torch.arange(1, max_num_agents+1).view(1, -1).expand(*bs, max_num_agents)
    num_agents = max_num_agents
    current_location = td["current_location"].view(-1,1).repeat(1, num_agents)
    capacity = td["init_capacity"].view(-1,1).repeat(1, num_agents)
    capacity[agent_pad_mask] = 0
    td.update({
        "init_capacity": capacity,
        "current_location": current_location,
        "agent_pad_mask": agent_pad_mask
    })
    return td
    