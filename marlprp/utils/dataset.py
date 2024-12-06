import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import ListConfig
from marlprp.env.env import MSPRPEnv


class SynetheticDataset(Dataset):
    def __init__(
            self, 
            env: MSPRPEnv,
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

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, _):
        td = self.env.reset(batch_size=self.batch_size)
        return td
    

class DistributableSynetheticDataset(Dataset):
    def __init__(
            self, 
            env: MSPRPEnv,
            num_samples: int, 
            **kwargs
        ) -> None:
        
        super().__init__()

        self.num_samples = num_samples 
        self.dataset = env.reset(batch_size=self.num_samples)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, 0)


class InstanceFilesDataset(Dataset):

    def __init__(
            self, 
            env: MSPRPEnv, 
            path: str, 
            sort_fn: callable = None
        ) -> None:

        super().__init__()
        self.env = env
        self.files = self.list_files(path, sort_fn)
        self.num_samples = len(self.files)
        self.read_kwargs = env.kwargs_from_files(self.files)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        file = self.files[idx]
        td = self.env.read_file(file, **self.read_kwargs)
        return td

    
    @staticmethod
    def list_files(path, sort_fn = None):
        import os
        files = [
            os.path.join(path, f) 
            for f in os.listdir(path) 
            if os.path.isfile(os.path.join(path, f))
        ]
        assert len(files) > 0, f"No instance files found in path {path}"
        if sort_fn is not None:
            files = sorted(files, key=sort_fn)
        return files

    def collate_fn(self, batch):
        td = torch.cat(batch, 0)
        return self.env.reset(td)

    
class EnvLoader(DataLoader):
    def __init__(
        self, 
        env: MSPRPEnv, 
        batch_size: int,
        dataset_size: int = None,
        path: str = None,
        shuffle: bool = False,
        reload_every_n: int = 1,
        sampler = None,
        batch_sampler = None,
        **kwargs
    ) -> None:

        if path is not None:
            dataset = InstanceFilesDataset(env, path, **kwargs)
            super().__init__(
                dataset=dataset, 
                batch_size=batch_size, 
                sampler=sampler, 
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
            )

        elif reload_every_n >= 1:
            dataset = DistributableSynetheticDataset(
                env=env,
                num_samples=dataset_size,
                **kwargs
            )
            super().__init__(
                dataset=dataset, 
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                batch_sampler=batch_sampler
            )

        else:
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
                batch_sampler=batch_sampler
            )
        

def get_file_dataloader(env, batch_size: int, file_dir: str = None):
    if file_dir is None:
        return []
    
    file_dirs = file_dir if isinstance(file_dir, (ListConfig, list)) else [file_dir]

    dataloader = [
        EnvLoader(
            env=env,
            path=file_dir, 
            batch_size=batch_size
        )
        for file_dir in file_dirs
    ]
    return dataloader
