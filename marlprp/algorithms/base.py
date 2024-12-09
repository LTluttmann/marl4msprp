import abc
from typing import Dict, Type, Any, Union
from omegaconf import ListConfig

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist

from lightning import LightningModule

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.optim_helpers import create_scheduler
from rl4co.utils import pylogger

from marlprp.utils.dataset import EnvLoader
from marlprp.utils.utils import monitor_lr_changes
from marlprp.utils.config import TrainingParams, ValidationParams, TestParams, save_config_to_dict
from marlprp.models.policies import RoutingPolicy
from marlprp.algorithms.model_args import ModelParams


log = pylogger.get_pylogger(__name__)

model_registry: Dict[str, Type['LearningAlgorithm']] = {}




class LearningAlgorithm(LightningModule, metaclass=abc.ABCMeta):

    name = ...

    def __init__(
        self, 
        env: RL4COEnvBase,
        policy: RoutingPolicy,
        model_params: ModelParams,
        train_params: TrainingParams,
        val_params: ValidationParams,
        test_params: TestParams
    ) -> None:
        
        super(LearningAlgorithm, self).__init__()
        
        self.model_params = model_params
        self.train_params = train_params
        self.val_params = val_params
        self.test_params = test_params

        self.env = env
        self.policy = policy

        self.validation_step_rewards = []

        self.n_devices = train_params.n_devices
        self.test_set_names = ["synthetic"]
        self.pylogger = log

    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        model_registry[cls.name] = cls

    def _get_optimizer(self):
        optimizer_kwargs = save_config_to_dict(self.train_params.optimizer_kwargs)
        learning_rate = optimizer_kwargs.pop("policy_lr")
        optimizer_kwargs = {
            "lr": learning_rate,
            **optimizer_kwargs
        }
        policy_params = self.policy.parameters()
        optimizer = Adam(policy_params, **optimizer_kwargs)
        return optimizer

    @classmethod
    def initialize(
        cls,
        env: RL4COEnvBase,
        policy: nn.Module,
        model_params: ModelParams,
        train_params: TrainingParams,
        val_params: ValidationParams,
        test_params: TestParams
    ):
        Algorithm = model_registry[model_params.algorithm]
        return Algorithm(
            env,
            policy,
            model_params,
            train_params,
            val_params,
            test_params,
        )

    # PyTorch Lightning's built-in validation_step method
    def validation_step(self, batch, batch_idx):
        state = self.env.reset(batch)
        reward = self.policy(state, self.env)["reward"]
        self.validation_step_rewards.append(reward)
        self.log("val/reward", reward.mean(), prog_bar=True, on_epoch=True, sync_dist=True)
    
    # PyTorch Lightning's built-in test_step method
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        state = self.env.reset(batch)
        test_reward = self.policy(state, self.env)["reward"]
        test_set_name = self.test_set_names[dataloader_idx]
        self.log(
            f"test/{test_set_name}/reward", 
            test_reward.mean(), 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True, 
            add_dataloader_idx=False,
            sync_dist=True
        )


    # PyTorch Lightning's built-in configure_optimizers method
    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        lr_scheduler = create_scheduler(
            optimizer, 
            self.train_params.lr_scheduler,
            **self.train_params.lr_scheduler_kwargs    
        )
        if self.train_params.lr_reduce_on_plateau_patience:
            self.reduce_on_plateau = ReduceLROnPlateau(
                optimizer=optimizer,
                mode="max",
                factor=0.5,
                patience=self.train_params.lr_reduce_on_plateau_patience
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "frequency": self.train_params.lr_scheduler_interval,
                "monitor": self.train_params.lr_scheduler_monitor,
            }
        }
    
    @abc.abstractmethod
    def training_step(self, batch: Any, batch_idx: int):
        pass

    # PyTorch Lightning's train_dataloader and val_dataloader for data loading
    def train_dataloader(self):
        train_dl = EnvLoader(
            env=self.env, 
            batch_size=self.train_params.batch_size,
            dataset_size=self.train_params.dataset_size
        )
        return train_dl

    def val_dataloader(self):
        val_dl = EnvLoader(
            env=self.env, 
            batch_size=self.val_params.batch_size,
            dataset_size=self.val_params.dataset_size,
        )
        return val_dl
    
    def test_dataloader(self):
        test_dir = self.test_params.data_dir or ""
        test_dirs = test_dir if isinstance(test_dir, (ListConfig, list)) else [test_dir]

        try:
            test_loader = [
                EnvLoader(
                    env=self.env,
                    path=test_dir, 
                    dataset_params=self.test_params
                )
                for test_dir in test_dirs
            ]
            for i in range(len(test_dirs)):
                self.test_set_names.insert(0, f"files{i+1}")

        except FileNotFoundError:
            log.warning("Could not find test files. Generate synthetic test dataset")
            test_loader = []

        test_loader.append(
            EnvLoader(
                env=self.env,
                batch_size=self.test_params.batch_size,
                dataset_size=self.test_params.dataset_size
            )
        )

        return test_loader
    
    def on_train_epoch_start(self) -> None:
        self._setup_decoding_strategy(self.train_params)

    def on_validation_epoch_start(self) -> None:
        self._setup_decoding_strategy(self.val_params)
    
    def on_test_epoch_start(self) -> None:
        self._setup_decoding_strategy(self.test_params)

    def _setup_decoding_strategy(self, params: Union[TrainingParams, ValidationParams, TestParams]):
        decoding_params = save_config_to_dict(params.decoding)
        decoding_strategy = decoding_params.pop("decoding_strategy")
        self.policy.set_decode_type(decoding_strategy, **decoding_params)

    def _get_global_validation_reward(self):
        # Stack the performance metrics on this GPU (without computing the mean yet)
        local_val_rewards = torch.cat(self.validation_step_rewards, dim=0)
        # Reset the list for the next validation epoch
        self.validation_step_rewards.clear()
        # Check if we're using multiple GPUs (DDP is enabled)
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        if world_size > 1:
            # If DDP is enabled, gather the unaggregated performance tensors from all GPUs
            all_val_rewards = [torch.zeros_like(local_val_rewards) for _ in range(world_size)]
            # Gather all performance metrics from all GPUs (unaggregated)
            dist.all_gather(all_val_rewards, local_val_rewards)
            # Concatenate all the gathered performance tensors into one global tensor
            all_val_rewards = torch.cat(all_val_rewards, dim=0)
            # remove padded entries
            all_val_rewards = all_val_rewards[all_val_rewards != 0]
        else:
            # If using a single GPU, the global performance tensor is just the local tensor
            all_val_rewards = local_val_rewards

        # Compute the average performance across all GPUs (or just this one if single GPU)
        epoch_average = all_val_rewards.mean()
        return epoch_average

    @monitor_lr_changes
    def on_validation_epoch_end(self) -> None:
        epoch_average = self._get_global_validation_reward().item()
        if hasattr(self, "reduce_on_plateau"):
            self.reduce_on_plateau.step(epoch_average)
        return epoch_average


class ManualOptLearningAlgorithm(LearningAlgorithm):
    def __init__(
            self, 
            env: RL4COEnvBase, 
            policy: RoutingPolicy, 
            model_params: ModelParams, 
            train_params: TrainingParams, 
            val_params: ValidationParams, 
            test_params: TestParams
        ) -> None:

        super().__init__(
            env=env, 
            policy=policy, 
            model_params=model_params, 
            train_params=train_params, 
            val_params=val_params, 
            test_params=test_params
        )

        self.automatic_optimization = False

    @monitor_lr_changes
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # update learning rate schedulers
        scheduler = self.lr_schedulers()
        scheduler.step()


