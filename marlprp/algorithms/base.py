import abc
import copy
import logging
from omegaconf import ListConfig
from dataclasses import dataclass
from rich.logging import RichHandler
from typing import Dict, Type, Any, Union, Optional, Callable

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist

from lightning import LightningModule

from rl4co.utils.optim_helpers import create_scheduler

from marlprp.env.env import MSPRPEnv
from marlprp.models.policies import RoutingPolicy
from marlprp.utils.utils import monitor_lr_changes
from marlprp.utils.dataset import EnvLoader, get_file_dataloader
from esel.algorithms.utils import SampleGenerator, make_replay_buffer
from marlprp.algorithms.model_args import ModelParams, ModelWithReplayBufferParams
from marlprp.utils.config import (
    EnvParams,
    TrainingParams, 
    ValidationParams, 
    TestParams, 
    save_config_to_dict
)


model_registry: Dict[str, Type['LearningAlgorithm']] = {}


# configure logging on module level, redirect to file
log = logging.getLogger("lightning.pytorch.core")
rich_handler = RichHandler(rich_tracebacks=True)
log.addHandler(rich_handler)


Numeric = Union[int, float]


@dataclass
class NumericParameter:
    _val: Numeric
    update_coef: Optional[float] = None
    min: Optional[Numeric] = None
    max: Optional[Numeric] = None
    dtype: Type | Callable = None

    def __post_init__(self):
        self.min = self.min or float("-inf")
        self.max = self.max or float("inf")

    @property
    def val(self):
        if self._val is None:
            return None
        val = min(max(self._val, self.min), self.max)
        if self.dtype is not None:
            val = self.dtype(val)
        return val
    
    @val.setter
    def val(self, value):
        self._val = value

    def update(self):
        if self.update_coef is not None:
            self.val *= self.update_coef
        return self
    

class LearningAlgorithm(LightningModule):

    name = ...

    def __init__(
        self, 
        env: MSPRPEnv,
        policy: RoutingPolicy,
        model_params: ModelParams,
        train_params: TrainingParams,
        val_params: ValidationParams,
        test_params: TestParams
    ) -> None:
        
        super(LearningAlgorithm, self).__init__()

        self.pylogger = log
        self.save_hyperparameters("model_params")

        self.model_params = model_params
        self.train_params = train_params
        self.val_params = val_params
        self.test_params = test_params

        # specifically define batch size to be altered by batch size finder
        self.param_container: Dict[NumericParameter] = {}
        self.train_batch_size = self.setup_parameter(train_params.batch_size, dtype=int)

        self.env = env
        self.policy = policy

        self.validation_step_rewards = []

        self.val_set_names = ["synthetic"]
        self.test_set_names = ["synthetic"]


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
        env: MSPRPEnv,
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
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        state = self.env.reset(batch)
        reward = self.policy(state, self.env)["reward"]
        val_set_name = self.val_set_names[dataloader_idx]

        if val_set_name == "synthetic":
            self.validation_step_rewards.append(reward)
            metric_name = "val/reward"
        else:
            metric_name = f"val/{val_set_name}/reward"

        self.log(
            metric_name, 
            reward.mean(), 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True,
            add_dataloader_idx=False,
        )
    
    # PyTorch Lightning's built-in test_step method
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        state = self.env.reset(batch)
        test_reward = self.policy(state, self.env)["reward"]
        test_set_name = self.test_set_names[dataloader_idx]
        self.log(
            f"test/{test_set_name}/reward", 
            test_reward.mean(), 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True,
            add_dataloader_idx=False,
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
    
    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError()

    def _get_dataloader(self, params: Union[TrainingParams, ValidationParams, TestParams]):

        dataloader = EnvLoader(
            env=self.env,
            batch_size=params.batch_size,
            dataset_size=params.dataset_size,
            reload_every_n=1,
        )

        return dataloader

    def train_dataloader(self):
        return EnvLoader(
            env=self.env, 
            batch_size=self.train_batch_size, 
            dataset_size=self.train_params.dataset_size,
            reload_every_n=self.train_params.reload_every_n
        )
    
    def val_dataloader(self):
        val_dl = self._get_dataloader(self.val_params)
        return val_dl
    
    def test_dataloader(self):
        try:
            test_file_dls = get_file_dataloader(
                self.env, 
                self.test_params.batch_size, 
                self.test_params.data_dir
            )
        except FileNotFoundError as e:
            log.warning(e.__str__())
            test_file_dls = {}

        test_dl = self._get_dataloader(self.test_params)
        test_file_dls["synthetic"] = test_dl

        keys, vals = map(lambda x: list(x), test_file_dls.items())
        self.test_set_names = keys
        return vals

    def on_train_epoch_start(self) -> None:
        self.train()
        self._setup_decoding_strategy(self.train_params)

    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.eval()
        self._setup_decoding_strategy(self.val_params)
    
    def on_test_epoch_start(self) -> None:
        self.eval()
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
        if self.trainer.sanity_checking:
            return
        self._update_params()
        epoch_average = self._get_global_validation_reward().item()
        if hasattr(self, "reduce_on_plateau"):
            self.reduce_on_plateau.step(epoch_average)
        return epoch_average

    @classmethod
    def init_from_checkpoint(
        cls, 
        env_params: EnvParams,
        train_params: TrainingParams, 
        val_params: ValidationParams, 
        test_params: TestParams,
        model_params: ModelParams = None,
    ):
        assert train_params.checkpoint is not None
        ckpt = torch.load(train_params.checkpoint)
        model_params_ckpt: ModelParams = ckpt["hyper_parameters"]["model_params"]
        if model_params is None:
            model_params = model_params_ckpt
        else:
            assert model_params.policy.policy == model_params_ckpt.policy.policy, \
            "Policy of checkpoint and the one specified in new model parameters diverge."

        env = MSPRPEnv.initialize(params=env_params, multiagent=model_params.policy.is_multiagent_policy)
        policy = RoutingPolicy.initialize(params=model_params.policy)

        Algorithm = model_registry[model_params.algorithm]

        model = Algorithm.load_from_checkpoint(
            train_params.checkpoint,
            map_location=torch.device("cpu"),
            env=env,
            policy=policy,
            model_params=model_params,
            train_params=train_params,
            val_params=val_params,
            test_params=test_params,
        )
        
        return model, model_params

    def __setattr__(self, name, value):
        """Intercept attribute setting to hook parameter setup."""
        if isinstance(value, NumericParameter):
            self.param_container[name] = value
            return super().__setattr__(name, value.val)
        super().__setattr__(name, value)

    def setup_parameter(
        self, 
        param: Union[Any, tuple[Union[float, int], Union[float, int], float]],
        dtype: Type = None
    ) -> NumericParameter:

        if isinstance(param, (ListConfig, list, tuple)):
            min_val, max_val, update_coef = param
            val = max_val if update_coef < 1 else min_val
            return NumericParameter(val, update_coef=update_coef, min=min_val, max=max_val, dtype=dtype)
        else:
            return NumericParameter(param, dtype=dtype)

    def _update_params(self):
        for name, param in self.param_container.items():
            setattr(self, name, param.update())


class ManualOptLearningAlgorithm(LearningAlgorithm):
    def __init__(
            self, 
            env: MSPRPEnv, 
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

    def manual_opt_step(self, loss, batch_idx: int) -> None:
        opt = self.optimizers()
        # opt.zero_grad()
        self.manual_backward(loss)
        if self.train_params.max_grad_norm is not None:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.train_params.max_grad_norm,
                gradient_clip_algorithm="norm",
            )

        if (batch_idx + 1) % self.train_params.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()



class LearningAlgorithmWithReplayBuffer(ManualOptLearningAlgorithm):
    def __init__(
        self, 
        env: MSPRPEnv,
        policy: nn.Module,
        model_params: ModelWithReplayBufferParams,
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

        self.model_params: ModelWithReplayBufferParams

        # setup reference policy
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.set_decode_type(
            decode_type="sampling", 
            tanh_clipping=train_params.decoding["tanh_clipping"],
            top_p=train_params.decoding["top_p"],
            temperature=model_params.ref_model_temp
        )

        # get rollout batch size
        rollout_batch_size = model_params.rollout_batch_size or train_params.batch_size
        if isinstance(rollout_batch_size, float):
            rollout_batch_size = int(rollout_batch_size * train_params.batch_size)
        self._rollout_batch_size = rollout_batch_size
        # determine minibatch size
        mini_batch_size = model_params.mini_batch_size or train_params.batch_size
        if isinstance(mini_batch_size, float):
            mini_batch_size = int(mini_batch_size * train_params.batch_size)
        self.mini_batch_size = mini_batch_size
        # make replay buffer
        self.rb = make_replay_buffer(
            buffer_size=model_params.buffer_size, 
            batch_size=self.mini_batch_size, 
            device=model_params.buffer_storage_device, 
            priority_key=model_params.priority_key,
            **model_params.buffer_kwargs
        )
        self.setup_inner_training_loop()


    def setup_inner_training_loop(self):
        if self.model_params.num_batches is not None and self.model_params.inner_epochs is not None:
            raise ValueError("Cannot use both, num_batches and inner_epochs for inner training loop")
        # got to initialize the backup attributes first, so that setup_experience_sampler in the setter doesnt fail
        if self.model_params.num_batches is not None:
            self._inner_epochs = None
            self.num_batches = self.setup_parameter(self.model_params.num_batches, dtype=int)
        else:
            self._num_batches = None
            self.inner_epochs = self.setup_parameter(self.model_params.inner_epochs, dtype=int)


    def setup_experience_sampler(self):
        self.experience_sampler = SampleGenerator(
            rb=self.rb, 
            num_iter=self.inner_epochs, 
            num_samples=self.num_batches
        )

    @property
    def num_batches(self):
        return self._num_batches
    
    @num_batches.setter
    def num_batches(self, value):
        self._num_batches = value
        self.setup_experience_sampler()

    @property
    def inner_epochs(self):
        return self._inner_epochs
    
    @inner_epochs.setter
    def inner_epochs(self, value):
        self._inner_epochs = value
        self.setup_experience_sampler()

    @property
    def rollout_batch_size(self):
        return min(self.train_batch_size, self._rollout_batch_size)
    
    @rollout_batch_size.setter
    def rollout_batch_size(self, value):
        self._rollout_batch_size = value

    def on_train_epoch_end(self):
        super().on_train_epoch_end()


class EvalModule(LightningModule):

    def __init__(
        self, 
        env: MSPRPEnv,
        policy: RoutingPolicy,
        test_params: TestParams
    ) -> None:
        
        super(EvalModule, self).__init__()
        
        self.env = env
        self.policy = policy
        self.test_params = test_params
        self.test_set_names = ["synthetic"]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("training_step not defined for eval module")
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("validation_step not defined for eval module")
    
    # PyTorch Lightning's built-in test_step method
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        test_reward = self.policy(batch, self.env)["reward"]
        test_set_name = self.test_set_names[dataloader_idx]
        self.log(
            f"test/{test_set_name}/reward", 
            test_reward.mean(), 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            add_dataloader_idx=False,
            sync_dist=True
        )

    def test_dataloader(self):

        test_loader = EnvLoader(
            env=self.env,
            dataset_params=self.test_params,
            reload_every_n=1
        )


        return test_loader
    
    @classmethod
    def init_from_checkpoint(cls, test_params: TestParams, env_params = None, policy_cfg = None):
        assert test_params.checkpoint is not None
        ckpt = torch.load(test_params.checkpoint)
        model_params = ckpt["hyper_parameters"]["model_params"]
        policy_params = model_params.policy
        env_params = env_params if env_params is not None else policy_params.env
        policy_params.env = env_params
        if policy_cfg is not None:
            policy_params.__dict__.update(policy_cfg)

        env = MSPRPEnv(params=env_params)
        policy = RoutingPolicy.initialize(policy_params)

        policy_state_dict = {
            k[len("policy."):]: v 
            for k,v in ckpt["state_dict"].items() 
            if k.startswith("policy.")
        }
        policy.load_state_dict(policy_state_dict)   
        
        return cls(env, policy, test_params), model_params

    def on_test_epoch_start(self) -> None:
        self._setup_decoding_strategy(self.test_params)

    def _setup_decoding_strategy(self, params: Union[TrainingParams, ValidationParams, TestParams]):
        decoding_params = save_config_to_dict(params.decoding)
        decoding_strategy = decoding_params.pop("decoding_strategy")
        self.policy.set_decode_type(decoding_strategy, **decoding_params)
