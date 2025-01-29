import os
import math
import copy
import numpy as np
from omegaconf import ListConfig
from collections import defaultdict
from rich.logging import RichHandler
from typing import Dict, Type, Any, Union

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rl4co.utils.optim_helpers import create_scheduler
from torchrl.data.replay_buffers import ReplayBufferEnsemble

from marlprp.env.env import MSPRPEnv
from marlprp.models.policies import RoutingPolicy
from marlprp.utils.utils import monitor_lr_changes
from marlprp.utils.logger import get_lightning_logger
from marlprp.utils.dataset import EnvLoader, get_file_dataloader
from marlprp.utils.ops import all_gather_w_padding, all_gather_numeric
from marlprp.algorithms.utils import NumericParameter, make_replay_buffer
from marlprp.algorithms.model_args import ModelParams, ModelWithReplayBufferParams
from marlprp.utils.config import (
    EnvParams,
    TrainingParams, 
    ValidationParams, 
    TestParams, 
    save_config_to_dict
)


model_registry: Dict[str, Type['LearningAlgorithm']] = {}


logger = get_lightning_logger(__name__, rzo=True)
rich_handler = RichHandler(
    # rich_tracebacks=True,
    omit_repeated_times=False,
    show_level=False,
    show_path=False,
    show_time=False,
)
logger.addHandler(rich_handler)
    

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

        self.pylogger = logger
        # self.save_hyperparameters("model_params")
        self.model_params = model_params
        self.train_params = train_params
        self.val_params = val_params
        self.test_params = test_params

        # specifically define batch size to be altered by batch size finder
        self.param_container: Dict[NumericParameter] = {}
        self.train_batch_size = self.setup_parameter(train_params.batch_size, dtype=int)

        self.env = env
        self.policy = policy

        self.instance_ids = [g.id for g in self.env.generators]
        self.monitor_instance = train_params.monitor_instance or self.instance_ids[0]
        self.validation_step_rewards = defaultdict(list) # {g.id: [] for g in self.env.generators}

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

    @property
    def world_size(self):
        return self.trainer.world_size
    
    @property
    def test_data_dirs(self):
        dir_fn = lambda instance: os.path.join(self.test_params.data_dir, instance, "td_data.pth")
        return {f"luttmann/{g.id}": dir_fn(g.id) for g in self.env.generators}

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

    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError()

    # PyTorch Lightning's built-in validation_step method
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch, instance_id = batch
        state = self.env.reset(batch)
        reward = self.policy(state, self.env)["reward"]
        val_set_name = self.val_set_names[dataloader_idx]

        if val_set_name == "synthetic":
            self.validation_step_rewards[instance_id].append(reward)
            if instance_id == self.monitor_instance:
                metric_name = f"val/reward"
            else:
                metric_name = f"val/reward/{instance_id}"
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
        batch, _ = batch
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
    
    def __setattr__(self, name, value):
        """Intercept attribute setting to hook parameter setup."""
        if isinstance(value, NumericParameter):
            self.log(f"parameter/{name}", value=value.val, rank_zero_only=True, sync_dist=True)
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
    
    def _get_dataloader(self, params: Union[TrainingParams, ValidationParams, TestParams]):

        dataloader = EnvLoader(
            env=self.env,
            batch_size=params.batch_size,
            dataset_size=params.dataset_size,
            reload_every_n=1,
            mode="val",
            dataset_distribution=self.generator_distribution,
        )

        return dataloader

    def train_dataloader(self):
        return EnvLoader(
            env=self.env, 
            batch_size=self.train_batch_size, 
            dataset_size=self.train_params.dataset_size,
            reload_every_n=1,
            mode="train",
            dataset_distribution=self.generator_distribution
        )
    
    def val_dataloader(self):
        val_dl = self._get_dataloader(self.val_params)
        return val_dl
    
    def test_dataloader(self):
        try:
            test_file_dls = get_file_dataloader(
                self.env, 
                self.test_params.batch_size, 
                self.test_data_dirs,
                num_agents=self.model_params.policy.env.num_agents
            )
        except FileNotFoundError as e:
            self.pylogger.warning(e.__str__())
            test_file_dls = {}

        test_dl = self._get_dataloader(self.test_params)
        test_file_dls["synthetic"] = test_dl

        keys, values = list(test_file_dls.keys()), list(test_file_dls.values())
        self.test_set_names = keys
        return values

    def _setup_decoding_strategy(self, params: Union[TrainingParams, ValidationParams, TestParams]):
        decoding_params = save_config_to_dict(params.decoding)
        decoding_strategy = decoding_params.pop("decoding_strategy")
        self.policy.set_decode_type(decoding_strategy, **decoding_params)

    def _get_global_validation_rewards(self):
        # Stack the performance metrics on this GPU (without computing the mean yet)
        avg_val_rewards = {}
        for instance, val_step_rewards in self.validation_step_rewards.items():
            local_val_rewards = torch.cat(val_step_rewards, dim=0)

            if self.world_size > 1:
                all_val_rewards = all_gather_w_padding(local_val_rewards, self.world_size)
                # Concatenate all the gathered performance tensors into one global tensor
                all_val_rewards = torch.cat(all_val_rewards, dim=0)
            else:
                # If using a single GPU, the global performance tensor is just the local tensor
                all_val_rewards = local_val_rewards

            # Compute the average performance across all GPUs (or just this one if single GPU)
            epoch_average = all_val_rewards.mean().item()
            # Reset the list for the next validation epoch
            self.validation_step_rewards[instance].clear()
            avg_val_rewards[instance] = epoch_average

        self.validation_step_rewards = defaultdict(list)
        return avg_val_rewards

    def on_train_epoch_start(self) -> None:
        self.train()
        self._setup_decoding_strategy(self.train_params)

    def on_validation_epoch_start(self) -> None:
        self.eval()
        self._setup_decoding_strategy(self.val_params)
    
    def on_test_epoch_start(self) -> None:
        self.eval()
        self._setup_decoding_strategy(self.test_params)

    # @monitor_lr_changes  # NOTE potential issues with multithreading
    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        self._update_params()
        epoch_averages = self._get_global_validation_rewards()
        if hasattr(self, "reduce_on_plateau"):
            self.reduce_on_plateau.step(epoch_averages[self.monitor_instance])
        return epoch_averages

    @property
    def generator_distribution(self):
        return None


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

    # @monitor_lr_changes
    def on_validation_epoch_end(self):
        epoch_averages = super().on_validation_epoch_end()
        # update learning rate schedulers
        scheduler = self.lr_schedulers()
        scheduler.step()
        return epoch_averages

    def manual_opt_step(self, loss) -> None:
        opt = self.optimizers()
        # opt.zero_grad()
        self.manual_backward(loss)
        if self.train_params.max_grad_norm is not None:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.train_params.max_grad_norm,
                gradient_clip_algorithm="norm",
            )
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
        torch.manual_seed(1234567)
        self.model_params: ModelWithReplayBufferParams
        self.inner_epochs = self.setup_parameter(model_params.inner_epochs)
        self._ref_model_temp = self.setup_parameter(model_params.ref_model_temp)
        # setup reference policy
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.set_decode_type(
            decode_type="sampling", 
            tanh_clipping=train_params.decoding["tanh_clipping"],
            top_p=train_params.decoding["top_p"],
            temperature=self.ref_model_temp
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
        self.rbs = {
            x.id: make_replay_buffer(
                buffer_size=model_params.buffer_size, 
                # batch_size=self.mini_batch_size, 
                device=model_params.buffer_storage_device, 
                priority_key=model_params.priority_key,
                **model_params.buffer_kwargs
            ) 
            for x in self.env.generators
        }
        self.rb_ensamble = ReplayBufferEnsemble(
            *self.rbs.values(),
            num_buffer_sampled=1
        )

        self.init_dist = np.array([1.] + [0.] * (len(self.env.generators) - 1), dtype=float)
        self.tgt_dist = np.array([1 / len(self.env.generators)] * len(self.env.generators), dtype=float)
        
    @property
    def generator_distribution(self):
        step_t = self.current_epoch / max(1, self.trainer.max_epochs - 10)
        return (1-step_t) * self.init_dist + step_t * self.tgt_dist


    @property
    def rb_size(self):
        return sum([len(rb) for rb in self.rb_ensamble._rbs])

    @property
    def num_training_batches(self):
        # one complete iteration through the rb uses ceil(||rb|| / bs) batches. 
        # We multiple this by inner_epochs, to do inner_epochs iters through rb
        num_batches = int(self.inner_epochs * math.ceil(self.rb_size / self.mini_batch_size))
        if self.world_size > 1:
            # in distributed settings, the number of batches determined above could be different
            # among ranks. Since this would cause ddp to fail, we take the max over ranks here
            all_num_batches = all_gather_numeric(num_batches, self.world_size, self.device)
            num_batches = max(all_num_batches).item()
        return num_batches

    @property
    def rollout_batch_size(self):
        return self._rollout_batch_size
    
    @rollout_batch_size.setter
    def rollout_batch_size(self, value):
        self._rollout_batch_size = value

    @property
    def mini_batch_size(self):
        return int(self._mini_batch_size / self.world_size)
    
    @mini_batch_size.setter
    def mini_batch_size(self, value):
        self._mini_batch_size = value

    @property
    def ref_model_temp(self):
        return self._ref_model_temp

    @ref_model_temp.setter
    def ref_model_temp(self, value):
        self.ref_model_temp = value
        self.policy_old.set_decode_type(
            decode_type="sampling", 
            tanh_clipping=self.train_params.decoding["tanh_clipping"],
            top_p=self.train_params.decoding["top_p"],
            temperature=self.ref_model_temp
        )  




class EvalModule(LearningAlgorithm):

    def __init__(
        self, 
        env: MSPRPEnv,
        policy: RoutingPolicy,
        test_params: TestParams
    ) -> None:
        
        super(LearningAlgorithm, self).__init__()
        
        self.env = env
        self.policy = policy
        self.test_params = test_params
        self.test_set_names = ["synthetic"]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("training_step not defined for eval module")
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("validation_step not defined for eval module")
        
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