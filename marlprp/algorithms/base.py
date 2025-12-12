import os
import gc
import math
import copy
import time
import wandb
import torch
import numpy as np

from dataclasses import asdict
from collections import defaultdict
from rich.logging import RichHandler
from tensordict import TensorDict
from torch.optim import Adam, AdamW
from lightning import LightningModule
from typing import Dict, Type, Any, Union
from omegaconf import ListConfig, DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.loggers.wandb import WandbLogger
from torchrl.data import LazyTensorStorage, ReplayBufferEnsemble, TensorDictReplayBuffer

from marlprp.env.env import MultiAgentEnv
from marlprp.models.policies import RoutingPolicy
from marlprp.utils.scheduler import create_scheduler
from marlprp.utils.logger import get_lightning_logger
from marlprp.utils.dataset import EnvLoader, get_file_dataloader
from marlprp.utils.ops import all_gather_w_padding, all_gather_numeric
from marlprp.algorithms.utils import NumericParameter, make_replay_buffer, AlterParamOnPlateau
from marlprp.algorithms.model_args import ModelParams, ModelWithReplayBufferParams
from marlprp.utils.config import (
    EnvParams,
    PolicyParams,
    TrainingParams, 
    ValidationParams, 
    TestParams, 
    config_to_dict
)
from marlprp.algorithms.losses import ce_loss
from .utils import GradNormAccumulator, generate_env_curriculum, get_grad_norm


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
        env: MultiAgentEnv,
        policy: RoutingPolicy,
        model_params: ModelParams,
        train_params: TrainingParams = None,
        val_params: ValidationParams = None,
        test_params: TestParams = None,
    ) -> None:
        
        super(LearningAlgorithm, self).__init__()
        # set env and policy
        self.env = env
        self.policy = policy

        self.pylogger = logger
        # self.save_hyperparameters(asdict(model_params))
        self.model_params = model_params
        self.train_params = train_params
        self.val_params = val_params
        self.test_params = test_params

        self.param_container: Dict[str, NumericParameter] = {}
        self.instance_ids = [g.id for g in self.env.generators]
        # init best rewards achieved so far
        self.best_rewards = {instance: float("-inf") for instance in self.instance_ids}
        self.best_avg_reward = float("-inf")
        self.test_set_names = None
        self.max_mem = {'train': 0, 'val': 0, 'test': 0}
        if train_params is not None:
            self._init_for_training()


    def _init_for_training(self):
        self.monitor_instance = self.train_params.monitor_instance or self.instance_ids[0]
        self.validation_step_rewards = defaultdict(list)
        self.val_set_names = ["synthetic"]
        # decide whether or not to use gradient normalizer when accumulating batches of different instance types
        if (
            len(self.instance_ids) > 1 
            and self.train_params.accumulate_grad_batches > 1 
            and self.train_params.norm_curriculum_grad
        ):
            self.grad_accumulator = GradNormAccumulator(self.policy)

        
    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        model_registry[cls.name] = cls

    def _get_optimizer(self):
        optimizer_kwargs = config_to_dict(self.train_params.optimizer_kwargs)
        use_adamw = optimizer_kwargs.pop("use_adamw", False)
        # rename learning rate key
        optimizer_kwargs["lr"] = optimizer_kwargs.pop("policy_lr")
        policy_params = self.policy.parameters()
        if use_adamw:
            optimizer = AdamW(policy_params, **optimizer_kwargs)
        else:    
            optimizer = Adam(policy_params, **optimizer_kwargs)
        
        return optimizer

    @property
    def world_size(self):
        return self.trainer.world_size

    @classmethod
    def initialize(
        cls,
        env: MultiAgentEnv,
        policy: RoutingPolicy,
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
        checkpoint_path: str,
        env_params: EnvParams = None,
        model_params: ModelParams = None,
        train_params: TrainingParams = None, 
        val_params: ValidationParams = None, 
        test_params: TestParams = None,
    ) -> tuple["LearningAlgorithm", ModelParams]:
        ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        # get old parameters
        try:
            model_params_ckpt: ModelParams = ckpt["hyper_parameters"]["model_params"]
        except KeyError:

            def load_config(config_path: str, config_name: str, overrides=None):
                from hydra.core.global_hydra import GlobalHydra
                from hydra import initialize_config_dir, compose
                # Clear Hydra singleton so we can re-initialize
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()

                config_path = os.path.join(os.environ["PROJECT_ROOT"], config_path)

                with initialize_config_dir(config_path): 
                    cfg = compose(config_name=config_name)
                return cfg
            
            path_to_config = os.path.join(os.path.split(os.path.split(checkpoint_path)[0])[0], ".hydra")
            # with open(path_to_config, "r") as f: config = yaml.safe_load(f)
            cfg = load_config(config_path=path_to_config, config_name="config.yaml")
            if env_params is None:
                env_params = EnvParams(**cfg.env)
            policy_params = PolicyParams.initialize(env=env_params, **cfg["policy"])
            model_params_ckpt = ModelParams.initialize(policy_params, **cfg["model"])

        if model_params is None:
            model_params = model_params_ckpt
        else:
            assert model_params.policy.policy == model_params_ckpt.policy.policy, \
            "Policy of checkpoint and the one specified in new model parameters diverge."

        env = MultiAgentEnv.initialize(params=env_params)
        policy = RoutingPolicy.initialize(params=model_params.policy)

        cls = model_registry[model_params.algorithm]

        model = cls.load_from_checkpoint(
            checkpoint_path,
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
    def validation_step(self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0):
        batch, instance_id = batch
        state = self.env.reset(batch)
        _, reward = self.policy.generate(state, self.env)
        val_set_name = self.val_set_names[dataloader_idx]

        if val_set_name == "synthetic":
            self.validation_step_rewards[instance_id].append(reward)
            metric_name = f"val/synthetic/{instance_id}/reward"
        else:
            metric_name = f"val/files/{val_set_name}/reward"

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
    def test_step(self, batch: tuple[TensorDict, str], batch_idx: int, dataloader_idx: int = 0):
        test_set_name = self.test_set_names[dataloader_idx]
        if batch_idx == 0:
            self.pylogger.info(f"Start testing on set {test_set_name}")

        batch, _ = batch
        state = self.env.reset(batch)
        # recoding time for logging
        start_time = time.time()
        _, test_reward = self.policy(state, self.env)
        step_duration = time.time() - start_time
        self.log(
            f"test/{test_set_name}/reward", 
            test_reward.mean(), 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True,
            add_dataloader_idx=False,
        )
        if batch.size(0) == 1:
            self.log(
                f"duration/{test_set_name}",
                step_duration, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True,
                add_dataloader_idx=False,
            )

    # PyTorch Lightning's built-in configure_optimizers method
    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        return_dict = {"optimizer": optimizer}

        if self.train_params.lr_scheduler is not None:
            lr_scheduler = create_scheduler(
                optimizer, 
                **self.train_params.lr_scheduler,  
            )
            return_dict["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "frequency": self.train_params.lr_scheduler_interval,
                "monitor": self.train_params.lr_scheduler_monitor,
            }

        if self.train_params.lr_reduce_on_plateau_patience:
            self.reduce_on_plateau = ReduceLROnPlateau(
                optimizer=optimizer,
                mode="max",
                factor=0.5,
                patience=self.train_params.lr_reduce_on_plateau_patience
            )

        return return_dict
    
    def __setattr__(self, name, value):
        """Intercept attribute setting to hook parameter setup."""
        if isinstance(value, NumericParameter):
            self.log(f"parameter/{name}", value=value.val, rank_zero_only=True, sync_dist=True)
            self.param_container[name] = value
            return super().__setattr__(name, value.val)
        super().__setattr__(name, value)

    def setup_parameter(
        self, 
        param: Union[int, float, str, Dict],
        dtype: Type = None
    ) -> NumericParameter:
        
        if param is None:
            return param
        elif isinstance(param, (DictConfig, dict)):
            return NumericParameter(**param, dtype=dtype)
        elif isinstance(param, (ListConfig, list, tuple)):
            min_val, max_val, update_coef = param
            val = max_val if update_coef < 1 else min_val
            return NumericParameter(val, dtype=dtype)
        else:
            return NumericParameter(param, dtype=dtype)

    def _update_params(self, metric):
        for name, param in self.param_container.items():
            setattr(self, name, param.update(metric))
    
    def _get_dataloader(self, params: Union[TrainingParams, ValidationParams, TestParams], dataset_dist = None):

        try:
            dataloader = get_file_dataloader(
                self.env, 
                params.batch_size, 
                params.data_dir,
                params.num_file_instances,
                **params.file_loader_kwargs
            )
        except FileNotFoundError:
            dataloader = {}

        if params.dataset_size > 0:
            synthetic_dl = EnvLoader(
                env=self.env,
                batch_size=params.batch_size,
                dataset_size=params.dataset_size,
                reload_every_n=1,
                dataset_distribution=dataset_dist,
            )
            dataloader["synthetic"] = synthetic_dl

        if len(dataloader) == 0:
            self.pylogger.warning(f"No dataset for parameters {params.__class__.__name__}")

        return dataloader

    @property
    def train_batch_size(self):
        return self.train_params.batch_size
    
    @train_batch_size.setter
    def train_batch_size(self, value):
        # specifically define batch size to be altered by batch size finder
        self.train_params.batch_size = value

    def train_dataloader(self):
        return EnvLoader(
            env=self.env, 
            batch_size=self.train_batch_size, 
            dataset_size=self.train_params.dataset_size,
            reload_every_n=1,
            dataset_distribution=self.generator_distribution
        )
    
    def val_dataloader(self):
        val_dls = self._get_dataloader(self.val_params, dataset_dist=None)
        val_dl_names, val_dls = list(val_dls.keys()), list(val_dls.values())
        self.val_set_names = val_dl_names
        return val_dls
    
    def test_dataloader(self):
        test_dls = self._get_dataloader(self.test_params, dataset_dist=None)
        test_dl_names, test_dls = list(test_dls.keys()), list(test_dls.values())
        self.test_set_names = test_dl_names
        return test_dls

    def _setup_decode_type(self, params: Union[TrainingParams, ValidationParams, TestParams]):
        self.policy.set_decode_type(params.decoding)

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
        self._setup_decode_type(self.train_params)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_epoch_start(self) -> None:
        self.eval()
        self._setup_decode_type(self.val_params)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def on_test_epoch_start(self) -> None:
        self.eval()
        self._setup_decode_type(self.test_params)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        
        epoch_averages = self._get_global_validation_rewards()
        new_avg_reward = self.average_set_metric(epoch_averages)

        improved = False
        self._update_params(new_avg_reward)
        if self.best_avg_reward < new_avg_reward:
            result_string = "\n".join([
                f"{idx}: {round(-self.best_rewards[idx], 2)} -> {round(-reward, 2)}" 
                for idx, reward in epoch_averages.items()
            ])
            self.pylogger.info(f"Improved val reward: \n{result_string}")
            self.best_avg_reward = new_avg_reward
            improved = True

        if hasattr(self, "reduce_on_plateau"):
            self.reduce_on_plateau.step(new_avg_reward)

        # update best known rewards
        new_best_rewards = {
            instance: epoch_average
            for instance, epoch_average in epoch_averages.items()
            if self.best_rewards[instance] < epoch_averages[instance]
        }
        self.best_rewards.update(new_best_rewards)
        self.log("val/reward/avg", new_avg_reward, prog_bar=True, sync_dist=True)
        # clear cache
        gc.collect()
        torch.cuda.empty_cache()
        self._log_gpu_memory('val')
        return epoch_averages, improved

    @property
    def generator_distribution(self):
        env_sizes = [gen.size for gen in self.env.generators]
        epochs_until_uniform = self.trainer.max_epochs // 2
        if self.current_epoch < epochs_until_uniform and len(env_sizes) > 1:
            all_dists = generate_env_curriculum(env_sizes, epochs_until_uniform)
            if self.current_epoch == 0 and self.logger and isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "MultiAgentEnv Probabilities until uniform": wandb.plot.line_series(
                            xs=list(range(epochs_until_uniform)),
                            ys=all_dists.T,
                            keys=[g.id for g in self.env.generators],
                            title="MultiAgentEnv Probabilities",
                            xname="Epoch"
                        )
                    },
                )   
                
            return all_dists[self.current_epoch]
        else:
            return np.ones_like(env_sizes) / len(env_sizes)
    

    
    @staticmethod
    def average_set_metric(epoch_averages):
        rewards = torch.tensor(list(epoch_averages.values()))
        scaled_rewards = torch.softmax(-torch.log(-rewards), dim=0) * rewards
        return scaled_rewards.nan_to_num(nan=0).sum() / scaled_rewards.isfinite().sum()
    
    def state_dict(self, *args, **kwargs):
        # Get the original state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Augment with metadata
        state_dict["best_rewards"] = self.best_rewards
        state_dict["best_avg_reward"] = self.best_avg_reward
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Extract metadata if it exists
        self.best_rewards = state_dict.pop("best_rewards", getattr(self, "best_rewards", None))
        self.best_avg_reward = state_dict.pop("best_avg_reward", getattr(self, "best_avg_reward", None))
        super().load_state_dict(state_dict, *args, **kwargs)

    def _log_gpu_memory(self, phase: str):
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.max_mem[phase] = mem_mb

            # Log both to Lightning & W&B
            self.log(f'max_gpu_mem_{phase}_MB', mem_mb, prog_bar=False, rank_zero_only=True)
            if self.logger and hasattr(self.logger.experiment, "log"):
                self.logger.experiment.log({f"max_gpu_mem_{phase}_MB": mem_mb})

    def on_train_epoch_end(self):
        self._log_gpu_memory('train')

    def on_test_epoch_end(self):
        self._log_gpu_memory('test')


class ManualOptLearningAlgorithm(LearningAlgorithm):
    def __init__(
            self, 
            env: MultiAgentEnv, 
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

    def on_validation_epoch_end(self):
        # update learning rate schedulers
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        return super().on_validation_epoch_end()

    def manual_opt_step(self, loss, instance_id: str = None) -> None:
        opt = self.optimizers()
        self.manual_backward(loss)

        # Step counter for accumulation
        if not hasattr(self, "_grad_accum_count"):
            self._grad_accum_count = 0

        if hasattr(self, "grad_accumulator"):
            # Accumulate normalized gradient for this difficulty
            self.grad_accumulator.accumulate()

        self._grad_accum_count += 1

        if self.model_params.log_grad_norm:
            grad_norm = get_grad_norm(self.policy)
            instance_id = instance_id or self.env.generators[0].id
            self.log(f"train/grad_norm/{instance_id}", grad_norm, on_step=False, on_epoch=True)

        if self._grad_accum_count % self.train_params.accumulate_grad_batches == 0:

            if hasattr(self, "grad_accumulator"):
                # Average the stored gradients and assign them to the policy
                self.grad_accumulator.average_and_assign()

            if self.train_params.max_grad_norm is not None:
                # optionally clip gradient magnitude
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.train_params.max_grad_norm,
                    gradient_clip_algorithm="norm",
                )
            
            opt.step()
            opt.zero_grad()
            self._grad_accum_count = 0  # Reset counter


class LearningAlgorithmWithReplayBuffer(ManualOptLearningAlgorithm):
    def __init__(
        self, 
        env: MultiAgentEnv,
        policy: RoutingPolicy,
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
        self.inner_epochs = self.setup_parameter(model_params.inner_epochs, dtype=int)

        # setup reference policy
        self.ref_policy = copy.deepcopy(self.policy)
        self.ref_policy.set_decode_type(model_params.ref_model_decoding)

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

        pin_memory = True if model_params.buffer_storage_device == "cpu" else False
        # make replay buffer
        self.rbs: dict[str, TensorDictReplayBuffer] = {
            x.id: make_replay_buffer(
                buffer_size=model_params.buffer_size, 
                device=model_params.buffer_storage_device, 
                priority_key=model_params.priority_key,
                pin_memory=pin_memory,
                **model_params.buffer_kwargs
            ) 
            for x in self.env.generators
        }
        self.rb_ensamble = ReplayBufferEnsemble(
            *self.rbs.values(),
            num_buffer_sampled=1,
            batch_size=self.mini_batch_size,
            prefetch=2 if model_params.buffer_storage_device == "cpu" else None,
            pin_memory=pin_memory,
        )


    def clear_buffer(self, reinitialize: bool = False):
        self.pylogger.info(f"Emptying replay buffer of size {self.rb_size}")
        for key, rb in self.rbs.items():
            rb.empty()
            if reinitialize:
                rb._storage.initialized = False
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_end(self):
        self.clear_buffer()
        super().on_train_end()

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



class EvalModule(LearningAlgorithm):
    training_step = LightningModule.training_step
    validation_step = LightningModule.validation_step
    
    def __init__(
        self, 
        env: MultiAgentEnv,
        policy: RoutingPolicy,
        model_params: ModelParams,
        test_params: TestParams
    ) -> None:
        
        super(LearningAlgorithm, self).__init__()


        self.pylogger = logger
        self.env = env
        self.policy = policy
        self.test_params = test_params
        self.model_params = model_params
        self.test_set_names = ["synthetic"]
        self.max_mem = {'test': 0}

    @classmethod
    def init_from_checkpoint(cls, checkpoint_path, env_params = None, model_params = None, train_params = None, val_params = None, test_params = None):
        assert test_params is not None
        model, model_params = super().init_from_checkpoint(checkpoint_path, env_params, model_params, train_params, val_params, test_params)
        return cls(
            model.env,
            model.policy,
            model_params,
            test_params,
        ), model_params

class ActiveSearchModule(EvalModule, ManualOptLearningAlgorithm):

    def __init__(self, env, policy, model_params, test_params):
        super().__init__(env, policy, model_params, test_params)
        self.pretrained_policy_state = copy.deepcopy(self.policy.state_dict())
        self.test_params.select_best = True
        self.as_time_budget = self.test_params.active_search_params.time_budget
        self.as_lr = self.test_params.active_search_params.lr
        self.as_bs = self.test_params.active_search_params.bs
        self.as_inner_epochs = self.test_params.active_search_params.inner_epochs

        # self.validation_step = None
        self.automatic_optimization = False
        # self.train_dataloader = self.test_dataloader
        self.train_params = TrainingParams(
            optimizer_kwargs={"policy_lr": self.as_lr},
            dataset_size=1,
            batch_size=1,
            **asdict(self.test_params.decoding)
        )
        self.init_test_rewards = []
        self.as_test_rewards = []

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx = 0):
        self.rb = TensorDictReplayBuffer(storage=LazyTensorStorage(1_000, device="auto"))

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        self.policy.load_state_dict(copy.deepcopy(self.pretrained_policy_state))

    def train_dataloader(self):
        test_dls = self._get_dataloader(self.test_params, dataset_dist=None)
        test_set_names, test_dls = list(test_dls.keys()), list(test_dls.values())
        assert len(test_dls) == 1, "active search only supports a single test dataset"
        self.test_set_name = test_set_names[0]
        return test_dls[0]


    @torch.no_grad()
    def _collect_training_data(self, state, batch_idx):
        state_stack = LazyTensorStorage(self.env.max_num_steps, device="auto")
        _, reward, state_stack = self.policy(state, self.env, storage=state_stack)
        if reward.mean() > self.curr_best_rew:
            self.pylogger.info(f"Improved performance for instance {batch_idx} during active search: {self.curr_best_rew} -> {reward.mean()}")
            if self.curr_best_rew == -torch.inf:
                self.init_test_rewards.append(reward.mean().item())
            # clear current buffer
            self.clear_buffer()
            # updateu best reward
            self.curr_best_rew = reward.mean()
            # flatten so that every step is an experience
            state_stack = state_stack.reshape(-1).contiguous()
            # filter out steps where the instance is already in terminal state. There is nothing to learn from
            state_stack = state_stack[~state_stack["state"].done]
            if hasattr(self.env, "augment_states"):
                state_stack = self.env.augment_states(state_stack)
            # put new best solution to experience replay buffer
            self.rb.extend(state_stack)
        else:
            # did not improve, keep current experience buffer and use it for fine tuning
            pass


    def training_step(self, batch: TensorDict, batch_idx: int, dataloader_idx: int = 0):
        batch, _ = batch
        assert batch.size(0) == 1
        state = self.env.reset(batch)
        start_time = time.time()
        self.curr_best_rew = -torch.inf
        while time.time() - start_time < self.test_params.active_search_params.time_budget:
            self._collect_training_data(state, batch_idx)
            for _ in range(self.as_inner_epochs):
                sub_td = self.rb.sample(self.as_bs).clone().to(self.device).squeeze(0)
                # with torch.set_grad_enabled(True):
                # get logp of target policy for pseudo expert actions 
                logp, _, entropy, mask = self.policy.evaluate(sub_td, self.env)
                # (bs)
                loss = ce_loss(logp, entropy, mask=mask).mean()
                self.manual_opt_step(loss)

        self.as_test_rewards.append(self.curr_best_rew.item())
        self.log(
            f"test/{self.test_set_name}/reward", 
            float(self.curr_best_rew), 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self.log(
            f"duration/{self.test_set_name}",
            time.time() - start_time, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def clear_buffer(self):
        self.rb.empty()
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_end(self):
        test_avg_reward = sum(self.as_test_rewards) / len(self.as_test_rewards)
        init_avg_reward = sum(self.init_test_rewards) / len(self.init_test_rewards)
        self.pylogger.info(f"Active search completed. Avg. test reward improved: {init_avg_reward} -> {test_avg_reward}")
        super().on_train_end()