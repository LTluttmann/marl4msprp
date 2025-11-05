import warnings
import torch.nn as nn

from copy import copy
from omegaconf import OmegaConf
from dataclasses import dataclass, field, replace
from typing import Literal, Dict, Type, List, Union, Optional, Any

from marlprp.utils.logger import get_lightning_logger
from marlprp.utils.data import infer_num_storage_locations


MAX_BATCH_SIZE = 32 * 2048  # (https://github.com/facebookresearch/xformers/issues/845)

log = get_lightning_logger(__name__)


model_config_registry: Dict[str, Type['ModelParams']] = {}
policy_config_registry: Dict[str, Type['PolicyParams']] = {}
env_config_registry: Dict[str, Type['PolicyParams']] = {}


def config_to_dict(config_struct: Union[OmegaConf, Dict]):
    try:
        config_struct = OmegaConf.to_container(config_struct)
    except ValueError:
        config_struct = copy(config_struct)
    return config_struct


@dataclass(frozen=True)
class DecodingConfig:
    decode_type: Literal["greedy", "sampling"]
    tanh_clipping: float
    top_p: float
    temperature: float
    num_starts: int = None
    num_augment: int = None
    num_strategies: int = None
    select_best: bool = False
    hybrid_decoding: bool = False
    num_decoding_samples: int = None

    def __post_init__(self):
        # Normalize all fields
        object.__setattr__(self, "num_starts", self._normalize(self.num_starts))
        object.__setattr__(self, "num_augment", self._normalize(self.num_augment))
        object.__setattr__(self, "num_strategies", self._normalize(self.num_strategies))

        if self.num_decoding_samples is not None:
            if self.num_starts > 1:
                assert self.num_samples == self.num_decoding_samples, "Specified both, num_starts and num_decoding_samples. Use only one of the two"
            else:
                num_starts = self.num_decoding_samples // (self.num_strategies * self.num_augment)
                assert num_starts >= 1, "Specified num_decoding_samples smaller than the product of augmentations and strategies. Check config."
                object.__setattr__(self, "num_starts", num_starts)
                if self.num_samples < self.num_decoding_samples:
                    warnings.warn(f"Decoding samples ({self.num_samples}) smaller than specified ({self.num_decoding_samples}). Check config.")
                    

    @staticmethod
    def _normalize(value) -> int:
        # If None → default to 1
        if value is None:
            return 1
        # If not int → try to cast
        if not isinstance(value, int):
            try:
                value = int(value)
            except Exception:
                return 1  # fallback if conversion fails
        # Enforce minimum of 1
        return max(1, value)

    @property
    def num_samples(self) -> int:
        return self.num_starts * self.num_augment * self.num_strategies
    
    def change_decode_type(self, decode_type: Literal["greedy", "sampling"]):
        return replace(self, decode_type=decode_type)



@dataclass
class BaseEnvParams:
    name: str
    id: str = None

    num_agents: int = 1
    num_depots: Union[List, int] = 1
    num_shelves: Union[List, int] = 10
    
    avg_loc_per_sku: int = None
    num_storage_locations: Optional[int] = 20

    min_demand: int = 1
    max_demand: int = 4
    min_supply: int = 1
    max_supply: int = None # will be calculated 
    avg_supply_to_demand_ratio: float = 2

    capacity: int = 6
    
    packing_ratio_penalty: float = 0.0

    goal: str = None
    num_augment: int = None

    def __post_init__(self):

        if self.max_supply is not None and self.avg_supply_to_demand_ratio is not None:
            log.info("Warning! Set both, max_supply and supply_demand_ratio. I will ignore max_supply")
            self.max_supply = None

        if self.goal is None:
            if self.num_agents is None or self.num_agents == 1:
                self.goal = "min-sum"
            else:
                self.goal = "min-max"
        
        if self.id is None:
            self.id = f"{self.num_shelves}s-{self.num_skus}i-{self.num_storage_locations}p"


    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        env_config_registry[cls.name] = cls

    @classmethod
    def initialize(cls, name: str = None, env: str = None, **kwargs):
        try:
            env_name = name or cls.name
            Config = env_config_registry[env_name]
            return Config(**kwargs)
        except KeyError:
            raise ValueError(f"No Config found for environment {env_name}.")
        

@dataclass(kw_only=True)
class EnvParams(BaseEnvParams):
    name: str = "msprp"
    num_skus: Union[List, int] = 3
    size: int = field(init=False)
    use_stay_token: bool = False
    

    def __post_init__(self):
        super().__post_init__()
        self.num_storage_locations = infer_num_storage_locations(
            self.num_skus, 
            self.num_shelves, 
            avg_loc_per_sku=self.avg_loc_per_sku, 
            num_storage_locations=self.num_storage_locations
        )
        self.size = (self.num_shelves + self.num_depots) * self.num_skus

@dataclass(kw_only=True)
class AREnvParams(EnvParams):
    name: str = "ar"


@dataclass(kw_only=True)
class SAEnvParams(EnvParams):
    name: str = "sa"


@dataclass
class EnvParamList:
    
    envs: List[EnvParams] 
    name: str = "msprp"
    id: str = "multi_instance"
    is_multiinstance: bool = True    

    def append(self, item):
        self.envs.append(item)

    def __getitem__(self, index):
        # Allows element access using indexing
        return self.envs[index]

    @property
    def goal(self):
        return self.envs[0].goal
    
    @property
    def num_agents(self):
        return self.envs[0].num_agents
    
    @classmethod
    def initialize(cfg, env_params: dict):
        param_list = []
        for params in env_params.values():
            params = EnvParams.initialize(**params)
            param_list.append(params)
        return cfg(envs=param_list)
    

    def __len__(self):
        # Returns the length of the list
        return len(self.envs)



@dataclass(kw_only=True)
class LargeEnvParams(BaseEnvParams):
    name: str = "xmsprp"

    num_total_skus: Union[List, int] = 200
    max_sku_per_shelf: int = 20
    min_sku_per_shelf: int = 1



@dataclass(kw_only=True)
class PolicyParams:
    policy: str
    env: Type["EnvParams"]
    config: str = None # the name of the config to use (as appears in the config_registry)
    embed_dim: int = 256
    bias: bool = True
    num_encoder_layers: int = 4
    dropout: float = 0.0
    is_multiagent_policy: bool = True
    # to be specified by the learning algorithm
    eval_multistep: bool = field(init=False)
    eval_per_agent: bool = field(init=False)
    use_critic: bool = field(init=False)
    stepwise_encoding: bool = field(init=False)

    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        policy_config_registry[cls.policy] = cls

    @classmethod
    def initialize(cls, policy: str = None, **kwargs):
        try:
            Config = policy_config_registry[policy]
            return Config(**kwargs)
        except KeyError:
            log.info("Policy of type {policy} has no Config. Use default config instead")
            return PolicyParams(policy=policy, **kwargs)

    def get_policy(self, policies: Dict[str, Type['nn.Module']]) -> nn.Module:
        return policies[self.policy](self)

    def __post_init__(self):
        self.config = self.config or self.policy
    


@dataclass(kw_only=True)
class ModelParams:
    
    policy: Type["PolicyParams"]

    # model architecture
    algorithm: str = None
    stepwise_encoding: bool = False
    warmup_params: "ModelParams" = None
    eval_multistep: bool = False
    eval_per_agent: bool = True
    use_critic: bool = False
    log_grad_norm: bool = False

    def __post_init__(self):
        self.policy.stepwise_encoding = self.stepwise_encoding
        self.policy.use_critic = self.use_critic
        self.eval_multistep = self.eval_multistep and self.policy.is_multiagent_policy and (
            self.policy.env.num_agents is None or self.policy.env.num_agents > 1
        )
        self.eval_per_agent = self.eval_per_agent and self.eval_multistep
        self.policy.eval_multistep = self.eval_multistep
        self.policy.eval_per_agent = self.eval_per_agent
    
    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        model_config_registry[cls.algorithm] = cls
    
    @classmethod
    def initialize(
        cls, 
        policy_params: PolicyParams, 
        algorithm: str = None, 
        **kwargs
    ):
        assert algorithm is not None, (f"specify the algorithm to use")
        Config = model_config_registry[algorithm]
        if kwargs.get("warmup_params", None) is not None:
            warmup_params = kwargs.pop("warmup_params")
            warmup_cfg = cls.initialize(policy_params=policy_params, **warmup_params) 
            return Config(policy=policy_params, warmup_params=warmup_cfg, **kwargs)
        else:
            return Config(policy=policy_params, **kwargs)


@dataclass(kw_only=True)
class ModelWithReplayBufferParams(ModelParams):
    inner_epochs: int = None
    mini_batch_size: int = None
    rollout_batch_size: int = None
    buffer_storage_device: Literal["gpu", "cpu"] = "cpu"
    buffer_size: int = 1_000_000
    # replay buffers allow us to gather experience in evaluation mode
    # Thus, memory leakage through growing gradient information is avoided
    stepwise_encoding: bool = True 
    buffer_kwargs: dict = field(default_factory= lambda: {
        "priority_alpha": 0.5,
        "priority_beta": 0.8
    })
    priority_key: str = None
    # ref model decoding
    ref_model_decode_type: Literal["greedy", "sampling"] = "sampling"
    ref_model_top_p: float = 1
    ref_model_tanh_clipping: float = 10.0
    ref_model_temp: float = 1
    ref_model_num_starts: int = None
    ref_model_num_augment: int = None
    ref_model_num_strategies: int = None
    ref_model_select_best: bool = True
    ref_model_hybrid_decoding: bool = False
    ref_model_num_decoding_samples: int = None
    

    @property
    def ref_model_decoding(self):
        return DecodingConfig(
            decode_type=self.ref_model_decode_type,
            top_p=self.ref_model_top_p,
            tanh_clipping=self.ref_model_tanh_clipping,
            temperature=self.ref_model_temp,
            num_starts=self.ref_model_num_starts,
            num_augment=self.ref_model_num_augment,
            num_strategies=self.ref_model_num_strategies,
            select_best=self.ref_model_select_best,
            hybrid_decoding=self.ref_model_hybrid_decoding,
            num_decoding_samples=self.ref_model_num_decoding_samples
        )


@dataclass(kw_only=True)
class PhaseParams:
    #data
    batch_size: int
    dataset_size: int
    data_dir: str = None
    num_file_instances: int = None
    file_loader_kwargs: dict = field(default_factory=dict)

    # decoding
    decode_type: Literal["greedy", "sampling"] = "sampling"
    top_p: float = 1
    tanh_clipping: float = 10.0
    temperature: float = 1
    num_starts: int = None
    num_augment: int = None
    num_strategies: int = None
    select_best: bool = True
    hybrid_decoding: bool = False
    num_decoding_samples: int = None

    @property
    def decoding(self) -> DecodingConfig:
        return DecodingConfig(
            decode_type=self.decode_type,
            top_p=self.top_p,
            tanh_clipping=self.tanh_clipping,
            temperature=self.temperature,
            num_starts=self.num_starts,
            num_augment=self.num_augment,
            num_strategies=self.num_strategies,
            select_best=self.select_best,
            hybrid_decoding=self.hybrid_decoding,
            num_decoding_samples=self.num_decoding_samples
        )



@dataclass
class TrainingParams(PhaseParams):
    # training
    checkpoint: Optional[str] = None
    train: bool = True

    optimizer_kwargs: Dict[str, Any] = field(default_factory= lambda: {
        "policy_lr": 2e-4,
    })
    lr_scheduler: str = None
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory= lambda: {})
    lr_scheduler_interval: int = 1
    lr_scheduler_monitor: str = "val/reward/avg"
    lr_reduce_on_plateau_patience: int = 5
    max_grad_norm: float = 1.
    epochs: int = 10
    accumulate_grad_batches: int = 1
    norm_curriculum_grad: bool = False

    precision: str = "32-true"
    distribution_strategy: str = "auto"
    accelerator: str = "auto"
    devices: List[int] = None
    reload_every_n: int = 1
    
    seed: int = 1234567
    data_dir = None
    monitor_instance: str = None  # instance used for monitoring



@dataclass
class ValidationParams(PhaseParams):
    decode_type: str = "greedy"


@dataclass
class TestParams(PhaseParams):
    devices: Union[str, List[int]] = "auto"
    checkpoint: str = None
    seed: int = 1234567
    gurobi_timeout: int = 3600