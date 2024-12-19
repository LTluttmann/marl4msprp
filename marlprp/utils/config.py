import torch.nn as nn

from copy import copy
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Literal, Dict, Type, List, Union, Optional, Any

from marlprp.utils.logger import get_lightning_logger
from marlprp.utils.data import infer_num_storage_locations


MAX_BATCH_SIZE = 32 * 2048  # (https://github.com/facebookresearch/xformers/issues/845)

log = get_lightning_logger(__name__)


model_config_registry: Dict[str, Type['ModelParams']] = {}
policy_config_registry: Dict[str, Type['PolicyParams']] = {}
env_config_registry: Dict[str, Type['PolicyParams']] = {}


def save_config_to_dict(config_struct: Union[OmegaConf, Dict]):
    try:
        config_struct = OmegaConf.to_container(config_struct)
    except ValueError:
        config_struct = copy(config_struct)
    return config_struct


@dataclass
class BaseEnvParams:
    name: str
    id: str = None

    num_agents: int = 1
    num_depots: Union[List, int] = 1
    num_shelves: Union[List, int] = 10
    
    avg_loc_per_sku: int = None
    num_storage_locations: Optional[int] = 20

    min_demand: int = 0
    max_demand: int = 4
    min_supply: int = 1
    max_supply: int = None # will be calculated 
    avg_supply_to_demand_ratio: float = 2

    capacity: int = 6

    is_multi_instance: bool = field(init=False)
    
    packing_ratio_penalty: float = 0.1
    zero_picks_penalty: float = 0.05

    always_mask_depot: bool = False

    goal: str = "min-sum"

    def __post_init__(self):

        if self.max_supply is not None and self.avg_supply_to_demand_ratio is not None:
            log.info("Warning! Set both, max_supply and supply_demand_ratio. I will ignore max_supply")
            self.max_supply = None

        if self.num_agents is None:
            # if num_agents is none, we have one picker per tour. Thus going to depot is only necessary when nothing else can be done
            # (i.e. everything has been collected)
            self.always_mask_depot = True

        if self.num_agents is None or self.num_agents == 1:
            self.goal = "min-sum"
        else:
            self.goal = "min-max"



    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        env_config_registry[cls.name] = cls

    @classmethod
    def initialize(cls, env: str = None, **kwargs):
        try:
            env = env or cls.name
            Config = env_config_registry[env]
            return Config(**kwargs)
        except KeyError:
            raise ValueError(f"No Config found for environment {env}.")
        

@dataclass(kw_only=True)
class EnvParams(BaseEnvParams):
    name: str = "msprp"
    num_skus: Union[List, int] = 3

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.num_shelves, int):
            self.num_storage_locations = infer_num_storage_locations(
                self.num_skus, 
                self.num_shelves, 
                avg_loc_per_sku=self.avg_loc_per_sku, 
                num_storage_locations=self.num_storage_locations
            )
            self.is_multi_instance = False
        else:
            self.is_multi_instance = True


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
    num_encoder_layers: int = 4
    dropout: float = 0.0
    eval_multistep: bool = True # field(init=False)
    eval_per_agent: bool = field(init=False)
    # to be specified by the learning algorithm
    _use_critic: bool = False
    _stepwise_encoding: bool = field(init=False)
    is_multiagent_policy: bool = True
    max_steps: int = None

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
    
    @property
    def stepwise_encoding(self):
        return self._stepwise_encoding
    
    @stepwise_encoding.setter
    def stepwise_encoding(self, value: bool):
        self._stepwise_encoding = value

    @property
    def use_critic(self):
        return self._use_critic
    
    @use_critic.setter
    def use_critic(self, value: bool):
        self._use_critic = value


@dataclass(kw_only=True)
class ModelParams:
    
    policy: Type["PolicyParams"]

    # model architecture
    algorithm: str = None

    train_decode_type: str = "sampling"
    val_decode_type: str = "greedy"
    test_decode_type: str = "greedy"
    stepwise_encoding: bool = False
    ref: str = None
    warmup_params: "ModelParams" = None
    eval_multistep: bool = False
    eval_per_agent: bool = True

    def __post_init__(self):
        self.policy.stepwise_encoding = self.stepwise_encoding
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
    buffer_storage_device: Literal["gpu", "cpu"] = "gpu"
    buffer_size: int = 1_000_000
    # replay buffers allow us to gather experience in evaluation mode
    # Thus, memory leakage through growing gradient information is avoided
    stepwise_encoding: bool = True 
    buffer_kwargs: dict = field(default_factory= lambda: {
        "priority_alpha": 0.5,
        "priority_beta": 0.8
    })
    priority_key: str = None
    ref_model_temp = 1.0
    

@dataclass
class TrainingParams:
    # data
    batch_size: int
    dataset_size: int
    # training
    checkpoint: Optional[str] = None
    train: bool = True

    optimizer_kwargs: Dict[str, Any] = field(default_factory= lambda: {
        "policy_lr": 2e-4,
    })
    lr_scheduler: str = None
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory= lambda: {})
    lr_scheduler_interval: int = 1
    lr_scheduler_monitor: str = "val/reward"
    lr_reduce_on_plateau_patience: int = 3
    max_grad_norm: float = 1.
    epochs: int = 10
    accumulate_grad_batches: int = 1

    precision: str = "32-true"
    distribution_strategy: str = "auto"
    accelerator: str = "auto"
    devices: List[int] = None
    reload_every_n: int = 1
    
    # decoding
    tanh_clipping: float = 10.0
    top_p: float = 1
    temperature: float = 1
    decode_type: Literal["greedy", "sampling"] = "sampling"
    num_decoding_samples: int = None
    warmup_epochs: int = None

    seed: int = 1234567

    def __post_init__(self):
        self.n_devices = len(self.devices)

    @property
    def decoding(self):
        return {
            "decoding_strategy": self.decode_type,
            "tanh_clipping": self.tanh_clipping,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }
    

@dataclass
class ValidationParams:
    #data
    dataset_size: int
    batch_size: int
    data_dir: str = None

    # decoding
    tanh_clipping: float = 10.0
    top_p: float = 1
    temperature: float = 1
    decode_type: Literal["greedy", "sampling"] = "greedy"
    num_decoding_samples: int = None

    @property
    def decoding(self) -> dict:
        return {
            "decoding_strategy": self.decode_type,
            "tanh_clipping": self.tanh_clipping,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "num_decoding_samples": self.num_decoding_samples,
            "store": True
        }

@dataclass
class TestParams:
    # data
    batch_size: int
    dataset_size: int = None
    # safe results
    render: bool = True
    dataset_path: str = None
    # decoding
    tanh_clipping: float = 10.0
    top_p: float = 1
    temperature: float = 1
    decode_type: Literal["greedy", "sampling"] = "sampling"
    num_decoding_samples: int = 100

    data_dir: str = None
    checkpoint: str = None
    seed: int = 1234567

    @property
    def decoding(self) -> dict:
        return {
            "decoding_strategy": self.decode_type,
            "tanh_clipping": self.tanh_clipping,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "num_decoding_samples": self.num_decoding_samples,
            "store": True
        }