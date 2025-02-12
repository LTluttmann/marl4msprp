import os
import time
import torch
import hydra
import pyrootutils
from dataclasses import asdict
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils import pylogger

from marlprp.env.env import MultiAgentEnv
from marlprp.algorithms.base import EvalModule
from marlprp.models.policies import RoutingPolicy
from marlprp.utils.utils import hydra_run_wrapper, get_wandb_logger
from marlprp.utils.config import (
    EnvParams,
    TrainingParams, 
    TestParams,
    PolicyParams,
    ModelParams
)


pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)
# A logger for this file
log = pylogger.get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="main")
@hydra_run_wrapper
def main(cfg: DictConfig):
    hc = HydraConfig.get()
    old_cfg_path = [x.path for x in hc.runtime.config_sources if os.path.basename(x.path) == ".hydra"]
    assert len(old_cfg_path) == 1, f"expected one path pointing to the old config, got {len(old_cfg_path)}"
    old_run_path = os.path.split(old_cfg_path[0])[0]
    model_path = os.path.join(old_run_path, "checkpoints", "last.ckpt")
    assert os.path.exists(model_path), f"No checkpoint found in {model_path}"

    if hasattr(cfg, "test_env"):
        from hydra.core.global_hydra import GlobalHydra
        from hydra.experimental import compose, initialize_config_dir
        # Re-initialize Hydra with the main config path to extend it
        GlobalHydra.instance().clear()  # Reset Hydra
        hydra.initialize(config_path="../configs")  # Adjust to your main config location
        # Load the new environment config
        new_cfg = compose(config_name="main.yaml", overrides=[f"env={cfg.test_env}"])
        test_env = OmegaConf.to_container(new_cfg.get("env"))
        test_env.pop("name")
        instance_params = EnvParams.initialize(name=cfg.env.name, **test_env)
    else:
        instance_params = EnvParams.initialize(**cfg.env)
    trainer_params = TrainingParams(**cfg.train)
    test_params = TestParams(**cfg.test)
    policy_params = PolicyParams.initialize(env=instance_params, **cfg.policy)
    model_params = ModelParams.initialize(policy_params=policy_params, **cfg.model)

    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    env = MultiAgentEnv.initialize(params=instance_params)
    policy = RoutingPolicy.initialize(policy_params)

    policy_state_dict = {
        k[len("policy."):]: v 
        for k,v in ckpt["state_dict"].items() 
        if k.startswith("policy.")
    }
    try:
        policy.load_state_dict(policy_state_dict)   
    except:
        from marlprp.models.nn.misc import PositionalEncoding
        pe = PositionalEncoding(256, 0.0, 1000)
        policy_state_dict["encoder.init_embedding.pe.pe"] = pe.state_dict()["pe"]
        policy.load_state_dict(policy_state_dict) 

    model = EvalModule(env, policy, model_params, test_params)

    if cfg.get("logger", None) is not None:
        log.info("Instantiating loggers...")
        logger = get_wandb_logger(cfg, model_params, hc, model, eval_only=True)
    else:
        logger = None
        
    trainer = RL4COTrainer(
        accelerator=trainer_params.accelerator,
        devices=trainer_params.devices,
        logger=logger,
        default_root_dir=hc.runtime.output_dir if hc else None,
    )


    for logger in trainer.loggers:
        hparams = {
            "model": asdict(model_params),
            "test": asdict(test_params)
        }
        logger.log_hyperparams(hparams)

    trainer.test(model)


if __name__ == "__main__":
     start = time.time()
     main()
     duration = time.time() - start
     print("Duration: ", duration)