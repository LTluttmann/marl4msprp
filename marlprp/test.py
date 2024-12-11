import hydra
import pyrootutils
from dataclasses import asdict
import lightning.pytorch as pl
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils import pylogger

from marlprp.algorithms.base import EvalModule
from marlprp.utils.utils import hydra_run_wrapper, get_wandb_logger
from marlprp.utils.config import (
    EnvParams,
    TrainingParams, 
    TestParams
)


pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)
# A logger for this file
log = pylogger.get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="main")
@hydra_run_wrapper
def main(cfg: DictConfig):
    
    instance_params = EnvParams.initialize(**cfg.env)
    trainer_params = TrainingParams(**cfg.train)
    test_params = TestParams(**cfg.test)

    hc = HydraConfig.get()
    pl.seed_everything(test_params.seed)

    model, model_params = EvalModule.init_from_checkpoint(test_params, instance_params, policy_cfg=cfg.get("policy", None))
    
    logger = get_wandb_logger(cfg, model_params, hc)

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
     main()