import hydra
import pyrootutils
from dataclasses import asdict
import lightning.pytorch as pl
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils import pylogger
from lightning.pytorch.utilities.model_helpers import is_overridden
from marlprp.algorithms.base import EvalModule, ActiveSearchModule
from marlprp.utils.utils import hydra_run_wrapper, get_wandb_logger
from marlprp.utils.config import (
    EnvParams,
    EnvParamList,
    TrainingParams, 
    TestParams
)


pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)
# A logger for this file
log = pylogger.get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="main")
@hydra_run_wrapper
def main(cfg: DictConfig):
    
    trainer_params = TrainingParams(**cfg.train)
    test_params = TestParams(**cfg.test)

    hc = HydraConfig.get()
    pl.seed_everything(test_params.seed)


    if test_params.active_search_params is not None:
        PlModule = ActiveSearchModule
    else:
        PlModule = EvalModule

    model, model_params = PlModule.init_from_checkpoint(
        test_params.checkpoint, 
        test_params=test_params,
    )
    
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
        enable_checkpointing=False,
        precision=trainer_params.precision,
        max_epochs=1,
    )


    for logger in trainer.loggers:
        hparams = {
            "model": asdict(model_params),
            "test": asdict(test_params)
        }
        logger.log_hyperparams(hparams)

    if is_overridden("training_step", model):
        trainer.fit(model)
    else:
        trainer.test(model)


if __name__ == "__main__":
     main()