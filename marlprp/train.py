import hydra
import logging
import pyrootutils
from dataclasses import asdict
import lightning.pytorch as pl
from omegaconf import DictConfig
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils import instantiate_callbacks
from hydra.core.hydra_config import HydraConfig

from marlprp.env.env import MSPRPEnv
from marlprp.models.policies import RoutingPolicy
from marlprp.utils.logger import get_lightning_logger
from marlprp.algorithms.base import LearningAlgorithm
from marlprp.utils.utils import hydra_run_wrapper, get_wandb_logger
from marlprp.utils.config import (
    EnvParams, 
    EnvParamList,
    PolicyParams,
    ModelParams, 
    TrainingParams, 
    ValidationParams,
    TestParams
)


pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)
log = get_lightning_logger(__name__)


def get_trainer(
        cfg: DictConfig, 
        train_params: TrainingParams,
        model_params: ModelParams,
        hc: HydraConfig,
        model,
    ) -> RL4COTrainer:
    
    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    if cfg.get("logger", None) is not None:
        log.info("Instantiating loggers...")
        logger = get_wandb_logger(cfg, model_params, hc, model)

    devices = train_params.devices
    log.info(f"Running job on GPU with ID {', '.join([str(x) for x in devices])}")
    log.info("Instantiating trainer...")

    trainer = RL4COTrainer(
        accelerator=train_params.accelerator,
        max_epochs=train_params.epochs,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=hc.runtime.output_dir if hc else None,
        strategy=train_params.distribution_strategy,
        precision=train_params.precision,
        gradient_clip_val=train_params.max_grad_norm,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True
    )

    return trainer


@hydra.main(version_base=None, config_path="../configs/", config_name="main")
@hydra_run_wrapper
def main(cfg: DictConfig):
    env_list = getattr(cfg, "env_list", None)
    if env_list is not None:
        instance_params = EnvParamList.initialize(env_list)
    else:
        instance_params = EnvParams.initialize(**cfg.env)
    train_params = TrainingParams(**cfg.train)
    val_params = ValidationParams(**cfg.val)
    test_params = TestParams(**cfg.test)

    hc = HydraConfig.get()
    
    pl.seed_everything(train_params.seed)

    if train_params.checkpoint is not None:
        model, model_params = LearningAlgorithm.init_from_checkpoint(
            instance_params,
            train_params, 
            val_params, 
            test_params
        )
        trainer = get_trainer(cfg, train_params, model_params, hc)

    else:

        policy_params = PolicyParams.initialize(env=instance_params, **cfg.policy)
        model_params = ModelParams.initialize(policy_params=policy_params, **cfg.model)
        env = MSPRPEnv(params=instance_params)
        policy = RoutingPolicy.initialize(policy_params)
       
        model = LearningAlgorithm.initialize(
            env, 
            policy, 
            model_params=model_params,
            train_params=train_params,
            val_params=val_params, 
            test_params=test_params
        )

        trainer = get_trainer(cfg, train_params, model_params, hc, model)
    # send hparams to all loggers

    for logger in trainer.loggers:
        hparams = {
            "model": asdict(model_params),
            "train": asdict(train_params),
            "val": asdict(val_params),
            "test": asdict(test_params)
        }
        logger.log_hyperparams(hparams)

    trainer.fit(model)
    try:
        trainer.test(ckpt_path='best')
    except:
        trainer.test(model)


if __name__ == "__main__":
     main()