import lightning.pytorch as pl
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import pyrootutils
from rl4co.utils import (
    instantiate_callbacks,
    instantiate_loggers,
    pylogger
)
from lightning.pytorch import seed_everything
from rl4co.utils.trainer import RL4COTrainer

from marlprp import ROOTDIR
from marlprp.env.env import MSPRPEnv
from marlprp.algorithms.base import LearningAlgorithm
from marlprp.models.policies import SchedulingPolicy
from marlprp.utils.config import (
    EnvParams, 
    PolicyParams,
    ModelParams, 
    TrainingParams, 
    ValidationParams,
    TestParams
)

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)
# A logger for this file
log = pylogger.get_pylogger(__name__)


def get_trainer(
    cfg: DictConfig, 
    hc: HydraConfig,
    train_params: TrainingParams, 
    model: pl.LightningModule
):
    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers = instantiate_loggers(cfg.get("logger"), model)

    log.info("Instantiating trainer...")
    # in distributed settings, we generate a dataset once at the start of the epoch and sample from 
    # it. In non-distributed settings, batches are generated on the fly
    reload_dl_every_n = 1 if train_params.n_devices > 1 else 0

    trainer = RL4COTrainer(
        accelerator=train_params.accelerator,
        max_epochs=train_params.epochs,
        devices=train_params.devices,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=hc.runtime.output_dir if hc else None,
        strategy=train_params.distribution_strategy,
        precision=train_params.precision,
        gradient_clip_val=train_params.max_grad_norm,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=reload_dl_every_n,
    )
    return trainer


@hydra.main(version_base=None, config_path="../configs/", config_name="main")
def main(cfg: DictConfig):
    instance_params = EnvParams.initialize(**cfg.env)
    policy_params = PolicyParams.initialize(env=instance_params, **cfg.policy)
    model_params = ModelParams.initialize(
        policy_params=policy_params,
        **cfg.model
    )

    train_params = TrainingParams(**cfg.train)
    val_params = ValidationParams(**cfg.val)
    test_params = TestParams(**cfg.test)

    hc = HydraConfig.get()
    seed_everything(instance_params.seed)

    env = MSPRPEnv(params=instance_params)
    policy = SchedulingPolicy.initialize(policy_params)
    model = LearningAlgorithm.initialize(
        env, 
        policy, 
        model_params=model_params,
        train_params=train_params,
        val_params=val_params, 
        test_params=test_params
    )

    trainer = get_trainer(cfg, hc, train_params, model)
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
     main()