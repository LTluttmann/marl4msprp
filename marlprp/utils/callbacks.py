import copy
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from marlprp.utils.logger import get_lightning_logger
from marlprp.algorithms.base import LearningAlgorithm
from lightning.pytorch.callbacks import BatchSizeFinder


# A logger for this file
log = get_lightning_logger(__name__)


class RolloutBatchSizeFinder(BatchSizeFinder):

    def __init__(
            self, 
            mode: str = "power", 
            steps_per_trial: int = 3, 
            init_val: int = 2, 
            max_trials: int = 25, 
            batch_arg_name: str = "train_batch_size"
    ) -> None:
        
        super().__init__(mode, steps_per_trial, init_val, max_trials, batch_arg_name)
        self._old_rollout_batch_size = None
        self._old_batch_size = None
        

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from lightning.pytorch.tuner.batch_size_scaling import _reset_dataloaders, _adjust_batch_size
        # NOTE power mode may return a new_size, which has not been tested if max_trials has been reached
        log.warning("batch_size_scaling is still buggy, so use with caution")
        # NOTE: set this to true in order to disable some hooks
        if hasattr(pl_module, "rollout_batch_size"):
            self._old_batch_size = getattr(pl_module, self._batch_arg_name)
            # ensure that the whole batch is consumed during a rollout
            setattr(pl_module, "rollout_batch_size", 9999)

        # NOTE bug in pl lightning implementation, which does not call _reset_dataloaders on first iter
        _adjust_batch_size(trainer, self._batch_arg_name, value=self._init_val)
        _reset_dataloaders(trainer)
        super().on_fit_start(trainer, pl_module)
        new_size = self.optimal_batch_size

        if self._old_batch_size is not None:
            new_size = min(new_size, self._old_batch_size)
            _adjust_batch_size(trainer, self._batch_arg_name, value=self._old_batch_size)
            _adjust_batch_size(trainer, "rollout_batch_size", value=new_size)
            _reset_dataloaders(trainer)
    

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return
    

class ValidationScheduler(Callback):
    def __init__(self, warmup_epochs_n=1, reset_policy_after_warmup: bool = False):
        self.warmup_epochs_n = warmup_epochs_n
        self.reset_model_after_warmup = reset_policy_after_warmup
        self.val_check_batch = None
        self._restored = False
        self.policy_state_dict = None
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: LearningAlgorithm):
        if self.warmup_epochs_n < 1:
            self._restored = True
            return
        
        if self.reset_model_after_warmup:
            self.policy_state_dict = copy.deepcopy(pl_module.policy.state_dict())

        self.val_check_batch = trainer.val_check_batch
        trainer.val_check_batch = 1  # warmup
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: LearningAlgorithm) -> None:
        if trainer.current_epoch >= self.warmup_epochs_n and not self._restored:
            log.info("Warmup complete...")
            if self.reset_model_after_warmup:
                log.info("resetting policy to initial state") 
                pl_module.policy.load_state_dict(self.policy_state_dict)
            trainer.val_check_batch = self.val_check_batch
            self._restored = True


class LearningRateWarmupCallback(pl.Callback):
    def __init__(self, warmup_steps: int, initial_lr: float = 1e-7):
        """
        Callback to warm up the learning rate linearly over a specified number of steps.

        Args:
            warmup_steps (int): Number of warmup steps.
            initial_lr (float): Initial learning rate for the warmup.
        """
        super().__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.backward_count = 0

    def on_train_start(self, trainer, pl_module):
        """
        Save the initial learning rate from the optimizer.
        """
        self.target_lrs = [group['lr'] for group in trainer.optimizers[0].param_groups]
        for group in trainer.optimizers[0].param_groups:
            group['lr'] = self.initial_lr

    def on_after_backward(self, trainer, pl_module):
        """
        Adjust the learning rate linearly after each backward pass.
        """
        self.backward_count += 1

        if self.backward_count < self.warmup_steps:
            warmup_factor = self.backward_count / self.warmup_steps
            new_lrs = [self.initial_lr + (target_lr - self.initial_lr) * warmup_factor
                       for target_lr in self.target_lrs]

            for param_group, lr in zip(trainer.optimizers[0].param_groups, new_lrs):
                param_group['lr'] = lr
        elif self.backward_count == self.warmup_steps:
            # Ensure the learning rate is set to the target value after warmup
            for param_group, target_lr in zip(trainer.optimizers[0].param_groups, self.target_lrs):
                param_group['lr'] = target_lr
