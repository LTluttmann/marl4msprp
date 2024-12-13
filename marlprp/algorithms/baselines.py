import copy
import torch
import torch.nn as nn

from marlprp.utils.ops import unbatchify
from marlprp.utils.logger import get_lightning_logger
from marlprp.algorithms.model_args import ReinforceParams

log = get_lightning_logger(__name__)


class Baseline(object):
    _Mapping = {}

    @classmethod
    def register(cls):
        cls._Mapping[cls.name] = cls
    
    @classmethod
    def initialize(cls, baseline: str, *args, **kwargs):
        try:
            Config = cls._Mapping[baseline]
            return Config(*args, **kwargs)
        except KeyError:
            raise KeyError("%s not registered as baseline, only %s are registered" % 
                           (baseline, ",".join([key.__repr__() for key in cls._Mapping.keys()])))


    def eval(self, td, reward, env):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, *args, **kw):
        pass

    def setup(self, *args, **kw):
        pass


class WarmupBaseline(Baseline):

    def __init__(
        self, 
        baseline: Baseline = None, 
        model_params: ReinforceParams = None,
        **kwargs
    ):
        super(WarmupBaseline, self).__init__()
        assert baseline is not None
        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(model_params)
        self.alpha = 0
        self.n_epochs = model_params.bl_warmup_epochs
        assert self.n_epochs > 0, "n_epochs to warmup must be positive"
        log.info("using exponential warmup of the baseline for %s epochs" % self.n_epochs)
        self.current_epoch = 0

    def setup(self, *args, **kw):
        self.baseline.setup(*args, **kw)

    def eval(self, td, reward, env):
        if self.alpha == 1:
            return self.baseline.eval(td, reward, env)
        if self.alpha == 0:
            return self.warmup_baseline.eval(td, reward, env)
        v, loss = self.baseline.eval(td, reward, env)
        vw, lw = self.warmup_baseline.eval(td, reward, env)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * loss + (1 - self.alpha) * lw

    def epoch_callback(self, *args, **kw):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(*args, **kw)
        if self.current_epoch < self.n_epochs:
            self.alpha = (self.current_epoch + 1) / float(self.n_epochs)
            log.info("Set warmup alpha = {}".format(self.alpha))
        self.current_epoch += 1


class NoBaseline(Baseline):

    name = None

    def __init__(self, **kwargs) -> None:
        super(NoBaseline, self).__init__()

    def eval(self, td, reward, env):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    name = "exponential"

    def __init__(self, model_params: ReinforceParams = None, **kwargs):
        super(ExponentialBaseline, self).__init__()

        self.beta = model_params.exp_beta
        self.v = None

    def eval(self, td, reward, env):

        if self.v is None:
            v = reward.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * reward.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss



class RolloutBaseline(Baseline):

    name = "rollout"

    def __init__(self, **kwargs):
        
        super(RolloutBaseline, self).__init__()
        self.best_reward = float("-inf")
        self.model: nn.Module = None
        self.current_epoch = 0

    def setup(self, model, *args, **kwargs):
        self.model = copy.deepcopy(model)

    def eval(self, td, reward, env):
        # set_decode_type(self.model, "greedy")
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            td = env.reset(td)
            v = self.model(td, env)["reward"]

        # There is no loss
        return v, 0

    def epoch_callback(self, model: nn.Module, val_reward: float):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        log.info("Evaluating candidate model on evaluation dataset")
        log.info(
            "Epoch {} candidate mean {:.3f}, mean {:.3f}, difference {:.3f}".format(
            self.current_epoch, val_reward, self.best_reward, val_reward - self.best_reward
            )
        )
        
        if val_reward > self.best_reward:
            self.model.load_state_dict(copy.deepcopy(model.state_dict()))

        self.current_epoch += 1

class POMOBaseline(Baseline):

    name = "pomo"

    def __init__(self, model_params: ReinforceParams = None, **kwargs):
        assert model_params.num_starts > 1, "POMO uses multiple starts per instance"
        self.num_starts = model_params.num_starts

    
    def eval(self, td, c, env):
        # Unbatchify reward to [batch_size, num_starts]
        unbatched_c = unbatchify(c, self.num_starts)
        # [bs]
        bl_val = unbatched_c.mean(1)
        bl_val = bl_val.repeat(self.num_starts)

        return bl_val, 0
    

# TODO refactor using new __init_subclass__ method
NoBaseline.register()
ExponentialBaseline.register()
RolloutBaseline.register()
POMOBaseline.register()