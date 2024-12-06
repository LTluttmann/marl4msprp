from typing import Tuple, Union
import logging
from collections import defaultdict
import torch
from tensordict.tensordict import TensorDict
from marlprp.utils.ops import batchify
from marlprp.utils.config import TrainingParams, TestParams
from .utils import process_logits

log = logging.getLogger(__name__)


def get_decoding_strategy(decoding_strategy, **config):
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
    }

    if decoding_strategy not in strategy_registry:
        log.warning(
            f"Unknown decode type '{decoding_strategy}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    return strategy_registry.get(decoding_strategy, Sampling)(**config)



class DecodingStrategy:
    name = "base"

    def __init__(
            self, 
            tanh_clipping: float = None,
            top_p: float = 1.0,
            temperature: float = 1.0,
            num_decoding_samples: float = None,
            only_store_selected_logp: bool = True, 
            select_best: bool = True,
        ) -> None:
        # init buffers
        self.actions = defaultdict(list)
        self.logp = defaultdict(list)

        self.num_starts = num_decoding_samples or 0
        self.temperature = temperature
        self.top_p = top_p
        self.tanh_clipping = tanh_clipping
        self.only_store_selected_logp = only_store_selected_logp
        self.select_best = select_best



    def _step(self, logp: torch.Tensor, td: TensorDict, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Must be implemented by subclass")

    def setup(self, td: TensorDict, env):
        self.actions = defaultdict(list)
        self.logp = defaultdict(list)

        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if self.num_starts > 1:
            # Expand td to batch_size * num_starts
            td = batchify(td, self.num_starts)

        return td, env, self.num_starts

    def post_decoder_hook(self, td, env):

        def stack_and_gather_logp(logp, actions):
            assert (
                len(self.logp) > 0
            ), "No outputs were collected because all environments were done. Check your initial state"
            # logprobs: Log probabilities of actions from the model (bs, seq_len, action_dim).
            all_logp = torch.stack(logp, 1)
            # actions: Selected actions (bs, seq_len).
            all_actions = torch.stack(actions, 1)
            # (bs, seq_len)
            logp_selected = all_logp.gather(-1, all_actions.unsqueeze(-1)).squeeze(-1)
            return logp_selected, all_actions
        
        if not self.only_store_selected_logp:
            ret = map(
                lambda key: stack_and_gather_logp(self.logp[key], self.actions[key]), 
                list(self.actions.keys())
            )
            logp_selected, all_actions = map(lambda x: torch.stack(x, dim=-1), zip(*ret))
            logp_selected = logp_selected.sum(-1)
        else:
            logp_selected = torch.stack(list(map(lambda x: torch.stack(x, dim=-1), self.logp.values())), -1).sum(-1)
            all_actions = torch.stack(list(map(lambda x: torch.stack(x, dim=-1), self.actions.values())), -1)
        
        if self.num_starts > 1 and self.select_best:
            logp_selected, all_actions, td, env = self._select_best_start(logp_selected, all_actions, td, env)

        return logp_selected, all_actions, td, env

    def step(
        self, 
        logp: torch.Tensor, 
        mask: torch.Tensor, 
        td: TensorDict,
        key="action",
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        assert not logp.isinf().all(1).any()

        logp, selected_actions, td = self._step(logp, mask, td, **kwargs)
        
        td.set(key, selected_actions)

        if self.only_store_selected_logp:
            logp = logp.gather(-1, selected_actions.unsqueeze(-1)).squeeze(-1)

        self.actions[key].append(selected_actions)
        self.logp[key].append(logp)

        return td

    def _select_best_start(self, logp, actions, td: TensorDict, env):
        aug_batch_size = logp.size(0)  # num nodes
        batch_size = aug_batch_size // self.num_starts
        rewards = env.get_reward(td)
        _, idx = torch.cat(rewards.unsqueeze(1).split(batch_size), 1).max(1)
        flat_idx = torch.arange(batch_size, device=rewards.device) + idx * batch_size
        return logp[flat_idx], actions[flat_idx], td[flat_idx], env
    

    def logits_to_logp(self, logits, mask=None):
        return process_logits(
            logits,
            mask,
            self.temperature,
            self.top_p, 
            self.tanh_clipping,
        )


class Greedy(DecodingStrategy):
    name = "greedy"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _step(
        self, logp: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        # [BS], [BS]
        selected = logp.argmax(1)

        assert not mask.gather(
            1, selected.unsqueeze(1)
        ).data.any(), "infeasible action selected"

        return logp, selected, td


class Sampling(DecodingStrategy):
    name = "sampling"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _step(
        self, logp: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        bs, n_ops, *ma = logp.shape
        probs = logp.exp()
        assert probs.sum(1).allclose(probs.new_ones((bs, *ma)))
        selected = probs.multinomial(1).view(bs, *ma)

        while mask.gather(1, selected.unsqueeze(1)).data.any():
            log.info("Sampled bad values, resampling!")
            selected = probs.multinomial(1).view(bs, *ma)

        return logp, selected, td