import torch
from typing import Tuple
from collections import defaultdict
from tensordict.tensordict import TensorDict

from marlprp.env.env import MSPRPEnv
from marlprp.utils.ops import unbatchify
from marlprp.env.instance import MSPRPState
from marlprp.utils.logger import get_lightning_logger
from marlprp.models.encoder.base import MatNetEncoderOutput

from .utils import process_logits

log = get_lightning_logger(__name__)

def get_decoding_strategy(decoding_strategy, **config):
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "beam_search": BeamSearch,
    }

    if decoding_strategy not in strategy_registry:
        log.warning(
            f"Unknown decode type '{decoding_strategy}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    return strategy_registry.get(decoding_strategy, Sampling)(**config)



class DecodingStrategy:
    name = ...

    def __init__(
            self, 
            tanh_clipping: float = None,
            top_p: float = 1.0,
            temperature: float = 1.0,
            num_decoding_samples: float = None,
            only_store_selected_logp: bool = True, 
            select_best: bool = True,
            store: bool = False,
        ) -> None:
        # init buffers
        self.top_p = top_p
        self.training = None
        self.temperature = temperature
        self.logp = defaultdict(list)
        self.actions = defaultdict(list)
        self.select_best = select_best
        self.tanh_clipping = tanh_clipping
        self.num_starts = num_decoding_samples or 0
        self.only_store_selected_logp = only_store_selected_logp
        self.store = store

    def _step(self, logp: torch.Tensor, td: TensorDict, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Must be implemented by subclass")

    def setup(self, training=False):
        # TODO rename _reset and put into hooks
        self.training = training
        self.actions = defaultdict(list)
        self.logp = defaultdict(list)

    #### hooks ####
    def pre_forward_hook(
        self, 
        state: MSPRPState, 
        embeddings: MatNetEncoderOutput, 
        env: MSPRPEnv
    ):
        """called by models, that encode every step. This hook is called before the encoder"""
        self.setup()

    def pre_step_hook():
        """called by models, that encode every step. This hook is called on act() and evaluate()
        functions, that perform only a single step"""
        ...

    def pre_decoding_hook():
        """called by models, that encode only once and then use the generated embeddings to encode
        a complete solution"""
        ...
    


    def post_decoder_hook(self, state: MSPRPState, env: MSPRPEnv):
        """called by all models after a full solution is obtained"""
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
            logp_selected, all_actions, state = self._select_best_start(logp_selected, all_actions, state, env)

        return logp_selected, all_actions, state

    def step(
        self, 
        logp: torch.Tensor, 
        mask: torch.Tensor, 
        td: TensorDict,
        key="action",
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        assert not logp.isinf().all(1).any()

        logp, selected_actions = self._step(logp, mask, **kwargs)
        
        if self.only_store_selected_logp:
            logp = logp.gather(-1, selected_actions.unsqueeze(-1)).squeeze(-1)

        if self.store:
            self.actions[key].append(selected_actions)
            self.logp[key].append(logp)

        return selected_actions, logp

    def _select_best_start(self, logp, actions, state: MSPRPState, env: MSPRPEnv):
        aug_batch_size = logp.size(0)  # num nodes
        batch_size = aug_batch_size // self.num_starts
        rewards = env.get_reward(state)
        _, idx = unbatchify(rewards, self.num_starts).max(1)
        flat_idx = torch.arange(batch_size, device=rewards.device) + idx * batch_size
        return logp[flat_idx], actions[flat_idx], state[flat_idx]
    

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

    def _step(self, logp: torch.Tensor, mask: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # [BS], [BS]
        selected = logp.argmax(1)

        assert not mask.gather(
            1, selected.unsqueeze(1)
        ).data.any(), "infeasible action selected"

        return logp, selected


class Sampling(DecodingStrategy):
    name = "sampling"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(self.temperature, torch.Tensor):
            self._start_specific_temp = True
        else:
            self._start_specific_temp = False

    def _step(self, logp: torch.Tensor, mask: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n_ops, *ma = logp.shape
        probs = logp.exp()
        assert probs.sum(1).allclose(probs.new_ones((bs, *ma)))
        selected = probs.multinomial(1).view(bs, *ma)

        while mask.gather(1, selected.unsqueeze(1)).data.any():
            log.info("Sampled bad values, resampling!")
            selected = probs.multinomial(1).view(bs, *ma)

        return logp, selected
    



class BeamSearch(DecodingStrategy):

    name = "beam_search"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beam_width = self.num_starts
        if self.beam_width <= 1:
            log.warning("Beam width is <= 1 in Beam search. This might not be what you want")

    def setup(self, training):
        super().setup(training)
        self.step_num = 0
        self.log_beam_probs = []
        self.beam_path = []

    def setup_state(self, state: MSPRPState):
        return state.repeat(self.beam_width)

    def _step(self, 
        probs: torch.Tensor, 
        state: MSPRPState, 
        ignore_in_beam=False, 
        penalty=0.0,
        **kwargs
    ):
        
        if ignore_in_beam:
            # when ignoring the respective key in the beam, we select only the MAP per beam parent. 
            # As a consequece, the beam parents for every batch instance are [0,1,2,...,BW-1]. Since
            # we have batches of beams, the indices are [0,0,0...,1,1,1,.....,BW-1,BW-1,BW-1...]
            aug_batch_size = probs.size(0) 
            batch_size = aug_batch_size // self.beam_width
            beam_parent = torch.arange(self.beam_width).repeat_interleave(batch_size)
            self.beam_path.append(beam_parent.to(probs.device))
            probs_selected, selected = probs.max(1)

        else:
            
            if self.training:
                selected, probs_selected, batch_beam_idx = self._make_stochastic_beam_step(probs)
            else:
                selected, probs_selected, batch_beam_idx = self._make_beam_step(probs, penalty)
            # first select the correct state representation according to beam parent
            state = state[batch_beam_idx] 

        self.step_num += 1

        return selected, probs_selected, state

    

    def finalize(self, state):
        """finalize not really necesarry if all infortmation is stored in state.
        """
        

    def _fill_up_beams(self, topk_ind, topk_logp, log_beam_prob):
        """There may be cases where there are less valid options than the specified beam width. This might not be a problem at 
        the start of the beam search, since a few valid options can quickly grow to a lot more options  (if each valid option
        splits up in two more options we have 2^x growth). However, there may also be cases in small instances where simply
        too few options exist. We define these cases when every beam parent has only one valid child and the sum of valid child
        nodes is less than the beam width. In these cases we will the missing child nodes by duplicating the valid ones.
        
        Moreover, in early phases of the algorithm we may choose invalid nodes to fill the beam. We hardcode these options to
        remain in the depot. These options get filtered out in later phases of the beam search since they have a logprob of -inf

        params:
        - topk_ind
        - topk_logp
        - log_beam_prob_hat [BS, num_nodes * beam_width]
        """

        if self.step_num > 0:

            bs = topk_ind.size(0)
            # [BS, num_nodes, beam_width]
            avail_opt_per_beam = torch.stack(log_beam_prob.split(bs), -1).gt(-torch.inf).sum(1)

            invalid = torch.logical_and(avail_opt_per_beam.le(1).all(1), avail_opt_per_beam.sum(1) < self.beam_width)
            if invalid.any():
                mask = topk_logp[invalid].isinf()
                new_prob, new_ind = topk_logp[invalid].max(1)
                new_prob_exp = new_prob[:,None].expand(-1, self.beam_width)
                new_ind_exp = topk_ind[invalid, new_ind][:, None].expand(-1, self.beam_width)
                topk_logp[invalid] = torch.where(mask, new_prob_exp, topk_logp[invalid])
                topk_ind[invalid] = torch.where(mask, new_ind_exp, topk_ind[invalid])

        # infeasible beam may remain in depot. Beam will be discarded anyway in next round
        topk_ind[topk_logp.eq(-torch.inf)] = 0

        return topk_ind, topk_logp


    def _make_beam_step(self, probs: torch.Tensor, penalty=0.0):

        aug_batch_size, num_nodes = probs.shape  # num nodes (with depot)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = torch.arange(0, batch_size).repeat(self.beam_width).to(probs.device)

        # do log transform in order to avoid that impossible actions are chosen in the beam
        # [BS*BW, num_nodes]
        logp = probs.clone().log()

        if self.step_num == 0:
            # [BS, num_nodes]
            log_beam_prob_hat = logp
            log_beam_prob_hstacked = log_beam_prob_hat[:batch_size]

            if num_nodes < self.beam_width:
                # pack some artificial nodes onto logp
                dummy = torch.full((batch_size, (self.beam_width-num_nodes)), -torch.inf, device=probs.device)
                log_beam_prob_hstacked = torch.hstack((log_beam_prob_hstacked, dummy))

            # [BS, BW]
            topk_logp, topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1, sorted=True)

        else:
            # determine the rank of every action per beam (descending order)
            ranks = torch.argsort(torch.argsort(logp, dim=1, descending=True), dim=1)
            # use the rank as penalty so as to promote the best option per beam
            # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
            log_beam_prob = logp + self.log_beam_probs[-1].unsqueeze(1) 
            log_beam_prob_hat = log_beam_prob - torch.nan_to_num(penalty * ranks, posinf=torch.inf, neginf=torch.inf)
            # [BS, num_nodes * BW]
            log_beam_prob_hstacked = torch.cat(log_beam_prob_hat.split(batch_size), dim=1)
            # [BS, BW]
            # _, topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1)
            # NOTE: for testing purposes
            topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1, sorted=True)[1].sort(1)[0]
            # we do not want to keep track of the penalty value, therefore discard it here
            topk_logp = torch.cat(log_beam_prob.split(batch_size), dim=1).gather(1, topk_ind)

        topk_ind, topk_logp = self._fill_up_beams(topk_ind, topk_logp, log_beam_prob_hat)

        # [BS*BW, 1]
        logp_selected = torch.hstack(torch.unbind(topk_logp,1))

        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind,1)) 

        # since we stack the logprobs from the distinct branches, the indices in 
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index
        if penalty == torch.inf:
            ...
            # this can actually happen, if two actions have the same probability. In this case, only
            # one has rank=0 and the other gets penalized towards -inf. torch.max might then chose the
            # other one in each case, leading to diverging results. But this is ok.
            # assert torch.eq(selected, probs.max(1)[1]).all(), (selected, probs.max(1)[1])
        # calc parent this branch comes from
        beam_parent = (topk_ind // num_nodes).int()

        # extract the correct representations from "batch". Consider instances with 10 nodes and 
        # a beam width of 3. If for the first problem instance the corresponding pointers are 1, 
        # 5 and 15, we know that the first two branches come from the first root hypothesis, while 
        # the latter comes from the second hypothesis. Consequently, the first two branches use the
        # first representation of that instance while the latter uses its second representation
        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size

        # ignore logp of MAP estimates
        if penalty < torch.inf:
            self.log_beam_probs.append(logp_selected)

        self.beam_path.append(beam_parent)
        probs_selected = probs[batch_beam_idx].gather(1, selected[:,None])

        return selected, probs_selected, batch_beam_idx



    def _make_stochastic_beam_step(self, probs: torch.Tensor):

        aug_batch_size, num_nodes = probs.shape  # num nodes (with depot)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = torch.arange(0, batch_size).repeat(self.beam_width).to(probs.device)

        # do log transform in order to avoid that impossible actions are chosen in the beam
        # [BS*BW, num_nodes]
        logp = probs.clone().log()

        if self.step_num == 0:
            # [BS, num_nodes]
            log_beam_prob = logp
            log_beam_prob_hstacked = log_beam_prob[:batch_size]
            if num_nodes < self.beam_width:
                # pack some artificial nodes onto logp
                dummy = torch.full((batch_size, (self.beam_width-num_nodes)), -torch.inf, device=probs.device)
                log_beam_prob_hstacked = torch.hstack((log_beam_prob_hstacked, dummy))

        else:
            # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
            log_beam_prob = logp + self.log_beam_probs[-1].unsqueeze(1)
            # [BS, num_nodes * BW]
            log_beam_prob_hstacked = torch.cat(log_beam_prob.split(batch_size), dim=1)

        # [BS, BW]
        topk_ind = torch.multinomial(log_beam_prob_hstacked.exp(), self.beam_width, replacement=False)
        topk_logp = log_beam_prob_hstacked.gather(1, topk_ind)
        topk_ind, topk_logp = self._fill_up_beams(topk_ind, topk_logp, log_beam_prob)
        # [BS*BW, 1]
        logp_selected = torch.hstack(torch.unbind(topk_logp,1))
        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind,1)) 
        # since we stack the logprobs from the distinct branches, the indices in 
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index
        # calc parent this branch comes from
        # [BS*BW, 1]
        beam_parent = (topk_ind // num_nodes).int()
        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size


        self.log_beam_probs.append(logp_selected)
        self.beam_path.append(beam_parent)
        probs_selected = probs[batch_beam_idx].gather(1, selected[:,None])

        return selected, probs_selected, batch_beam_idx
