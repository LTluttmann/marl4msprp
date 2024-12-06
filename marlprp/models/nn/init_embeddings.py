import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from marlprp.models.policy_args import PolicyParams, TransformerParams
from marlprp.models.nn.misc import PositionalEncodingWithOffset



class MultiAgentInitEmbedding(nn.Module):
    def __init__(self, model_params: TransformerParams):
        super(MultiAgentInitEmbedding, self).__init__()
        self.embed_dim = model_params.embed_dim
        self.scaling_factor = 100 # according to pirnay
        self.init_ops_embed = nn.Linear(2, model_params.embed_dim, bias=False)
        self.pos_encoder = PositionalEncodingWithOffset(dropout=model_params.input_dropout)

    def _shelf_features(self, td):
        next_op = td["next_op"]
        # (bs, jobs, ops)
        proc_times = td["proc_times"]
        # (bs, jobs, ops)
        op_ready_feat = torch.zeros_like(proc_times)
        # (bs, jobs)
        ma_of_op = td["job_next_ma"]
        a_op = td["time_job_ready"]
        a_ma = td["time_ma_ready"].gather(1, ma_of_op)
        # (bs, jobs)
        schedulable_at = torch.maximum(a_op, a_ma)
        schedulable_at_shifted = schedulable_at - schedulable_at.min(1, keepdims=True).values
        schedulable_at_shifted = schedulable_at_shifted.unsqueeze(2).expand_as(op_ready_feat)
        op_ready_feat.scatter_add_(2, next_op.unsqueeze(2), schedulable_at_shifted)
        feats = [
            proc_times / self.scaling_factor,
            op_ready_feat / self.scaling_factor,
        ]
        return torch.stack(feats, dim=-1)

    def _init_ops_embed(self, td: TensorDict):
        ops_feat = self._op_features(td)
        ops_emb = self.init_ops_embed(ops_feat)
        ops_emb = self.pos_encoder(ops_emb, offsets=td["next_op"])
        return ops_emb

    def forward(self, td):
        return self._init_ops_embed(td)
    

class MultiAgentJSSPInitEmbedding(nn.Module):
    def __init__(self, model_params: TransformerParams):
        super(MultiAgentJSSPInitEmbedding, self).__init__()
        self.embed_dim = model_params.embed_dim
        self.scaling_factor = 100 # according to pirnay
        
        self.init_ops_embed = nn.Linear(2, model_params.embed_dim, bias=False)
        self.init_ma_embed = nn.Linear(1, model_params.embed_dim, bias=False)

        self.pos_encoder = PositionalEncodingWithOffset(dropout=model_params.input_dropout)

    def _op_features(self, td):
        next_op = td["next_op"]
        # (bs, jobs, ma)
        proc_times = td["proc_times"]
        # (bs, jobs, ma)
        op_ready_feat = torch.zeros_like(proc_times)
        # (bs, jobs)
        ma_of_op = td["job_next_ma"]
        a_op = td["time_job_ready"]
        a_ma = td["time_ma_ready"].gather(1, ma_of_op)
        # (bs, jobs)
        schedulable_at = torch.maximum(a_op, a_ma)
        schedulable_at_shifted = schedulable_at - schedulable_at.min(1, keepdims=True).values
        schedulable_at_shifted = schedulable_at_shifted.unsqueeze(2).expand_as(op_ready_feat)
        op_ready_feat.scatter_add_(2, next_op.unsqueeze(2), schedulable_at_shifted)
        feats = [
            proc_times / self.scaling_factor,
            op_ready_feat / self.scaling_factor,
        ]
        return torch.stack(feats, dim=-1)

    def _init_ops_embed(self, td: TensorDict):
        ops_feat = self._op_features(td)
        ops_emb = self.init_ops_embed(ops_feat)
        ops_emb = self.pos_encoder(ops_emb, offsets=td["next_op"])
        return ops_emb

    def _init_ma_embed(self, td: TensorDict):
        a_ma = td["time_ma_ready"]
        a_ma_shifted = a_ma - a_ma.min(1, keepdims=True).values
        return self.init_ma_embed(a_ma_shifted.unsqueeze(2))

    def forward(self, td):
        # (bs, jobs, ma, emb)
        ops_emb = self._init_ops_embed(td)
        # (bs, ma, emb)
        ma_emb = self._init_ma_embed(td)
        edge_emb = None
        return ops_emb, ma_emb, edge_emb
    


class FJSPInitEmbedding(nn.Module):
    def __init__(self, model_params: TransformerParams):
        super(FJSPInitEmbedding, self).__init__()
        self.embed_dim = model_params.embed_dim
        self.scaling_factor = model_params.env.max_processing_time
        self.init_ops_embed = nn.Linear(3, model_params.embed_dim, bias=False)
        self.init_ma_embed = nn.Linear(2, model_params.embed_dim, bias=False)
        self.pos_encoder = PositionalEncodingWithOffset(dropout=model_params.input_dropout)

    def _op_features(self, td):
        # (bs, jobs)
        next_op = td["next_op"]
        # (bs, jobs, ops)
        avg_proc_times = td["proc_times"].mean(-1)
        # std_proc_times = td["proc_times"].std(-1)
        num_eligible = td["proc_times"].gt(0).sum(-1)
        num_ma = td["proc_times"].size(-1)
        # (bs, jobs, ops)
        op_ready_feat = torch.zeros_like(avg_proc_times)
        # (bs, jobs)
        schedulable_at = td["time_job_ready"]

        # (bs, jobs)
        schedulable_at_shifted = schedulable_at - schedulable_at.min(1, keepdims=True).values
        schedulable_at_shifted[td["job_done"]] = 0  # mask finished jobs
        schedulable_at_shifted = schedulable_at_shifted.unsqueeze(2).expand_as(op_ready_feat)
        op_ready_feat.scatter_add_(2, next_op.unsqueeze(2), schedulable_at_shifted)
        feats = [
            avg_proc_times / self.scaling_factor,
            # std_proc_times / self.scaling_factor,
            num_eligible / num_ma,
            op_ready_feat / self.scaling_factor,
        ]
        return torch.stack(feats, dim=-1)

    def _init_ops_embed(self, td: TensorDict):
        ops_feat = self._op_features(td)
        ops_emb = self.init_ops_embed(ops_feat)
        ops_emb = self.pos_encoder(ops_emb, offsets=td["next_op"])
        return ops_emb

    def _init_ma_embed(self, td: TensorDict):
        a_ma = td["time_ma_ready"]
        a_ma_shifted = a_ma - a_ma.min(1, keepdims=True).values

        num_eligible = td["proc_times"].gt(0).sum((-3, -2))
        num_remaining_ops = torch.sum(~(td["pad_mask"] + td["op_scheduled"]), dim=(1,2))
        
        ma_feats = torch.stack([
            a_ma_shifted / self.scaling_factor,
            num_eligible / (num_remaining_ops[:, None] + 1e-6)
        ], dim=-1)
        return self.init_ma_embed(ma_feats)

    def forward(self, td):
        return self._init_ops_embed(td)
    
    def forward(self, td):
        # (bs, jobs, ops, emb)
        ops_emb = self._init_ops_embed(td)
        # (bs, ma, emb)
        ma_emb = self._init_ma_embed(td)
        # (bs, jobs, ops, ma)
        edge_emb = rearrange(td["proc_times"], "b j o m -> b (j o) m") / self.scaling_factor
        return ops_emb, ma_emb, edge_emb