import torch
from torch.distributions import Categorical
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import TransformerEncoderLayer
from tensordict import TensorDict
from einops import rearrange, einsum

from marlprp.models.nn.misc import MLP
from marlprp.models.policy_args import TransformerParams
from marlprp.utils.ops import gather_by_index
from marlprp.models.decoder.base import BasePointer
from marlprp.models.encoder.base import MatNetEncoderOutput
from marlprp.models.nn.misc import MHAWaitOperationEncoder
    

class JobMachinePointer(BasePointer):
    """Decodes a job-machine pair given job and machine embeddings"""

    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads
        self.input_dim = 2 * params.embed_dim

        self.final_trf_block = TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=params.norm_first,
            batch_first=True,
        )

        self.mlp = MLP(
            input_dim=self.input_dim,
            output_dim=1,
            num_neurons=[self.input_dim] * params.num_decoder_ff_layers,
        )

        self.is_multiagent_policy = params.is_multiagent_policy
        self.wait_op_encoder = MHAWaitOperationEncoder(self.embed_dim, params=params)

    def forward(
            self, 
            embeddings: MatNetEncoderOutput, 
            td: TensorDict, 
        ):

        ops_emb = embeddings["operations"]
        ma_emb = embeddings["machines"]

        # (bs, n_jobs, emb)
        job_emb = gather_by_index(ops_emb, td["next_op"], dim=2)
        # (bs, 1, emb)
        wait_emb = self.wait_op_encoder(ops_emb, td)
        # (bs, 1+n_jobs, emb)
        job_emb_w_wait = torch.cat((wait_emb, job_emb), dim=1)
        
        # (bs, 1 + n_jobs, n_ma, 2*emb)
        job_ma_embs = fjsp_emb_combine(td, job_emb_w_wait, ma_emb)
        # self attetion
        job_ma_embs = self._sa_block(job_ma_embs, td)

        # (bs, ma, jobs+1)
        job_ma_logits = self.mlp(job_ma_embs).squeeze(-1)
        logit_mask = ~td["action_mask"] # (bs, ma, jobs+1)
        return job_ma_logits, logit_mask
    

    def _sa_block(self, job_ma_embs: torch.Tensor, td: TensorDict):
        bs, nj, nm, emb = job_ma_embs.shape

        attn_mask = self._get_attn_mask(td)

        # transformer layer over final embeddings
        job_ma_embs = self.final_trf_block(
            src=rearrange(job_ma_embs, "b j m d -> b (j m) d"), 
            src_mask=attn_mask
        )
        job_ma_embs = rearrange(job_ma_embs, "b (j m) d -> b m j d", j=nj, m=nm)
        return job_ma_embs
    

    def _get_attn_mask(self, td: torch.Tensor):
        mask = td["action_mask"].clone()
        # NOTE in multiagent settings, wait op is a valid action, thus should attend to other ops
        mask[..., 0] = self.is_multiagent_policy  
        mask = ~rearrange(mask, "b m j -> b (j m)")  # TODO change dims in env alread?
        # get statistics
        bs, n_actions = mask.shape
        # expand self
        attn_mask = (
            mask
            .unsqueeze(1)
            .expand(bs, n_actions, n_actions)
        )
        # make all actions attend to at least themselves
        attn_mask = attn_mask.diagonal_scatter(
            torch.full_like(mask, fill_value=False),
            dim1=-2, dim2=-1
        )
        # make head dim
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        return attn_mask
    

def jssp_emb_combine(td, job_emb, ma_emb=None):
    if ma_emb is not None:
        ma_emb_per_op = einsum(td["ops_ma_adj"], ma_emb, "b m o, b m e -> b o e")
        # (bs, n_j, emb)
        ma_emb_per_job = gather_by_index(ma_emb_per_op, td["next_op"], dim=1)
        # (bs, n_j, 2 * emb)
        job_emb = torch.cat((job_emb, ma_emb_per_job), dim=2)
    return job_emb


def fjsp_emb_combine(td, job_emb, ma_emb):
    n_ma = ma_emb.size(1)
    # (bs, n_jobs, n_ma, emb)
    job_emb_expanded = job_emb.unsqueeze(-2).expand(-1, -1, n_ma, -1)
    ma_emb_expanded = ma_emb.unsqueeze(-3).expand_as(job_emb_expanded)

    # Input of actor MLP
    # shape: [bs, num_jobs, num_mas, 2*emb]
    h_actions = torch.cat((job_emb_expanded, ma_emb_expanded), dim=-1)
    return h_actions
