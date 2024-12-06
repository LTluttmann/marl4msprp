import torch
from einops import rearrange
from tensordict import TensorDict
from torch import nn
from torch.nn.modules import TransformerEncoderLayer

from marlprp.models.nn.init_embeddings import get_init_emb_layer
from marlprp.models.policy_args import TransformerParams
from marlprp.models.nn.misc import Normalization

from .base import BaseEncoder, MatNetEncoderOutput
from .mixed_attention import EfficientMixedScoreMultiHeadAttention


class OperationsSelfAttnBlock(nn.Module):

    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.num_heads = params.num_heads
        self.mha_block = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=params.norm_first,
            batch_first=True,
        )

    def _get_ops_of_same_job_mask(self, td: TensorDict):
        bs, nj, no = td["finish_times"].shape
        # attend on ops belonging to same job
        op_scheduled = td["op_scheduled"]
        # initially, all ops in a job attend to each other
        job_ops_mask = torch.full(
            size=(bs, nj, no, no), 
            fill_value=False,
            dtype=torch.bool,
            device=td.device
        )
        # mask only ops that have been scheduled already in attention
        job_ops_mask[op_scheduled.unsqueeze(2).expand_as(job_ops_mask)] = True
        # hack to avoid nans
        job_ops_mask = job_ops_mask.diagonal_scatter(
            torch.full_like(op_scheduled, fill_value=False),
            dim1=2, dim2=3
        )
        # fuse job dimension into batch dimension, to perform memory efficient block attention
        job_ops_mask = job_ops_mask.view(bs * nj, no, no)
        job_ops_mask = job_ops_mask.repeat_interleave(
            self.num_heads, dim=0
        )
        return job_ops_mask

    def forward(self, ops_emb, td):
        batch_size, num_jobs, num_operations, latent_dim = ops_emb.shape
        job_ops_mask = self._get_ops_of_same_job_mask(td)

        ops_emb = self.mha_block(
            src=ops_emb.view(batch_size * num_jobs, num_operations, latent_dim),
            src_mask=job_ops_mask,
        )

        ops_emb = ops_emb.view(batch_size, num_jobs, num_operations, latent_dim)
        
        return ops_emb


class MatNetEncoderLayer(nn.Module):

    def __init__(self, params: TransformerParams) -> None:
        super().__init__()

        self.norm_first = params.norm_first
        self.ops_mha = OperationsSelfAttnBlock(params)
        self.ma_mha = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=self.norm_first,
            batch_first=True,
        )

        self.cross_attn = EfficientMixedScoreMultiHeadAttention(params)

        self.op_norm = Normalization(embed_dim=params.embed_dim, normalization="instance")
        self.ma_norm = Normalization(embed_dim=params.embed_dim, normalization="instance")

        # TODO test
        if self.norm_first:
            self.op_out_norm = Normalization(embed_dim=params.embed_dim, normalization="instance")
            self.ma_out_norm = Normalization(embed_dim=params.embed_dim, normalization="instance")


    def forward(
        self, 
        ops_emb, 
        ma_emb, 
        td,
        cost_mat=None, 
    ):
        # (bs, num_job, num_ma)
        cross_mask = td["proc_times"] > 0 & ~td["op_scheduled"][..., None]
        cross_mask = rearrange(cross_mask, "b j o m -> b (j o) m")

        # get problem sizes
        bs, n_jobs, n_ops, emb_dim = ops_emb.shape

        # (bs, num_job*num_ma, emb)
        ops_emb_flat = ops_emb.view(bs, n_jobs * n_ops, emb_dim)

        #### CROSS ATTENTION ####

        if self.norm_first:
            ops_emb_out_flat, ma_emb_out = self.cross_attn(
                self.op_norm(ops_emb_flat), 
                self.ma_norm(ma_emb), 
                attn_mask=cross_mask, 
                cost_mat=cost_mat
            )
            
            #### SKIP CONN AND NORM ####
            ops_emb_out_flat = ops_emb_out_flat + ops_emb_flat
            ma_emb_out = ma_emb_out + ma_emb

        else:
            ops_emb_out_flat, ma_emb_out = self.cross_attn(
                ops_emb_flat, 
                ma_emb, 
                attn_mask=cross_mask, 
                cost_mat=cost_mat
            )
            
            #### SKIP CONN AND NORM ####
            ops_emb_out_flat = self.op_norm(ops_emb_out_flat + ops_emb_flat)
            ma_emb_out = self.ma_norm(ma_emb_out + ma_emb)

        #### SELF ATTENTION ####

        # (bs, num_jobs, ops_per_job, emb)
        ops_emb_out = ops_emb_out_flat.view(
            bs, n_jobs, n_ops, emb_dim
        ).contiguous()
        ops_emb_out = self.ops_mha(ops_emb_out, td)

        # (bs, num_ma, emb)
        ma_emb_out = self.ma_mha(ma_emb_out)

        if self.norm_first:
            ops_emb_out = self.op_out_norm(ops_emb_out.view(bs, n_jobs * n_ops, emb_dim)).view(
                bs, n_jobs, n_ops, emb_dim
            ).contiguous()
            ma_emb_out = self.ma_out_norm(ma_emb_out)

        return ops_emb_out, ma_emb_out



class MatNetEncoder(BaseEncoder):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads
        self.init_embedding = get_init_emb_layer(params)
        self.encoder = nn.ModuleList([])
        for _ in range(params.num_encoder_layers):
            self.encoder.append(MatNetEncoderLayer(params))

    def forward(self, td: TensorDict) -> MatNetEncoderOutput:
        # (bs, jobs, ops, emb); (bs, ma, emb); (bs, jobs*ops, ma)
        ops_embed, ma_embed, edge_feat = self.init_embedding(td)
        

        for layer in self.encoder:
            ops_embed, ma_embed = layer(
                ops_embed, 
                ma_embed, 
                td,
                cost_mat=edge_feat, 
            )

        return TensorDict(
            {"operations": ops_embed, "machines": ma_embed}, 
            batch_size=td.batch_size
        )
