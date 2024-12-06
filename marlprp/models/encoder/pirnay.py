import torch
from einops import rearrange
from tensordict import TensorDict
from torch import nn
from torch.nn.modules import TransformerEncoderLayer

from marlprp.models.nn.init_embeddings import JSSPInitEmbedding
from marlprp.models.policy_args import TransformerParams

from .base import BaseEncoder


class OperationsEncoder(BaseEncoder):

    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads
        self.init_embedding = JSSPInitEmbedding(params)
        self.encoder = nn.ModuleList([])
        for _ in range(params.num_encoder_layers):

            block = TransformerEncoderLayer(
                d_model=params.embed_dim,
                nhead=params.num_heads,
                dim_feedforward=params.feed_forward_hidden,
                dropout=params.dropout,
                activation=params.activation,
                norm_first=params.norm_first,
                batch_first=True,
            )

            self.encoder.append(block)


    def get_ops_on_same_ma_mask(self, td: TensorDict):
        # attend on ops on same machine
        flat_ops_ma_ids = rearrange(td["ops_machine_ids"], "b j o -> b (j o)")
        flat_ops_scheduled = rearrange(td["op_scheduled"], "b j o -> b (j o)")
        ops_machines_mask = ~(flat_ops_ma_ids[:, None] == flat_ops_ma_ids[...,None])
        # exclude scheduled ops in attention mechanism
        ops_machines_mask[flat_ops_scheduled.unsqueeze(1).expand_as(ops_machines_mask)] = True
        # hack to avoid nans
        ops_machines_mask = ops_machines_mask.diagonal_scatter(
            torch.full_like(flat_ops_scheduled, fill_value=False),
            dim1=1, dim2=2
        )
        ops_machines_mask = ops_machines_mask.repeat_interleave(
            self.num_heads, dim=0
        )
        return ops_machines_mask
    
    def get_ops_of_same_job_mask(self, td: TensorDict):
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
    

    def forward(self, td: TensorDict):

        # Take operations, embed them. D = Latent dimension
        operations_embedded = self.init_embedding(td)  # (B, J, O, D)
        batch_size, num_jobs, num_operations, latent_dim = operations_embedded.shape

        ops_machines_mask = self.get_ops_on_same_ma_mask(td)
        job_ops_mask = self.get_ops_of_same_job_mask(td)

        # Pipe the operations through the transformer blocks
        for i, trf_block in enumerate(self.encoder):
            if i % 2 == 0:
                # Even blocks: Operations of the individual jobs attend to the other operations within their job.
                # So we fold the job dimension into the batch dimension.
                operations_embedded = trf_block(
                    src=operations_embedded.view(
                        batch_size * num_jobs, num_operations, latent_dim
                    ),
                    src_mask=job_ops_mask,
                ).view(batch_size, num_jobs, num_operations, latent_dim)
            else:
                # Odd blocks: Operations running on same machine attend to each other
                operations_embedded = trf_block(
                    src=operations_embedded.view(
                        batch_size, num_jobs * num_operations, latent_dim
                    ),
                    src_mask=ops_machines_mask,
                ).view(batch_size, num_jobs, num_operations, latent_dim)

        return TensorDict({"operations": operations_embedded}, batch_size=td.batch_size)
