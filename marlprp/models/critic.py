import torch
import torch.nn as nn
from einops import rearrange
from marlprp.utils.config import PolicyParams
from marlprp.models.policy_args import TransformerParams


class OperationsCritic(nn.Module):
    def __init__(self, policy_params: PolicyParams) -> None:
        super().__init__()
        self.input_dim = policy_params.embed_dim
        self.critic = nn.Linear(self.input_dim, 1)

    def forward(self, embeddings, td):
        ops_emb = embeddings["operations"]
        index_tensor = td["next_op"][:, :, None, None].repeat(
            (1, 1, 1, self.input_dim)
        ) 
        # (bs, n_jobs, emb)
        job_emb = torch.gather(ops_emb, dim=2, index=index_tensor).squeeze(dim=2) 
        job_emb_pooled = job_emb.mean(1)
        return self.critic(job_emb_pooled).squeeze(-1)
    


class CLSCritic(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.input_dim = params.embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_dim), requires_grad=True)
        self.num_heads = params.num_heads
        self.encoder = nn.MultiheadAttention(
            embed_dim=self.input_dim, num_heads=params.num_heads, batch_first=True
        )
        self.critic = nn.Linear(self.input_dim, 1)

    def forward(self, embeddings, td):
        ops_emb = embeddings["operations"]
        bs = ops_emb.size(0)
        # (bs, ops)
        cls_emb = self.cls_token.expand(bs, 1, -1)  

        attn_mask = ~rearrange(td["op_scheduled"] + td["pad_mask"], "b j o -> b 1 (j o)")
        attn_mask = torch.logical_or(td["done"][..., None], attn_mask)
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        cls_emb, _ = self.encoder(
            query=cls_emb, 
            key=rearrange(ops_emb, "b j o e -> b (j o) e"), 
            value=rearrange(ops_emb, "b j o e -> b (j o) e"), 
            attn_mask=~attn_mask  # True means: not attend
        )
        cls_emb = cls_emb.squeeze(1)
        # (bs, 1, emb)
        return self.critic(cls_emb).squeeze(-1)
