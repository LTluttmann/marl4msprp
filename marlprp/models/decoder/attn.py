import math
import torch
from torch import nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

from marlprp.models.nn.misc import MLP
from marlprp.env.instance import MSPRPState
from marlprp.models.nn.kvl import get_kvl_emb
from marlprp.models.decoder.base import BasePointer
from marlprp.models.nn.context import get_context_emb
from marlprp.models.encoder.base import MatNetEncoderOutput
from marlprp.models.policy_args import TransformerParams, MahamParams



class AttentionPointerMechanism(nn.Module):
    """Calculate logits given query, key and value and logit key.

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        sdp_fn: scaled dot product attention function (SDPA)
        check_nan: whether to check for NaNs in logits
    """

    def __init__(self, params: MahamParams, check_nan=True):
        super(AttentionPointerMechanism, self).__init__()
        self.num_heads = params.num_heads
        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(
            params.embed_dim, params.embed_dim, bias=False
        )
        if params.use_rezero:
            self.resweight = nn.Parameter(torch.tensor(0.))
        else:
            self.norm = LayerNorm(params.embed_dim)
            self.resweight = 1
        self.dropout = nn.Dropout(params.dropout)
        self.check_nan = check_nan

    def forward(self, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)
        # (bs, m, d); NOTE use ReZERO logic here
        glimpse = self.dropout(self.project_out(heads))
        glimpse = query + self.resweight * glimpse
        if hasattr(self, "norm"):
            glimpse = self.norm(glimpse)
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = torch.bmm(glimpse, logit_key.transpose(-2, -1)) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

    def _inner_mha(self, query, key, value, attn_mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if attn_mask is not None:
            # make mask the same number of dimensions as q
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )
        heads = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)


class AttentionPointer(BasePointer):
    def __init__(
        self,
        params: TransformerParams,
        check_nan: bool = True,
        decoder_type: str = None
    ):
        super(AttentionPointer, self).__init__()
        self.model_params = params
        self.head_dim = params.qkv_dim
        self.num_heads = params.num_heads
        self.emb_dim = params.embed_dim
        self.stepwise_encoding = params.stepwise_encoding
        self.pointer = AttentionPointerMechanism(params, check_nan)
        self.context_embedding = get_context_emb(params, key=decoder_type)
        self.kvl_emb = get_kvl_emb(params, key=decoder_type)
        self.agent_ranker = MLP(self.emb_dim, 1, num_neurons=[self.emb_dim, self.emb_dim])


    @property
    def device(self):
        return next(self.parameters()).device

    def compute_cache(self, embs: MatNetEncoderOutput) -> None:
        # shape: 3 * (bs, n, emb_dim)
        self.kvl_emb.compute_cache(embs)

    def forward(self, embs: MatNetEncoderOutput, state: MSPRPState, attn_mask: Tensor = None):

        # (bs, a, emb)
        q = self.context_embedding(embs, state)

        # (bs, heads, nodes, key_dim) | (bs, heads, nodes, key_dim)  |  (bs, nodes, emb_dim)
        k, v, logit_key = self.kvl_emb(embs, state)
        # (b, a, nodes)
        logits = self.pointer(q, k, v, logit_key, attn_mask=attn_mask)
        return logits

