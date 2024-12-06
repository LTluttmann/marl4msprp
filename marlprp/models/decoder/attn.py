import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from tensordict import TensorDict
from einops import rearrange, einsum
import math

from marlprp.models.encoder.base import MatNetEncoderOutput
from marlprp.models.nn.misc import MHAWaitOperationEncoder
from marlprp.models.decoder.base import BasePointer
from marlprp.models.policy_args import TransformerParams, marlprpParams
from marlprp.utils.ops import gather_by_index
from marlprp.models.nn.dynamic import get_dynamic_emb
from marlprp.models.nn.context import get_context_emb



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

    def __init__(self, params: marlprpParams, check_nan=True):
        super(AttentionPointerMechanism, self).__init__()
        self.num_heads = params.num_heads
        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(
            params.embed_dim, params.embed_dim, bias=False
        )
        if params.use_rezero:
            self.resweight = nn.Parameter(torch.rand(1))
        else:
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
        glimpse = (1-self.resweight) * query + self.resweight * glimpse

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

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
        dyn_emb_key: str = None
    ):
        super(AttentionPointer, self).__init__()
        self.model_params = params
        self.head_dim = params.qkv_dim
        self.num_heads = params.num_heads
        self.emb_dim = params.embed_dim
        self.stepwise_encoding = params.stepwise_encoding

        self.Wq = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.Wkvl = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=False)
        self.pointer = AttentionPointerMechanism(params, check_nan)

        self.context_embedding = get_context_emb(params, extra_key=dyn_emb_key)
        self.dynamic_embedding = get_dynamic_emb(params, key=dyn_emb_key)

        self.cache = None
        self.wait_op_encoder = MHAWaitOperationEncoder(self.emb_dim, params)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_cache(self, tgt_emb: Tensor) -> None:
        # shape: 3 * (bs, n, emb_dim)
        self.cache = self.Wkvl(tgt_emb).chunk(3, dim=-1)

    def _compute_kvl(self, td, tgt_emb: Tensor):

        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td, tgt_emb)

        if self.cache is not None:
            # gather static keys, values and logit keys
            wait_emb = self.wait_op_encoder(tgt_emb, td)
            cache = tuple(map(
                lambda x: gather_by_index(x, td["next_op"], dim=2), 
                self.cache
            ))
            k, v, l = tuple(map(lambda x: torch.cat((wait_emb, x), dim=1), cache))
            
        else:
            # dynamically compute keys, values and logit keys
            wait_emb = self.wait_op_encoder(tgt_emb, td)
            tgt_emb = gather_by_index(tgt_emb, td["next_op"], dim=2)
            # (bs, n_jobs + 1, emb)
            tgt_emb_w_wait = torch.cat((wait_emb, tgt_emb), dim=1)
            k, v, l = self.Wkvl(tgt_emb_w_wait).chunk(3, dim=-1)

        k_dyn = k + glimpse_k_dyn
        v_dyn = v + glimpse_v_dyn
        l_dyn = l + logit_k_dyn

        return k_dyn, v_dyn, l_dyn


    def forward(self, embs: MatNetEncoderOutput, td: TensorDict, attn_mask: Tensor = None):
        src_emb = embs["machines"]
        tgt_emb = embs["operations"]
        # (bs, m, emb)
        q = self.context_embedding(td, src_emb)

        # (bs, heads, nodes, key_dim) | (bs, heads, nodes, key_dim)  |  (bs, nodes, emb_dim)
        k, v, logit_key = self._compute_kvl(td, tgt_emb)

        logits = self.pointer(q, k, v, logit_key, attn_mask=attn_mask)

        return logits, attn_mask


class MultiAgentAttentionPointer(AttentionPointer):

    def __init__(self, params: marlprpParams):

        super(MultiAgentAttentionPointer, self).__init__(params, check_nan=False)
        
    
    def forward(self, embs: MatNetEncoderOutput, td: TensorDict):
        attn_mask = td["action_mask"].clone()
        attn_mask[..., 0] = True

        logits, _ = super().forward(embs, td, attn_mask=attn_mask)
        logit_mask = ~td["action_mask"]
        return logits, logit_mask
