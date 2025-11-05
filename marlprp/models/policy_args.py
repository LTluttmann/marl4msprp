from dataclasses import dataclass, field
from marlprp.utils.config import PolicyParams


@dataclass(kw_only=True)
class TransformerParams(PolicyParams):
    policy: str = "transformer"
    num_heads: int = 8
    feed_forward_hidden: int = None
    qkv_dim: int = field(init=False)
    input_dropout: float = 0.0 # dropout after positional encoding
    activation: str = "gelu"
    norm_first: bool = False # True
    scale_supply_by_demand: bool = True
    bias: bool = True
    ms_hidden_dim: int = None
    mask_no_edge: bool = True
    decoder_attn_mask: bool = False
    use_rezero: bool = False
    param_sharing: bool = True
    cost_mat_dims: int = 1
    chunk_ms_scores_batch: int = 0
    ms_scores_softmax_temp: float = 1.0
    ms_scores_tanh_clip: float = 0.0
    ms_sparse_attn: bool = False
    use_sku_attn: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.feed_forward_hidden = self.feed_forward_hidden or 2*self.embed_dim
        self.qkv_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, "self.kdim must be divisible by num_heads"
        self.ms_hidden_dim = self.ms_hidden_dim or self.qkv_dim


@dataclass(kw_only=True)
class HAMParams(TransformerParams):
    policy: str = "ham"
    eval_multistep: bool = False


@dataclass(kw_only=True)
class MahamParams(TransformerParams):
    policy: str = "maham"
    eval_multistep: bool = True
    eval_per_agent: bool = True
    use_communication: bool = True
    use_ranking_pe: bool = False
    agent_ranking: str = "learned" # random index

@dataclass(kw_only=True)
class ETParams(TransformerParams):
    policy: str = "et"
    eval_multistep: bool = False
    def __post_init__(self):
        super().__post_init__()
        assert self.env.name == "ar", "EquityTransformer only works with pure autoregressive env"


@dataclass(kw_only=True)
class Ptr2DParams(TransformerParams):
    policy: str = "2dptr"
    eval_multistep: bool = False
    use_communication: bool = True
    use_ranking_pe: bool = False

    # def __post_init__(self):
    #     super().__post_init__()
    #     assert self.env.name == "ar", "2D-Ptr only works with pure autoregressive env"

@dataclass(kw_only=True)
class ParcoParams(TransformerParams):
    policy: str = "parco"
    eval_multistep: bool = True
    eval_per_agent: bool = True
    use_communication: bool = True
    use_ranking_pe: bool = False
    agent_ranking: str = "logp" # random index