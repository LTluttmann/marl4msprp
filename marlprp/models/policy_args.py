from dataclasses import dataclass, field
from marlprp.utils.config import PolicyParams


@dataclass(kw_only=True)
class TransformerParams(PolicyParams):
    policy: str = "transformer"
    num_heads: int = 8
    feed_forward_hidden: int = None
    qkv_dim: int = field(init=False)
    input_dropout: float = 0.0 # dropout after positional encoding
    num_decoder_ff_layers: int = 2
    activation: str = "gelu"
    norm_first: bool = False # True
    scale_supply_by_demand: bool = True
    bias: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.feed_forward_hidden = self.feed_forward_hidden or 2*self.embed_dim
        self.qkv_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, "self.kdim must be divisible by num_heads"

@dataclass(kw_only=True)
class MatNetParams(TransformerParams):
    policy: str = "ham"
    ms_hidden_dim: int = None
    mask_no_edge: bool = True
    decoder_attn_mask: bool = False
    ms_split_heads: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.ms_hidden_dim = self.ms_hidden_dim or self.qkv_dim




@dataclass(kw_only=True)
class MahamParams(MatNetParams):
    policy: str = "maham"
    use_communication: bool = True
    use_rezero: bool = False
