import torch
import torch.nn as nn
from tensordict import TensorDict
from marlprp.utils.config import ModelParams


def get_dynamic_emb(params: ModelParams, key: str = None):
    # if we encode stepwise, dynamic embeddings are not needed
    if params.stepwise_encoding:
        return StaticEmbedding(params)
    
    emb_registry = {
        "marlprp": DynEmb
    }

    EmbCls = emb_registry[params.policy]
    
    if key is not None:
        EmbCls = EmbCls[key]

    return EmbCls(params)


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td, emb: torch.Tensor):
        return 0, 0, 0


class DynEmb(nn.Module):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.project_node_step = nn.Linear(1, 3 * params.policy.embed_dim, bias=False)
        self.scaling_factor = params.policy.env.scaling_factor

    def forward(self, td, emb: torch.Tensor):
        raise NotImplementedError