import torch.nn as nn

from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import MahamParams
from marlprp.models.encoder.base import MatNetEncoderOutput


def get_kvl_emb(params: MahamParams, key: str = None) -> "ShelfKVL":
    emb_registry = {
        "maham": {
            "shelf": ShelfKVL,
            "sku": SkuKVL,
        }
    }
    EmbCls = emb_registry[params.policy]
    
    if key is not None:
        EmbCls = EmbCls[key]

    return EmbCls(params)


def get_dynamic_emb(params: MahamParams, key: str = None):
    # if we encode stepwise, dynamic embeddings are not needed
    if params.stepwise_encoding:
        return StaticEmbedding(params)
    
    emb_registry = {
        "maham": {
            "shelf": ...,
            "sku": ...
        }
    }

    EmbCls = emb_registry[params.policy]
    
    if key is not None:
        EmbCls = EmbCls[key]

    return EmbCls(params)



class ShelfKVL(nn.Module):

    def __init__(self,params: MahamParams):
        super().__init__()
        self.dynamic_embedding = get_dynamic_emb(params)
        self.Wkvl = nn.Linear(params.embed_dim, 3 * params.embed_dim, bias=False)
        self.cache = None

    def compute_cache(self, embs: MatNetEncoderOutput) -> None:
        # shape: 3 * (bs, n, emb_dim)
        self.cache = self.Wkvl(embs["shelf"]).chunk(3, dim=-1)

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState, cache = None):

        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(emb, state)

        if cache is not None:
            k, v, l = cache
            
        else:
            k, v, l = self.Wkvl(emb["shelf"]).chunk(3, dim=-1)

        k_dyn = k + glimpse_k_dyn
        v_dyn = v + glimpse_v_dyn
        l_dyn = l + logit_k_dyn

        return k_dyn, v_dyn, l_dyn
    

class SkuKVL(nn.Module):

    def __init__(self,params: MahamParams):
        super().__init__()
        self.dynamic_embedding = get_dynamic_emb(params)
        self.Wkvl = nn.Linear(params.embed_dim, 3 * params.embed_dim, bias=False)
        self.cache = None
        
    def compute_cache(self, embs: MatNetEncoderOutput) -> None:
        # shape: 3 * (bs, n, emb_dim)
        self.cache = self.Wkvl(embs["sku"]).chunk(3, dim=-1)

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState, cache = None):

        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(emb, state)

        if cache is not None:
            k, v, l = cache
            
        else:
            k, v, l = self.Wkvl(emb["sku"]).chunk(3, dim=-1)

        k_dyn = k + glimpse_k_dyn
        v_dyn = v + glimpse_v_dyn
        l_dyn = l + logit_k_dyn

        return k_dyn, v_dyn, l_dyn


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState):
        return 0, 0, 0


class DynEmb(nn.Module):
    def __init__(self, params: MahamParams) -> None:
        super().__init__()
        self.project_node_step = nn.Linear(1, 3 * params.embed_dim, bias=False)

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState):
        raise NotImplementedError