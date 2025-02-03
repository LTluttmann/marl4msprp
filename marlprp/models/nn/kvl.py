import torch
import torch.nn as nn

from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import MahamParams
from marlprp.models.encoder.base import MatNetEncoderOutput


def get_kvl_emb(params: MahamParams, key: str = None) -> "ShelfKVL":
    kvl_layer = {
        "shelf": ShelfKVL,
        "sku": SkuKVL,
    }
    emb_registry = dict.fromkeys(["ham", "maham", "et", "2dptr", "parco"], kvl_layer)
    EmbCls = emb_registry[params.policy]
    
    if key is not None:
        EmbCls = EmbCls[key]

    return EmbCls(params)


def get_dynamic_emb(params: MahamParams, key: str = None):
    # if we encode stepwise, dynamic embeddings are not needed
    if params.stepwise_encoding:
        return StaticEmbedding(params)
    
    dyn_emb = {
        "shelf": ShelfDynEmb,
        "sku": SkuDynEmb
    }
    emb_registry = dict.fromkeys(["ham", "maham", "et", "2dptr", "parco"], dyn_emb)

    EmbCls = emb_registry[params.policy]
    
    if key is not None:
        EmbCls = EmbCls[key]

    return EmbCls(params)



class ShelfKVL(nn.Module):

    def __init__(self,params: MahamParams):
        super().__init__()
        self.dynamic_embedding = get_dynamic_emb(params, key="shelf")
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
        self.dynamic_embedding = get_dynamic_emb(params, key="sku")
        self.Wkvl = nn.Linear(params.embed_dim, 3 * params.embed_dim, bias=False)
        self.dummy = nn.Parameter(torch.zeros(1, 1, params.embed_dim), requires_grad=False)
        self.cache = None
        
    def compute_cache(self, embs: MatNetEncoderOutput) -> None:
        # shape: 3 * (bs, n, emb_dim)
        self.cache = self.Wkvl(embs["sku"]).chunk(3, dim=-1)

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState, cache = None):
        bs = emb.batch_size
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(emb, state)

        if cache is not None:
            k, v, l = cache
            
        else:
            k, v, l = self.Wkvl(emb["sku"]).chunk(3, dim=-1)

        k_dyn = k + glimpse_k_dyn
        v_dyn = v + glimpse_v_dyn
        l_dyn = l + logit_k_dyn

        k_dyn_w_dummy_sku = torch.cat((self.dummy.expand(*bs, 1, -1), k_dyn), dim=1)
        v_dyn_w_dummy_sku = torch.cat((self.dummy.expand(*bs, 1, -1), v_dyn), dim=1)
        l_dyn_w_dummy_sku = torch.cat((self.dummy.expand(*bs, 1, -1), l_dyn), dim=1)
        return k_dyn_w_dummy_sku, v_dyn_w_dummy_sku, l_dyn_w_dummy_sku


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState):
        return 0, 0, 0


class ShelfDynEmb(nn.Module):
    def __init__(self, params: MahamParams) -> None:
        super().__init__()
        self.project_edge_step = nn.Linear(params.embed_dim,  3 * params.embed_dim, bias=False)

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState):
        bs, n_nodes, emb_dim = emb["shelf"].shape
        # (bs, nodes, sku)
        supply_scaled = state.supply_w_depot / state.capacity
        edge_feature = torch.einsum('ijk,ikm->ijm', supply_scaled, emb["sku"]).view(bs, n_nodes, emb_dim)
        edge_update = self.project_edge_step(edge_feature)
        return edge_update.chunk(3, dim=-1)
    

class SkuDynEmb(nn.Module):
    def __init__(self, params: MahamParams) -> None:
        super().__init__()
        self.project_sku_step = nn.Linear(1, 3 * params.embed_dim, bias=False)
        self.project_edge_step = nn.Linear(params.embed_dim,  3 * params.embed_dim, bias=False)

    def forward(self, emb: MatNetEncoderOutput, state: MSPRPState):
        bs, n_sku, emb_dim = emb["sku"].shape
        sku_update = self.project_sku_step(state.demand.unsqueeze(-1) / state.capacity)
        # (bs, sku, nodes)
        supply_scaled = state.supply_w_depot.transpose(-2,-1) / state.capacity
        edge_feature = torch.einsum('ijk,ikm->ijm', supply_scaled, emb["shelf"]).view(bs, n_sku, emb_dim)
        edge_update = self.project_edge_step(edge_feature)
        return (edge_update + sku_update).chunk(3, dim=-1)
