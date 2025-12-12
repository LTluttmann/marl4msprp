import torch
from torch import nn
from tensordict import TensorDict
from torch.nn.modules import TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm

from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import TransformerParams
from marlprp.models.nn.init_embeddings import get_init_emb_layer

from .sparse import EfficientSparseCrossAttention
from .utils import BaseEncoder, MatNetEncoderOutput, MockTransformer
from .cross_attention import EfficientMixedScoreMultiHeadAttentionLayer, MixedScoreMultiHeadAttentionLayer



class MatNetEncoderLayer(nn.Module):

    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.params = params
        self.norm_first = params.norm_first

        if self.params.use_self_attn:
            self.shelf_mha = TransformerEncoderLayer(
                d_model=params.embed_dim,
                nhead=params.num_heads,
                dim_feedforward=params.feed_forward_hidden,
                dropout=params.dropout,
                activation=params.activation,
                norm_first=params.norm_first,
                batch_first=True,
            )
        else:
            self.shelf_mha = MockTransformer(params)

        if self.params.use_sku_attn:
            self.sku_mha = TransformerEncoderLayer(
                d_model=params.embed_dim,
                nhead=params.num_heads,
                dim_feedforward=params.feed_forward_hidden,
                dropout=params.dropout,
                activation=params.activation,
                norm_first=params.norm_first,
                batch_first=True,
            )
        else:
            # only use transformer mlp for sku embeddings
            self.sku_mha = MockTransformer(params)


        if params.param_sharing and params.ms_sparse_attn:
            self.cross_attn = EfficientSparseCrossAttention(params)
        elif params.param_sharing and not params.ms_sparse_attn:
            self.cross_attn = EfficientMixedScoreMultiHeadAttentionLayer(params)
        else:
            self.cross_attn = MixedScoreMultiHeadAttentionLayer(params)

        self.shelf_norm = LayerNorm(params.embed_dim)
        self.sku_norm = LayerNorm(params.embed_dim)

        if self.norm_first:
            self.shelf_out_norm = LayerNorm(params.embed_dim)
            self.sku_out_norm = LayerNorm(params.embed_dim)


    def forward(
        self, 
        shelf_emb, 
        sku_emb, 
        cost_mat=None, 
        cross_mask=None,
        shelf_mask=None,
        sku_mask=None,
    ):

        #### CROSS ATTENTION ####
        if self.norm_first:
            shelf_emb_out, sku_emb_out = self.cross_attn(
                self.shelf_norm(shelf_emb), 
                self.sku_norm(sku_emb), 
                cost_mat=cost_mat,
                attn_mask=cross_mask, 
            )
            
            #### SKIP CONN AND NORM ####
            shelf_emb_out = shelf_emb_out + shelf_emb
            sku_emb_out = sku_emb_out + sku_emb

        else:
            shelf_emb_out, sku_emb_out = self.cross_attn(
                shelf_emb, 
                sku_emb, 
                cost_mat=cost_mat,
                attn_mask=cross_mask, 
            )
            
            #### SKIP CONN AND NORM ####
            shelf_emb_out = self.shelf_norm(shelf_emb_out + shelf_emb)
            sku_emb_out = self.sku_norm(sku_emb_out + sku_emb)

        #### SELF ATTENTION ####
        shelf_emb_out = self.shelf_mha(shelf_emb_out, src_mask=shelf_mask)
        # (bs, num_ma, emb)
        sku_emb_out = self.sku_mha(sku_emb_out, src_mask=sku_mask)


        ###### FINAL NORMS ##########
        if self.norm_first:
            shelf_emb_out = self.shelf_out_norm(shelf_emb_out)
            sku_emb_out = self.sku_out_norm(sku_emb_out)

        return shelf_emb_out, sku_emb_out


class MatNetEncoder(BaseEncoder):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads
        self.init_embedding = get_init_emb_layer(params)
        self.mask_no_edge = params.mask_no_edge
        self.encoder = nn.ModuleList([])
        for _ in range(params.num_encoder_layers):
            self.encoder.append(MatNetEncoderLayer(params))


    def forward(self, state: MSPRPState) -> MatNetEncoderOutput:
        # (bs, jobs, ops, emb); (bs, ma, emb); (bs, jobs*ops, ma)
        node_emb, sku_emb, edge_feat = self.init_embedding(state)

        # optionally mask edges that do not exist
        if self.mask_no_edge:
            # (bs, num_job, num_ma)
            cross_mask = edge_feat.eq(0)
            cross_mask = cross_mask[:, None].expand(-1, self.num_heads, -1, -1).contiguous()
        else:
            cross_mask = None

        # mask skus that have no demand
        sku_mask = state.demand.eq(0).unsqueeze(1).repeat(1, state.num_skus, 1)
        sku_mask = sku_mask.diagonal_scatter(
            torch.full_like(state.demand, fill_value=False),
            dim1=1, dim2=2
        )
        sku_mask = sku_mask.repeat_interleave(self.num_heads, dim=0)

        # mask shelves that have no demanded items in stock
        shelf_mask = (state.demand.unsqueeze(1) * state.supply_w_depot).eq(0).all(-1)
        # current locations of agents will also be used during self attention
        shelf_mask = shelf_mask.scatter(1, state.current_location, False)
        shelf_mask[:, :state.num_depots] = False  # depots are never masked
        shelf_mask = shelf_mask.unsqueeze(1).repeat(1, state.num_shelves+state.num_depots, 1)
        shelf_mask = shelf_mask.diagonal_scatter(
            torch.full_like(state.coordinates[...,0], fill_value=False),
            dim1=1, dim2=2
        )
        shelf_mask = shelf_mask.repeat_interleave(self.num_heads, dim=0)

        # run through the layers 
        for layer in self.encoder:
            node_emb, sku_emb = layer(
                node_emb, 
                sku_emb, 
                cost_mat=edge_feat, 
                cross_mask=cross_mask,
                sku_mask=sku_mask,
                shelf_mask=shelf_mask,
            )
        
        return TensorDict(
            {"shelf": node_emb, "sku": sku_emb}, 
            batch_size=state.batch_size
        )


class ETEncoder(BaseEncoder):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads
        self.init_embedding = get_init_emb_layer(params)
        self.mask_no_edge = params.mask_no_edge

        self.agent_mha = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=params.norm_first,
            batch_first=True,
        )

        self.encoder = nn.ModuleList([])
        for _ in range(params.num_encoder_layers):
            self.encoder.append(MatNetEncoderLayer(params))


    def forward(self, state: MSPRPState) -> MatNetEncoderOutput:
        # (bs, jobs, ops, emb); (bs, ma, emb); (bs, jobs*ops, ma)
        node_emb, sku_emb, edge_feat, agent_emb = self.init_embedding(state)
        agent_emb = self.agent_mha(agent_emb)

        if self.mask_no_edge:
            # (bs, num_job, num_ma)
            cross_mask = edge_feat.eq(0)
        else:
            cross_mask = None

        # run through the layers 
        for layer in self.encoder:
            node_emb, sku_emb = layer(
                node_emb, 
                sku_emb, 
                cost_mat=edge_feat, 
                cross_mask=cross_mask,
            )
        
        return TensorDict(
            {"shelf": node_emb, "sku": sku_emb, "agent": agent_emb}, 
            batch_size=state.batch_size
        )
