import torch
from einops import rearrange
from tensordict import TensorDict
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm

from marlprp.env.instance import MSPRPState
from marlprp.models.policy_args import MatNetParams
from marlprp.models.nn.init_embeddings import get_init_emb_layer

from .base import BaseEncoder, MatNetEncoderOutput
from .mixed_attention import EfficientMixedScoreMultiHeadAttention


class MatNetEncoderLayer(nn.Module):

    def __init__(self, params: MatNetParams) -> None:
        super().__init__()

        self.norm_first = params.norm_first

        self.shelf_mha = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=params.norm_first,
            batch_first=True,
        )

        self.sku_mha = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=self.norm_first,
            batch_first=True,
        )

        self.cross_attn = EfficientMixedScoreMultiHeadAttention(params)

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
                attn_mask=cross_mask, 
                cost_mat=cost_mat
            )
            
            #### SKIP CONN AND NORM ####
            shelf_emb_out = shelf_emb_out + shelf_emb
            sku_emb_out = sku_emb_out + sku_emb

        else:
            shelf_emb_out, sku_emb_out = self.cross_attn(
                shelf_emb, 
                sku_emb, 
                attn_mask=cross_mask, 
                cost_mat=cost_mat
            )
            
            #### SKIP CONN AND NORM ####
            shelf_emb_out = self.shelf_norm(shelf_emb_out + shelf_emb)
            sku_emb_out = self.sku_norm(sku_emb_out + sku_emb)

        #### SELF ATTENTION ####

        shelf_emb_out = self.shelf_mha(shelf_emb_out, src_mask=shelf_mask)

        # (bs, num_ma, emb)
        sku_emb_out = self.sku_mha(sku_emb_out, src_mask=sku_mask)


        ###### FINAL NORMS AND REARRANGE ##########
        if self.norm_first:
            shelf_emb_out = self.shelf_out_norm(shelf_emb_out)
            sku_emb_out = self.sku_out_norm(sku_emb_out)

        return shelf_emb_out, sku_emb_out


class MatNetEncoder(BaseEncoder):
    def __init__(self, params: MatNetParams) -> None:
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
        depot_emb, shelf_emb, sku_emb, edge_feat = self.init_embedding(state)

        if self.mask_no_edge:
            # (bs, num_job, num_ma)
            cross_mask = state.supply.gt(0).clone()
        else:
            cross_mask = None

        # run through the layers 
        for layer in self.encoder:
            shelf_emb, sku_emb = layer(
                shelf_emb, 
                sku_emb, 
                cost_mat=edge_feat, 
                cross_mask=cross_mask,
            )
        node_emb = torch.cat((depot_emb, shelf_emb), dim=1)
        return TensorDict(
            {"shelf": node_emb, "sku": sku_emb}, 
            batch_size=state.batch_size
        )
