from audioop import bias
import torch 
import torch.nn as nn 
from einops import rearrange, repeat

from torch import Tensor
from typing import Optional

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Guided_Attn(nn.Module):
    def __init__(self, num_frames, dims,  num_heads=8, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0.3, attn_drop=0., 
            drop_path=0.3, act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.d_model = dims # self.cfg.MVIT.EMBED_DIM
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = nn.MultiheadAttention(self.d_model, num_heads=num_heads, dropout=0.3)
        # self.obj_queries = nn.Embedding(self.cfg.MODEL.NUM_QUERIES, self.d_model)  # pos embeddings for the object queries

        self.norm = norm_layer(self.d_model)
        mlp_hidden_dim = int(self.d_model  * mlp_ratio)
        self.mlp = Mlp(
            in_features= self.d_model , hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.bbox_to_feature = nn.Linear(4, self.d_model , bias=True) ## [Batch, T, O, dims]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        else:
            for p in m.parameters(): nn.init.normal_(p, std=0.02)

    def forward_boxes_dynamics(self, box_tensors, T):

        Tratio = box_tensors.shape[1] // T
        box_tensors = box_tensors[:,::Tratio] # [BS, T , O, 4]
        O = box_tensors.shape[-2]

        box_queries = box_tensors.flatten(0, 1) ## [BT, 0, 5]
        box_queries = self.bbox_to_feature(box_queries)  ## [BT, O, d]
        box_queries = rearrange(box_queries, 't_b o d -> o t_b d')

        return box_tensors, box_queries

    def forward(self, vid_embeds, orvit_boxes):
        
        b, d, t, h, w = vid_embeds.shape

        vid_embeds = rearrange(vid_embeds, 'b d t h w -> (h w) (b t) d')
        ## vid_pos_embed is: [T*B, H, W, d_model]

        _, box_queries = self.forward_boxes_dynamics(orvit_boxes, T=t) 

        patch_tokens = self.attn(query=vid_embeds,
                                   key=box_queries,
                                   value=box_queries, attn_mask=None,
                                   key_padding_mask=None)[0]   

        vid_embeds = vid_embeds + self.drop_path(patch_tokens)    ## [HW, BT, d]
        vid_embeds = vid_embeds + self.drop_path(self.mlp(self.norm(vid_embeds)))   ## [HW, BT ,d]

        vid_embeds = rearrange(vid_embeds, '(h w) (b t) d -> h w b t d', h=h, w=w, t=t, b=b)
        vid_embeds = rearrange(vid_embeds, 'h w b t d -> b d t h w', t=t, b=b, h=h, w=w)

        return vid_embeds   ###  
    
class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, 
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x