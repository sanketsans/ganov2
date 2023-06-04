"""
This file contains a wrapper for Video-Swin-Transformer so it can be properly used as a temporal encoder for MTTR.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from einops import rearrange
# from mmcv import Config


class VideoSwinTransformerBackbone(nn.Module):
    """
    A wrapper which allows using Video-Swin Transformer as a temporal encoder for MTTR.
    Check out video-swin's original paper at: https://arxiv.org/abs/2106.13230 for more info about this architecture.
    Only the 'tiny' version of video swin was tested and is currently supported in our project.
    Additionally, we slightly modify video-swin to make it output per-frame embeddings as required by MTTR (check our
    paper's supplementary for more details), and completely discard of its 4th block.
    """
    def __init__(self, backbone_pretrained=True, train_backbone=True):
        super(VideoSwinTransformerBackbone, self).__init__()

        swin_backbone = torchvision.models.video.swin3d_b(torchvision.models.video.Swin3D_B_Weights.DEFAULT, progress=True)

        self.patch_embed = swin_backbone.patch_embed
        self.pos_drop = swin_backbone.pos_drop
        self.features = swin_backbone.features

        self.train_backbone = train_backbone
        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, x):
        h_still, h_fast = x
        vid_frames = rearrange(x, 't b c h w -> b c t h w')
        # vid_frames = samples.tensors
        vid_embeds = self.patch_embed(vid_frames)
        vid_embeds = self.pos_drop(vid_embeds)
        vid_embeds = self.features(vid_embeds)
        vid_embeds = rearrange(vid_embeds, 'b t h w c -> t b c h w')

        return vid_embeds
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
