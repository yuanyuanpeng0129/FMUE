# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from easydict import EasyDict

import timm.models.vision_transformer
from models.custom_modules  import Block


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        act_layer = nn.GELU
        tuning_config = EasyDict(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=64,
        d_model=kwargs['embed_dim'],
        vpt_on=False,
        vpt_num=1,
        )
        drop_path_rate = 0.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, kwargs['depth'])]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'], qkv_bias=kwargs['qkv_bias'], drop=0.,
                attn_drop=0., drop_path=dpr[i], norm_layer=kwargs['norm_layer'], act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(kwargs['depth'])])

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    def get_last_selfattention(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            print(len(self.blocks))
            print(i)
            if i < len(self.blocks) - 1:
                x = blk(x,return_attention=False)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

