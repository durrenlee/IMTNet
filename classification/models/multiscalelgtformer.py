"""
modified from https://github.com/JIAOJIAYUASD/dilateformer/blob/main/models/dilateformer.py
"""

import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .DCNv4_op.DCNv4.modules import DCNv4

from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader


class DWConvAttention(nn.Module):
    def __init__(self, dim=768):
        super(DWConvAttention, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x

class MultiDilateLocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])

        self.dwconv_attn = DWConvAttention(dim=dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        dwconv_attn_out = self.dwconv_attn(x.permute(0, 3, 2, 1))

        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        # x shape:torch.Size([1, 72, 56, 56])

        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # qkv shape:torch.Size([3, 3, 1, 24, 56, 56])

        # num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # x shape:torch.Size([3, 1, 56, 56, 24])

        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            # qkv[i][0]:torch.Size([1, 24, 56, 56])
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
            # x[i]:torch.Size([1, 56, 56, 24])
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)

        # fuse window and dw conv
        x = x + dwconv_attn_out.permute(0, 3, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GroupedDilatedAttention_GlobalQKV(nn.Module):
    "Implementation of grouped dilate attention"

    def __init__(self, dim, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3], channel_groups=4):
        super().__init__()
        assert dim % self.channel_groups == 0, f"dim {dim} must be divisible by channel groups {self.channel_group}."
        self.dim = dim
        self.channel_groups = channel_groups
        group_dim = self.dim // self.channel_groups
        self.scale = qk_scale or group_dim ** -0.5
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num_dilation = len(dilation)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)

        self.dilate_attention = nn.ModuleList([
            nn.ModuleList(
                [DilateAttention(group_dim // self.num_dilation, qk_scale, attn_drop, kernel_size, dilation[i])
                 for i in range(self.num_dilation)])
            for _ in range(self.channel_groups)
        ])

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        G = self.channel_groups
        group_dim = C // G

        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        qkv = self.qkv(x)
        qkv = qkv.reshape(G, self.num_dilation, 3, B, group_dim // 3, H, W)

        x = x.reshape(G, B, self.num_dilation, group_dim // self.num_dilation, H, W).permute(0, 2, 1, 4, 5, 3)

        for g in range(G):  # Process each group independently

            for i in range(self.num_dilation):
                x[g][i] = self.dilate_attention[g][i](qkv[g][i][0], qkv[g][i][1],qkv[g][i][2])

        x = x.reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GroupedDilatedAttention_LocalGKV(nn.Module):
    "Implementation of grouped dilate attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3], channel_groups=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)

        self.channel_groups = channel_groups
        group_dim = self.dim // self.channel_groups
        assert dim % self.channel_groups == 0, f"dim {dim} must be divisible by channel groups {self.channel_group}."

        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"

        # group dim based on groups
        self.qkv = nn.Conv2d(group_dim, group_dim * 3, 1, bias=qkv_bias)

        self.dilate_attention = nn.ModuleList([
            nn.ModuleList(
                [DilateAttention(group_dim // self.num_dilation, qk_scale, attn_drop, kernel_size, dilation[i])
                 for i in range(self.num_dilation)])
            for _ in range(self.channel_groups)
        ])

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        G = self.channel_groups
        group_dim = C // G

        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        # group qkv convolution
        x_groups = x.reshape(G, B, group_dim, H, W)

        group_qkv_list = []
        for g in range(G):
            group_qkv = self.qkv(x_groups[g])

            group_qkv = group_qkv.reshape(self.num_dilation, 3, B, group_dim // 3, H, W)  # num_dilate, 3, 1, 12, 56, 56
            group_qkv_list.append(group_qkv)

        x = x.reshape(G, B, self.num_dilation, group_dim // self.num_dilation, H, W).permute(0, 2, 1, 4, 5, 3)

        for g in range(G):  # Process each group independently

            for i in range(self.num_dilation):
                x[g][i] = self.dilate_attention[g][i](group_qkv_list[g][i][0], group_qkv_list[g][i][1],
                                                      group_qkv_list[g][i][2])

        x = x.reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GroupedDilatedAttention(nn.Module):
    "Implementation of grouped dilate attention"

    def __init__(self, dim, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3], channel_groups=4):
        super().__init__()

        assert dim % self.channel_groups == 0, f"dim {dim} must be divisible by channel groups {self.channel_group}."

        self.dim = dim
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.channel_groups = channel_groups
        group_dim = self.dim // self.channel_groups
        self.scale = qk_scale or group_dim ** -0.5

        self.num_dilation = len(dilation)

        # group dim based on groups
        self.qkv = nn.Conv2d(group_dim, group_dim * 3, 1, bias=qkv_bias)

        self.dilate_attention = nn.ModuleList([
            nn.ModuleList(
                [DilateAttention(group_dim // self.num_dilation, qk_scale, attn_drop, kernel_size, dilation[i])
                 for i in range(self.num_dilation)])
            for _ in range(self.channel_groups)
        ])

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        G = self.channel_groups
        group_dim = C // G

        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        # group qkv convolution
        x_groups = x.reshape(G, B, group_dim, H, W)

        group_qkv_list = []
        for g in range(G):
            group_qkv = self.qkv(x_groups[g])
            group_qkv = group_qkv.reshape(self.num_dilation, 3, B, group_dim // 3, H, W)  # num_dilate, 3, 1, 12, 56, 56
            group_qkv_list.append(group_qkv)

        x = x.reshape(G, B, self.num_dilation, group_dim // self.num_dilation, H, W).permute(0, 2, 1, 4, 5, 3)

        for g in range(G):  # Process each group independently

            for i in range(self.num_dilation):
                x[g][i] = self.dilate_attention[g][i](group_qkv_list[g][i][0], group_qkv_list[g][i][1],
                                                      group_qkv_list[g][i][2])

        x = x.reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x.clone())
        qkv = qkv.reshape(B, H * W, 3, self.num_heads,
                          C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalBlock(nn.Module):
    "Implementation of Dilate-attention block"

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False, channel_groups=4, local_attn='GDA'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)

        # local attention type
        if local_attn == 'dilate':
            self.attn = MultiDilateLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
        elif local_attn == 'GDA':
            self.attn = GroupedDilatedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation,
                                                channel_groups=channel_groups)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        # B, C, H, W
        return x


class GlobalBlock(nn.Module):
    """
    Implementation of Transformer
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_block=False):
        super().__init__()
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class DCNBlock(nn.Module):
    """
    Implementation of Transformer
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, cpe_per_block=False,

                 group=12,
                 offset_scale=1.0,
                 dw_kernel_size=None,  # for InternImage-H/G
                 center_feature_scale=False,  # for InternImage-H/G
                 remove_center=False,  # for InternImage-H/G
                 # norm_layer_dcn="LN",
                 # act_layer_dcn='GELU',
                 output_bias=True,
                 without_pointwise=False,
                 ):
        super().__init__()
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.dcn = DCNv4(
            channels=dim,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=group,
            offset_scale=offset_scale,
            # act_layer=act_layer_dcn,
            # norm_layer=norm_layer_dcn,
            dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
            center_feature_scale=center_feature_scale,  # for InternImage-H/G
            remove_center=remove_center,
            output_bias=output_bias,
            without_pointwise=without_pointwise,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        # torch.Size([2, 288, 50, 84])
        # torch.Size([2, 576, 25, 42])
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
            # torch.Size([2, 288, 50, 84])
        x = x.permute(0, 2, 3, 1)
        # torch.Size([2, 50, 84, 288])

        B, H, W, D = x.shape  # 2, 50, 84, 288
        dcn_x = x
        dcn_x = dcn_x.reshape(B, H * W, D)
        # torch.Size([2, 4200, 288])
        # torch.Size([2, 1050, 576])
        dcn_out = self.drop_path(self.dcn(self.norm1(dcn_x), (H, W)))  # in case H is not equal to W
        dcn_out = dcn_out.reshape(B, H, W, D)
        x = x + dcn_out
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    """

    def __init__(self, img_size=224, in_chans=3, hidden_dim=16,
                 patch_size=4, embed_dim=96, patch_way=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        assert patch_way in ['overlaping', 'nonoverlaping', 'pointconv'], \
            "the patch embedding way isn't exist!"
        if patch_way == "nonoverlaping":
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 224x224
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 56x56
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=1, stride=1,
                          padding=0, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 56x56
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=1, stride=1,
                          padding=0, bias=False),  # 56x56
            )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # B, C, H, W
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(self, in_channels, out_channels, merging_way, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge
        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)

    def forward(self, x):
        # x: B, C, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class LocalStage(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """

    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None,
                 channel_groups=4, local_attn='GDA'):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            LocalBlock(dim=dim, num_heads=num_heads,
                       kernel_size=kernel_size, dilation=dilation,
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block,
                       channel_groups=channel_groups, local_attn=local_attn)

            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class GlobalStage(nn.Module):
    """ A basic Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):
        super().__init__()

        # build blocks
        self.blocks = nn.ModuleList([
            GlobalBlock(dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class DCNStage(nn.Module):
    """ A basic Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None,

                 group=12,
                 offset_scale=1.0,
                 dw_kernel_size=None,
                 center_feature_scale=False,
                 remove_center=False,

                 output_bias=True,
                 without_pointwise=False,
                 ):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            DCNBlock(dim=dim, num_heads=num_heads,
                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                     qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block,

                     group=group,
                     offset_scale=offset_scale,
                     dw_kernel_size=dw_kernel_size,
                     center_feature_scale=center_feature_scale,
                     remove_center=remove_center,

                     output_bias=output_bias,
                     without_pointwise=without_pointwise,
                     )
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x):
        # torch.Size([2, 288, 50, 84])
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)

        return x


class MultiScaleLgtFormer(nn.Module):  # groups=[3, 6, 12, 24]
    def __init__(self, img_size=64, patch_size=4, in_chans=3, num_classes=200, embed_dim=96,
                 depths=[2, 2, 18, 4], groups=[4, 8, 18, 36], num_heads=[3, 6, 12, 24], channel_groups=[4, 8, 16, 32],
                 kernel_size=3,
                 dilation=[1, 2, 3],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer='layer_norm',
                 merging_way='conv3_2',
                 patch_way='overlaping',
                 # dilate_attention=[True, True, False, False],
                 dilate_attention=['dilate', 'global', 'global', 'dcn'],
                 downsamples=[True, True, True, False],
                 cpe_per_satge=False, cpe_per_block=True,

                 offset_scale=1.0,
                 dw_kernel_size=3,
                 center_feature_scale=False,
                 remove_center=False,

                 output_bias=True,
                 without_pointwise=False,

                 init_cfg=None
                 ):
        super().__init__()
        print(f'***************************num_classes:{num_classes}')
        print(f'***************************img_size:{img_size}')
        print(f'***************************dilate_attention:{dilate_attention}')
        print(f'***************************downsamples:{downsamples}')

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        if downsamples[3]:
            print(f'downsamples[3]:{downsamples[3]}')
            self.num_features = int(embed_dim * 2 ** self.num_layers)
        else:
            print(f'downsamples[3]:{downsamples[3]}')
            self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.mlp_ratio = mlp_ratio
        norm_layer = partial(nn.LayerNorm, eps=1e-6) if norm_layer == 'layer_norm' else norm_layer

        # patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if dilate_attention[i_layer] == 'dilate' or dilate_attention[i_layer] == 'GDA':
                print(f'dilate_attention[i_layer]:{dilate_attention[i_layer]}')
                stage = LocalStage(dim=int(embed_dim * 2 ** i_layer),
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   kernel_size=kernel_size,
                                   dilation=dilation,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=downsamples[i_layer],
                                   cpe_per_block=cpe_per_block,
                                   cpe_per_satge=cpe_per_satge,
                                   merging_way=merging_way,
                                   channel_groups=channel_groups[i_layer],
                                   local_attn=dilate_attention[i_layer]
                                   )
            elif dilate_attention[i_layer] == 'global':
                stage = GlobalStage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
            else:
                stage = DCNStage(dim=int(embed_dim * 2 ** i_layer),
                                 depth=depths[i_layer],
                                 num_heads=num_heads[i_layer],
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                 norm_layer=norm_layer,
                                 downsample=downsamples[i_layer],
                                 cpe_per_block=cpe_per_block,
                                 cpe_per_satge=cpe_per_satge,
                                 merging_way=merging_way,

                                 group=groups[i_layer],
                                 offset_scale=offset_scale,
                                 dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                                 center_feature_scale=center_feature_scale,  # for InternImage-H/G
                                 remove_center=remove_center,  # for InternImage-H/G

                                 output_bias=output_bias,
                                 without_pointwise=without_pointwise,
                                 )
            self.stages.append(stage)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.init_cfg = init_cfg
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'model' in ckpt:
                _state_dict = ckpt['model']
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']

            # remove head weights
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in _state_dict:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del _state_dict[k]
            self.load_state_dict(_state_dict, False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def multiscalelgtformer_tiny(pretrained=False, pretrained_cfg=None, **kwargs):
    init_cfg = None

    # please replace checkpoint value with your actual checkpoint.pth
    if pretrained:
        init_cfg = pretrained_cfg
    """
    ========================================================
    parameters: 22.938M | FLOPs: 3.496G
    ------------------------------------
        img_size=224
        depths=[2, 4, 6, 2],
        embed_dim=72,
        groups=[4, 8, 18, 36],
        dilate_attention=['dilate', 'dilate', 'dcn', 'dcn'],
        downsamples=[True, True, True, True],
        num_heads=[3, 6, 12, 24],
    """
    model = MultiScaleLgtFormer(
        img_size=224,
        depths=[2, 4, 6, 2],
        embed_dim=72,
        groups=[4, 8, 18, 36],
        dilate_attention=['dilate', 'dilate', 'dcn', 'dcn'],
        downsamples=[True, True, True, True],
        num_heads=[3, 6, 12, 24],
        channel_groups=[8, 16, 8, 16],
        init_cfg=init_cfg,
        **kwargs)
    model.default_cfg = _cfg()
    model.init_weights()

    return model


@register_model
def multiscalelgtformer_small(pretrained=False, pretrained_cfg=None, **kwargs):
    init_cfg = None

    # please replace checkpoint value with your actual checkpoint.pth
    if pretrained:
        init_cfg = pretrained_cfg
    """
    parameters: 40.62M | FLOPs:  5.891G
        img_size=224,
        depths=[2, 4, 6, 2],
        embed_dim=96,
        groups=[4, 8, 24, 48],
        dilate_attention=['dilate', 'dilate', 'dcn', 'dcn'],
        downsamples=[True, True, True, True],
        num_heads=[3, 6, 12, 24],
    """
    model = MultiScaleLgtFormer(
        img_size=224,
        depths=[2, 4, 6, 2],
        embed_dim=96,
        groups=[4, 8, 24, 48],
        dilate_attention=['dilate', 'dilate', 'dcn', 'dcn'],
        num_heads=[3, 6, 12, 24],
        downsamples=[True, True, True, True],
        channel_groups=[8, 16, 8, 16],
        init_cfg=init_cfg,
        **kwargs)
    model.default_cfg = _cfg()
    model.init_weights()

    return model


@register_model
def multiscalelgtformer_base(pretrained=False, pretrained_cfg=None, **kwargs):
    init_cfg = None

    # please replace checkpoint value with your actual checkpoint.pth
    if pretrained:
        init_cfg = pretrained_cfg

    model = MultiScaleLgtFormer(
        img_size=224,
        depths=[3, 5, 8, 3],
        embed_dim=96,
        groups=[4, 12, 24, 48],  # for dim:96
        dilate_attention=['dilate', 'dilate', 'dcn', 'dcn'],
        downsamples=[True, True, True, True],
        num_heads=[3, 6, 12, 24],
        init_cfg=init_cfg,
        **kwargs)
    model.default_cfg = _cfg()
    model.init_weights()
    return model


@register_model
def multiscalelgtformer_large(pretrained=False, pretrained_cfg=None, **kwargs):
    init_cfg = None

    # please replace checkpoint value with your actual checkpoint.pth
    if pretrained:
        init_cfg = pretrained_cfg

    model = MultiScaleLgtFormer(
        img_size=224,
        depths=[4, 8, 10, 4],
        embed_dim=96,
        groups=[4, 12, 24, 48],  # for dim:96
        dilate_attention=['dilate', 'dilate', 'dcn', 'dcn'],
        downsamples=[True, True, True, True],
        num_heads=[3, 6, 12, 24],
        init_cfg=init_cfg,
        **kwargs)
    model.default_cfg = _cfg()
    model.init_weights()
    return model


if __name__ == "__main__":
    x = torch.rand([2, 3, 224, 224])
    m = multiscalelgtformer_tiny(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None)
    y = m(x)
    print(y.shape)
