import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from collections import OrderedDict

from ..DCNv4_op.DCNv4.modules import DCNv4

from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader


class DWConvAttention(nn.Module):
    def __init__(self, dim=768):
        super(DWConvAttention, self).__init__()
        # self.dwconv1 = nn.Conv2d(dim, 2*dim, 3, 1, 1, bias=True, groups=dim)
        # self.dwconv2 = nn.Conv2d(2*dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        # x = self.dwconv1(x)
        # x = self.dwconv2(x)
        x = self.dwconv(x)
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


class ConvTransform(nn.Module):
    def __init__(self, input_channels=576, output_tokens=197, output_dim=512):
        super(ConvTransform, self).__init__()
        self.output_dim = output_dim
        # 使用全局平均池化来减少空间维度
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 使用1x1卷积来调整通道数
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_tokens * output_dim, kernel_size=1)

    def forward(self, x):
        # 应用全局平均池化
        x = self.gap(x)
        # 通过1x1卷积调整通道数
        x = self.conv(x)
        # 调整形状为 [batch_size, 197, 512]
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.output_dim)
        return x


class LowRankTransform(nn.Module):
    def __init__(self, input_channels, output_tokens, output_dim, rank):
        super(LowRankTransform, self).__init__()
        # Adaptive average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Parameters for low-rank SVD approximation
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.output_tokens = output_tokens
        self.rank = rank
        # Weight matrix for transformation
        device = torch.cuda.current_device()
        self.W = torch.randn(input_channels, output_tokens * output_dim).to(device)

        # Perform SVD on the initial weight matrix
        U, S, V = torch.svd(self.W)
        # Reduce to rank 'r'
        self.U_reduced = U[:, :rank]
        self.S_reduced = S[:rank]
        self.V_reduced = V[:, :rank]

    def forward(self, x):
        # Pool the input to [batch_size, input_channels, 1, 1]
        x = self.gap(x)  # Resulting shape: [batch_size, input_channels, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, input_channels]

        # Reconstruct the approximated weight matrix W_approx
        W_approx = self.U_reduced @ torch.diag(self.S_reduced) @ self.V_reduced.T

        # Transform the input using W_approx
        # Transforming x from [batch_size, input_channels] to [batch_size, output_tokens * output_dim]
        output = x @ W_approx

        # Reshape to [batch_size, output_tokens, output_dim]
        output = output.view(x.size(0), self.output_tokens, self.output_dim)
        return output


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

        # self.dwconv_attn = DWConvAttention(dim=dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x_dwconv_attn_out = x.permute(0, 3, 2, 1)
        # clone is used for variable inplace issue while computing gradient
        # dwconv_attn_out = self.dwconv_attn(x_dwconv_attn_out.clone())
        # dwconv_attn_out = self.dwconv_attn(x.permute(0, 3, 2, 1))

        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        # clone is used for variable inplace issue while computing gradient
        qkv_out = self.qkv(x.clone())
        qkv_out = qkv_out.reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv_out[i][0], qkv_out[i][1], qkv_out[i][2])  # B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)

        # fuse window and dw conv
        # x = x + dwconv_attn_out.permute(0, 3, 2, 1)

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
                 cpe_per_block=False):
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
        self.attn = MultiDilateLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

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

    def __init__(self, img_size=(800, 1333), in_chans=3, hidden_dim=16,
                 patch_size=4, embed_dim=96, patch_way=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution_height = img_size[0] // patch_size[0]  # 200
        patches_resolution_width = int(int(img_size[1] + patch_size[1] - 1) / patch_size[1])  # 334

        patches_resolution = [patches_resolution_width, patches_resolution_height]
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 66,600 patches
        self.img_size = img_size
        assert patch_way in ['overlaping', 'nonoverlaping', 'pointconv'], \
            "the patch embedding way isn't exist!"
        if patch_way == "nonoverlaping":
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 800x1333
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=3, stride=2,
                          padding=1, bias=False),  # 400x667
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=1,
                          padding=1, bias=False),  # 400x667
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),  # 200x334
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=1, stride=1,
                          padding=0, bias=False),
            )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # print('input x H:')
        # print(H)
        # print('input x W:')
        # print(W)
        # print('==============')
        # print('model H:')
        # print(self.img_size[0])
        # print('model W:')
        # print(self.img_size[1])

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
                 downsample=True, merging_way=None):
        super().__init__()

        # build blocks
        self.blocks = nn.ModuleList([
            LocalBlock(dim=dim, num_heads=num_heads,
                       kernel_size=kernel_size, dilation=dilation,
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
            # torch.Size([2, 72, 200, 334])
            # torch.Size([2, 144, 100, 167])
        x = self.downsample(x)
        # torch.Size([2, 144, 100, 167])
        # torch.Size([2, 288, 50, 84])
        return x


class Globalstage(nn.Module):
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
        # print('group in GlobalStage:')
        # print(group)
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


from mmdet.registry import MODELS


@MODELS.register_module()
class MultiScaleLgtFormer(nn.Module):  # groups=[3, 6, 12, 24]
    def __init__(self,
                 img_size=(800, 1333),
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[4, 8, 21, 4],
                 groups=[4, 8, 18, 36],
                 num_heads=[3, 6, 12, 24],
                 kernel_size=3,
                 dilation=[1, 2, 3],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.1,
                 norm_layer='layer_norm',
                 merging_way='conv3_2',
                 patch_way='overlaping',
                 dilate_attention=['dilate', 'global', 'global', 'dcn'],
                 downsamples=[True, True, True, False],
                 cpe_per_satge=False,
                 cpe_per_block=True,

                 offset_scale=1.0,
                 dw_kernel_size=3,
                 center_feature_scale=False,
                 remove_center=False,

                 output_bias=True,
                 without_pointwise=False,

                 out_indices=(0, 1, 2, 3),
                 task='DET',
                 init_cfg=None
                 ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        norm_layer = partial(nn.LayerNorm, eps=1e-6) if norm_layer == 'layer_norm' else norm_layer

        # patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if dilate_attention[i_layer] == 'dilate':
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
                                   merging_way=merging_way
                                   )
            elif dilate_attention[i_layer] == 'global':
                stage = Globalstage(dim=int(embed_dim * 2 ** i_layer),
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
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.task = task
        if task == 'STR':

            # for downsamples=[True, True, True, False]
            # final_dim = int(embed_dim * 2 ** (self.num_layers - 1))
            # for downsamples=[True, True, True, True]
            final_dim = int(embed_dim * 2 ** self.num_layers)

            self.conv_transform = ConvTransform(input_channels=final_dim, output_tokens=197, output_dim=512)
            # SVD to reduce train parameters
            # self.lowrank_transform = LowRankTransform(input_channels=final_dim, output_tokens=197, output_dim=512, rank=16)

        self.init_cfg = init_cfg
        # if self.init_cfg:
        #     self.init_weight_pth()
        # else:
        self.apply(self._init_weights)

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
        if self.init_cfg:
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
            # for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            #     if k in _state_dict:
            #         print(f"Removing key {k} from pretrained checkpoint")
            #         del _state_dict[k]

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # for param_tensor in _state_dict:
            #     print(param_tensor, "\t", _state_dict[param_tensor].size())
            # interpolate position embedding? not need because depth wise Convolution is used as CPE

            # load state_dict
            meg = self.load_state_dict(state_dict, False)
            logger.info(meg)
        else:
            logger.info("train without weights.")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        # used for object detection task
        if 'DET' in self.task:
            out_indices = self.out_indices
            outs = []
            i = 0
            for stage in self.stages:
                x = stage(x)
                if i in out_indices:
                    outs.append(x)
                i = i + 1
            return tuple(outs)

        # used for input of text recognition task or classification
        if self.task == 'CLS' or self.task == 'STR':
            for stage in self.stages:
                x = stage(x)
                # print('x shape after each stage:')
                # print(x.shape)
            return x

    def forward(self, x):
        # x = torch.stack(x, dim=0)
        # x = x.float()

        # used for classification task
        if self.task == 'CLS':
            x = self.forward_features(x)
            return x

        # used for object detection task on mmdetection framework
        if self.task == 'DET':
            tuple_out = self.forward_features(x)
            return tuple_out

        # used for input of text recognition task
        if self.task == 'STR':
            x = self.forward_features(x)

            x = self.conv_transform(x)
            # print('x = self.conv_transform(x)')
            # print(x.shape)
            # x = self.lowrank_transform(x)
            return x

        # used for input of det and str tasks
        # if self.task == 'STR-DET':
        #     tuple_out = self.forward_features(x)
        #     permuted_tuple_out = tuple(item.permute(0, 1, 3, 2) for item in tuple_out)
        #     return permuted_tuple_out


@register_model
def multiscalelgtformer_tiny(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    init_cfg = None

    # please replace checkpoint value with your actual checkpoint.pth
    if pretrained:
        init_cfg = pretrained_cfg

    model = MultiScaleLgtFormer(depths=[2, 2, 6, 2],
                                embed_dim=72,
                                num_heads=[3, 6, 12, 24],
                                out_indices=(0, 1, 2, 3),
                                init_cfg=init_cfg,
                                **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def multiscalelgtformer_small(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    init_cfg = None

    # please replace checkpoint value with your actual checkpoint.pth
    if pretrained:
        init_cfg = pretrained_cfg

    model = MultiScaleLgtFormer(depths=[3, 5, 8, 3],
                                embed_dim=72,
                                num_heads=[3, 6, 12, 24],
                                out_indices=(0, 1, 2, 3),
                                init_cfg=init_cfg,
                                **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def multiscalelgtformer_base(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    init_cfg = None

    # please replace checkpoint value with your actual checkpoint.pth
    if pretrained:
        init_cfg = pretrained_cfg

    model = MultiScaleLgtFormer(depths=[4, 8, 10, 3],
                                embed_dim=96,
                                num_heads=[3, 6, 12, 24],
                                out_indices=(0, 1, 2, 3),
                                init_cfg=init_cfg,
                                **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == "__main__":
    x = torch.rand([2, 3, 224, 224])
    m = multiscalelgtformer_tiny(pretrained=False)
    y = m(x)
    print(y.shape)
