import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_

import numpy as np
from torchvision.ops import DeformConv2d

from basicsr.utils.registry import ARCH_REGISTRY
from huggingface_hub import PyTorchModelHubMixin


class RefDFE(nn.Module):
    """ Dual Feature Extraction with Reference Fusion
    Args:
        in_features (int): 输入通道数（LR特征维度）
        ref_features (int): 参考特征通道数
        out_features (int): 输出通道数
    """

    def __init__(self, in_features, ref_features, out_features):
        super().__init__()

        # 原有DFE的双分支
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_features // 5, out_features, 1, 1, 0)
        )
        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

        # [新增] 参考特征对齐模块：可变形卷积对齐参考特征
        self.ref_align = nn.Sequential(
            nn.Conv2d(60, out_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            DeformConv2d(out_features, out_features, 3, padding=1)  # 已自动预测 offset
        )

        # [新增] 自适应融合门控机制
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_features * 2, out_features, 1),
            nn.Sigmoid()  # 输出0-1的融合权重
        )

    def forward(self, x, x_size, ref_feat):
        """
        Args:
            x: 输入特征 (B, L, C)
            x_size: 特征图尺寸 (H, W)
            ref_feat: 参考特征 (B, C_ref, H, W)
        """
        B, L, C = x.shape
        H, W = x_size

        # 原始DFE处理
        x_2d = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        conv_out = self.conv(x_2d)  # (B, C_out, H, W)
        linear_out = self.linear(x_2d)  # (B, C_out, H, W)
        dfe_out = conv_out * linear_out  # 原有特征融合

        # [新增] 参考特征对齐与融合
        aligned_ref = self.ref_align(ref_feat)  # (B, C_out, H, W)

        print("dfe_out shape:", dfe_out.shape)
        print("aligned_ref shape:", aligned_ref.shape)
        # 如果 aligned_ref 的空间尺寸与 dfe_out 不同，则用插值调整（以 dfe_out 尺寸为标准）
        if dfe_out.shape[2:] != aligned_ref.shape[2:]:
            aligned_ref = F.interpolate(aligned_ref, size=dfe_out.shape[2:], mode='bilinear', align_corners=False)
            print("After interpolate, aligned_ref shape:", aligned_ref.shape)

        fused = torch.cat([dfe_out, aligned_ref], dim=1)  # (B, 2*C_out, H, W)
        gate = self.fusion_gate(fused)  # (B, C_out, H, W)
        final_out = dfe_out * gate + aligned_ref * (1 - gate)  # 自适应加权融合

        # 恢复序列格式
        final_out = final_out.view(B, -1, H * W).permute(0, 2, 1).contiguous()  # (B, L, C_out)
        return final_out

#new 门控
class GatedMlp(nn.Module):
    """ MLP with Gated Reference Fusion """

    def __init__(self, in_features, ref_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features

        # 主分支
        self.fc1 = nn.Linear(in_features, hidden_features)

        # 参考分支
        self.fc1_ref = nn.Linear(ref_features, hidden_features)

        # 动态门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_features * 2, in_features),
            nn.Sigmoid()  # 输出0-1的权重
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x, ref_feat):
        # 处理LR特征
        x_lr = self.fc1(x)  # (B, L, H)
        x_lr = self.act(x_lr)
        x_lr = self.drop(x_lr)

        # 处理参考特征
        x_ref = self.fc1_ref(ref_feat)  # (B, L, H)
        x_ref = self.act(x_ref)
        x_ref = self.drop(x_ref)

        # 生成门控权重
        gate_input = torch.cat([x_lr, x_ref], dim=-1)
        gate = self.gate(gate_input)  # (B, L, C)

        # 加权融合
        fused = x_lr * gate + x_ref * (1 - gate)
        out = self.fc2(fused)
        return out

class Mlp(nn.Module):
    """ MLP-based Feed-Forward Network
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of heads for spatial self-correlation.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class SCC(nn.Module):
    """ Spatial-Channel Correlation.
    Args:
        dim (int): Number of input channels.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of heads for spatial self-correlation.
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, base_win_size, window_size, num_heads, value_drop=0., proj_drop=0.,ref_dim = 64):

        super().__init__()
        # [新增] 参考特征投影层
        self.ref_proj = nn.Linear(ref_dim, dim)
        # parameters
        self.dim = dim
        self.window_size = window_size 
        self.num_heads = num_heads

        # feature projection
        self.qv = RefDFE(dim, ref_dim, dim)
        self.proj = nn.Linear(dim, dim)

        # dropout
        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # base window size
        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)

        # normalization factor and spatial linear layer for S-SC
        head_dim = dim // (2*num_heads)
        self.scale = head_dim
        self.spatial_linear = nn.Linear(self.window_size[0]*self.window_size[1] // (self.base_win_size[0]*self.base_win_size[1]), 1)

        # define a parameter table of relative position bias
        self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
    
    def spatial_linear_projection(self, x):
        B, num_h, L, C = x.shape
        H, W = self.window_size
        map_H, map_W = self.base_win_size

        x = x.view(B, num_h, map_H, H//map_H, map_W, W//map_W, C).permute(0,1,2,4,6,3,5).contiguous().view(B, num_h, map_H*map_W, C, -1)
        x = self.spatial_linear(x).view(B, num_h, map_H*map_W, C)
        return x
    
    def spatial_self_correlation(self, q, v, ref_q=None):

        # [新增] ref_q为参考特征生成的Query

        print("q shape:", q.shape)
        print("v shape:", v.shape)

        if ref_q is not None:
            corr_map = (q @ ref_q.transpose(-2, -1)) / self.scale  # 与参考特征交互
        else:
            corr_map = (q @ v.transpose(-2, -1)) / self.scale

        B, num_head, L, C = q.shape

        # spatial projection
        v = self.spatial_linear_projection(v)

        # compute correlation map
        corr_map = (q @ v.transpose(-2,-1)) / self.scale

        # add relative position bias
        # generate mother-set
        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        pos = self.pos(rpe_biases)

        # select position bias
        coords_h = torch.arange(self.H_sp, device=v.device)
        coords_w = torch.arange(self.W_sp, device=v.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.base_win_size[0], self.window_size[0]//self.base_win_size[0], self.base_win_size[1], self.window_size[1]//self.base_win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(0,1,3,5,2,4).contiguous().view(
            self.window_size[0] * self.window_size[1], self.base_win_size[0]*self.base_win_size[1], self.num_heads, -1).mean(-1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        corr_map = corr_map + relative_position_bias.unsqueeze(0)

        # transformation
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop).permute(0,2,1,3).contiguous().view(B, L, -1) 

        return x
    
    def channel_self_correlation(self, q, v):
        
        B, num_head, L, C = q.shape

        # apply single head strategy
        q = q.permute(0,2,1,3).contiguous().view(B, L, num_head*C)
        v = v.permute(0,2,1,3).contiguous().view(B, L, num_head*C)

        # compute correlation map
        corr_map = (q.transpose(-2,-1) @ v) / L
        
        # transformation
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop.transpose(-2,-1)).permute(0,2,1).contiguous().view(B, L, -1)

        return x

    def forward(self, x, ref_feat):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        xB, xH, xW, xC = x.shape
        # [修改] 将参考特征与LR特征共同输入DFE（假设DFE已扩展为RefDFE）
        qv = self.qv(x.view(xB, -1, xC), (xH, xW), ref_feat)  # 传入参考特征
        qv = qv.view(xB, xH, xW, xC)

        # window partition
        qv = window_partition(qv, self.window_size)
        qv = qv.view(-1, self.window_size[0]*self.window_size[1], xC)

        # qv splitting
        B, L, C = qv.shape
        qv = qv.view(B, L, 2, self.num_heads, C // (2*self.num_heads)).permute(2,0,3,1,4).contiguous()
        q, v = qv[0], qv[1] # B, num_heads, L, C//num_heads

        # spatial self-correlation (S-SC)
        x_spatial = self.spatial_self_correlation(q, v)
        x_spatial = x_spatial.view(-1, self.window_size[0], self.window_size[1], C//2)
        x_spatial = window_reverse(x_spatial, (self.window_size[0],self.window_size[1]), xH, xW)  # xB xH xW xC

        # channel self-correlation (C-SC)
        x_channel = self.channel_self_correlation(q, v)
        x_channel = x_channel.view(-1, self.window_size[0], self.window_size[1], C//2)
        x_channel = window_reverse(x_channel, (self.window_size[0], self.window_size[1]), xH, xW) # xB xH xW xC

        # spatial-channel information fusion
        x = torch.cat([x_spatial, x_channel], -1)
        x = self.proj_drop(self.proj(x))

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class HierarchicalTransformerBlock(nn.Module):
    """ Hierarchical Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, base_win_size, window_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ref_dim=64):
        super().__init__()
        # [新增] 若SCC需要参考特征投影
        self.ref_proj = nn.Linear(ref_dim, dim)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size 
        self.mlp_ratio = mlp_ratio

        # check window size
        if (window_size[0] > base_win_size[0]) and (window_size[1] > base_win_size[1]):
            assert window_size[0] % base_win_size[0] == 0, "please ensure the window size is smaller than or divisible by the base window size"
            assert window_size[1] % base_win_size[1] == 0, "please ensure the window size is smaller than or divisible by the base window size"


        self.norm1 = norm_layer(dim)
        self.correlation = SCC(
            dim, base_win_size=base_win_size, window_size=self.window_size, num_heads=num_heads,
            value_drop=value_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def check_image_size(self, x, win_size):
        x = x.permute(0,3,1,2).contiguous()
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = x.permute(0,2,3,1).contiguous()
        return x

    def forward(self, x, x_size, win_size, ref_feat):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)
        
        # padding
        x = self.check_image_size(x, win_size)
        _, H_pad, W_pad, _ = x.shape # shape after padding

        # 修改SCC调用
        x = self.correlation(x, ref_feat)  # [修改] 传入参考特征

        # unpad
        x = x[:, :H, :W, :].contiguous()

        # norm
        x = x.view(B, H * W, C)
        x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Hierarchical Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0.,drop_path=0., norm_layer=nn.LayerNorm,
                   downsample=None, use_checkpoint=False, hier_win_ratios=[0.5,1,2,4,6,8]):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.win_hs = [int(base_win_size[0] * ratio) for ratio in hier_win_ratios]
        self.win_ws = [int(base_win_size[1] * ratio) for ratio in hier_win_ratios]

        # build blocks
        self.blocks = nn.ModuleList([
            HierarchicalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, 
                                 base_win_size=base_win_size,
                                 window_size=(self.win_hs[i], self.win_ws[i]),
                                 mlp_ratio=mlp_ratio,
                                 drop=drop, value_drop=value_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, ref_feat):  # [新增] ref_feat参数
        i = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                # [修改] 传递参考特征到检查点
                x = checkpoint.checkpoint(blk, x, x_size, (self.win_hs[i], self.win_ws[i]), ref_feat)
            else:
                # [修改] 常规调用时传递参考特征
                x = blk(x, x_size, (self.win_hs[i], self.win_ws[i]), ref_feat)
            i = i + 1
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class RHTB(nn.Module):
    """Residual Hierarchical Transformer Block (RHTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample=None, use_checkpoint=False, img_size=224, patch_size=4, 
                 resi_connection='1conv', hier_win_ratios=[0.5,1,2,4,6,8], ref_dim=3):
        super(RHTB, self).__init__()
        # [新增] 参考图像编码器
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(ref_dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         base_win_size=base_win_size,
                                         mlp_ratio=mlp_ratio,
                                         drop=drop, value_drop=value_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         hier_win_ratios=hier_win_ratios)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size, ref_feat):  # [新增] ref_feat参数
        # [修改] 将参考特征传递到BasicLayer
        residual_output = self.residual_group(x, x_size, ref_feat)  # 传入ref_feat
        # 后续处理保持原逻辑
        conv_output = self.conv(self.patch_unembed(residual_output, x_size))
        return self.patch_embed(conv_output) + x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


@ARCH_REGISTRY.register()
class HiT_SIR(nn.Module, PyTorchModelHubMixin):
    """ HiT-SIR (RefSR版本)
    此版本在原有 HiT-SIR 架构上增加了对参考图像（ref）的支持，
    包括浅层参考特征编码、对齐模块以及在各层传递参考特征。

    Args:
        img_size (int | tuple(int)): 输入图像尺寸，默认64。
        patch_size (int | tuple(int)): Patch尺寸，默认1。
        in_chans (int): 输入通道数，默认3。
        embed_dim (int): Patch embedding 维度，默认60。
        depths (tuple[int]): 每个 Transformer block 的深度。
        num_heads (tuple[int]): 各层 self-attention 头数。
        base_win_size (tuple[int]): 基础窗口尺寸，例如[8, 8]。
        mlp_ratio (float): FFN的隐藏层比率。
        drop_rate (float): Dropout比率。
        value_drop_rate (float): Value dropout比率。
        drop_path_rate (float): Stochastic depth比率。
        norm_layer (nn.Module): 归一化层，默认nn.LayerNorm。
        ape (bool): 是否使用绝对位置嵌入。
        patch_norm (bool): 是否在 patch embedding 后做归一化。
        use_checkpoint (bool): 是否使用checkpoint节省内存。
        upscale (int): 放大倍数，例如4。
        img_range (float): 图像范围，1. 或255.
        upsampler (str): 上采样方式，支持'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None。
        resi_connection (str): 残差连接模块，可选'1conv'或'3conv'。
        hier_win_ratios (list): 层次窗口比例列表。
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 base_win_size=[8, 8], mlp_ratio=2.,
                 drop_rate=0., value_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='pixelshuffledirect',
                 resi_connection='1conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
                 **kwargs):
        super(HiT_SIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.base_win_size = base_win_size

        #####################################
        # 1. 浅层特征提取 (LR分支)
        #####################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        # 参考图像编码器：用于提取参考图像的浅层特征
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        )

        #####################################
        # 2. 深层特征提取
        #####################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        # 将 conv_first 输出经过 PatchEmbed 转为序列形式
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=embed_dim, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            num_patches = self.patch_embed.num_patches
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         base_win_size=base_win_size,
                         mlp_ratio=self.mlp_ratio,
                         drop=drop_rate, value_drop=value_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection,
                         hier_win_ratios=hier_win_ratios,
                         ref_dim=in_chans)  # 参考图像通道数
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )

        #####################################
        # 3. 重建模块 (上采样使用 pixelshuffledirect)
        #####################################
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, '仅支持 x4'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def feed_data(self, lq, gt, ref):
        """
        feed_data: 设置训练数据。
        Args:
            lq: 低分辨图像 tensor (B, C, H, W)
            gt: 高分辨图像 tensor (B, C, H*scale, W*scale)
            ref: 参考图像 tensor (B, C, H, W)；若无可传 None
        """
        device = self.conv_first.weight.device
        self.lq = lq.to(device)
        self.gt = gt.to(device)
        if ref is not None:
            self.ref = ref.to(device)
        else:
            self.ref = None

    def forward_features(self, x, ref):
        """
        x: 由 conv_first 处理后的 LR 特征
        ref: 参考图像 tensor，用于额外特征提取
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        # 对参考图进行编码（浅层参考特征）
        ref_feat = self.ref_encoder(ref)
        # 逐层传递参考特征
        for layer in self.layers:
            x = layer(x, x_size, ref_feat)
        x = self.norm(x)  # (B, L, C)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x, ref):
        """
        forward 接口中同时输入 LR 图像 x 和参考图像 ref。
        """
        H, W = x.shape[2:4]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        # 浅层特征提取
        x_conv = self.conv_first(x)
        # 深层特征提取（传入参考图信息）
        x_feat = self.forward_features(x_conv, ref)
        if self.upsampler == 'pixelshuffle':
            x = self.conv_after_body(x_feat) + x_conv
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_after_body(x_feat) + x_conv
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_after_body(x_feat) + x_conv
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))


if __name__ == '__main__':
    upscale = 4
    base_win_size = [8, 8]
    height = (1024 // upscale // base_win_size[0] + 1) * base_win_size[0]
    width = (720 // upscale // base_win_size[1] + 1) * base_win_size[1]
    # 初始化 HiT-SIR (RefSR版本) 模型
    model = HiT_SIR(upscale=4, img_size=(height, width),
                    base_win_size=base_win_size, img_range=1., depths=[6, 6, 6, 6],
                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("params: ", params_num)


