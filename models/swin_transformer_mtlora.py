# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Swin Transformer (https://github.com/microsoft/Swin-Transformer)
#
# Original file:
# Copyright (c) 2021 Microsoft
# Licensed under the MIT License
# Written by Ze Liu
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.lora8 import MTLoRALinear  # use 7 for DoRA
#from models.lora8 import MTDoRALinear as MTLoRALinear
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random 
import math
from .utils import show_frequency_bands_grid

try:
    import os
    import sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


def init_ssf_scale_shift(blocks, dim):
    scale = nn.Parameter(torch.ones(blocks, dim))
    shift = nn.Parameter(torch.zeros(blocks, dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift

def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


# declare the spatial sizes you expect (yours: 56, 28, 14, 14)
stage_hw = [(56,56), (28,28), (14,14), (14,14)]


class GlobalFilter2D_real(nn.Module):
    """
    AFNO-style global frequency filter with learnable residual scale.
    Expects x: (B, N, C) with N = H*W. Use spatial_size=(H, W).
    """
    def __init__(self, blocks, dim, h, w, init_alpha=0.1, per_block_alpha=True):
        super().__init__()
        # real-valued spectral weights on rFFT grid (H, W//2+1, C)
        self.lora_complex_weight = nn.Parameter(torch.randn(blocks, h, w, dim, dtype=torch.float32) * 0.02)
        self.h, self.w = h, w
        self.lora_ssf_scale, self.lora_ssf_shift = init_ssf_scale_shift(blocks, dim)
        if per_block_alpha:
            self.lora_alpha = nn.Parameter(torch.full((blocks, 1, 1, 1), float(init_alpha)))
        else:
            self.lora_alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, block, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size  # feature H, W (not input image H, W!)
        # sanity: weight must match rFFT grid
        assert self.h == a and self.w == (b // 2 + 1), f"Filter grid {(self.h,self.w)} != rFFT grid {(a,b//2+1)}"

        x = x.to(torch.float32).view(B, a, b, C)
        res = x
        X = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')                # (B, a, b//2+1, C)
        W = self.lora_complex_weight[block].squeeze()                        # (a, b//2+1, C)
        Y = torch.fft.irfft2(X * W, s=(a, b), dim=(1, 2), norm='ortho') # (B, a, b, C)

        # FiLM (scale/shift) + residual scale
        Y = ssf_ada(Y, self.lora_ssf_scale[block], self.lora_ssf_shift[block])
        alpha = self.lora_alpha[block] if self.lora_alpha.dim() == 4 else self.lora_alpha
        out = res + alpha * Y
        return out.reshape(B, N, C)


class GlobalFilter2D_real(nn.Module):
    """
    AFNO-style global frequency filter with learnable residual scale.
    Works on features (activations), not weights.
    x: (B, N, C) with N = H*W. Pass spatial_size=(H, W).

    All learnable params are prefixed with 'lora_'.
      - lora_amp_p:     identity-safe amplitude logits (per freq bin, per channel)
      - lora_chA/B:     (optional) low-rank channel mixing in frequency: X @ (I + A B)
      - lora_ssf_scale/shift: FiLM after iFFT
      - lora_alpha:     residual scale
    """
    def __init__(self, blocks, dim, h, w,
                 init_alpha=0.1, per_block_alpha=True,
                 amp_scale=0.25,     # max ± amplitude deviation around 1.0
                 rank=0):            # >0 enables channel mixing (e.g., 8 or 16)
        super().__init__()
        self.h, self.w, self.dim = h, w, dim
        self.amp_scale = float(amp_scale)

        # --- identity-safe amplitude params ---
        # amp = 1 + amp_scale * tanh(lora_amp_p)  ∈ [1-amp_scale, 1+amp_scale]
        self.lora_amp_p = nn.Parameter(torch.zeros(blocks, h, w, dim))

        # --- optional low-rank channel mixing in frequency: X @ (I + A B) ---
        self.rank = int(rank)
        if self.rank > 0:
            self.lora_chA = nn.Parameter(torch.randn(blocks, dim, self.rank) * 0.02)
            self.lora_chB = nn.Parameter(torch.randn(blocks, self.rank, dim) * 0.02)

        # FiLM scale/shift (learnable)
        self.lora_ssf_scale, self.lora_ssf_shift = init_ssf_scale_shift(blocks, dim)

        # residual scale (learnable)
        if per_block_alpha:
            self.lora_alpha = nn.Parameter(torch.full((blocks, 1, 1, 1), float(init_alpha)))
        else:
            self.lora_alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, block, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size  # feature H, W

        # rFFT grid & channel checks
        assert self.h == a and self.w == (b // 2 + 1), \
            f"Filter grid {(self.h,self.w)} != rFFT grid {(a,b//2+1)}"
        assert C == self.dim, f"Channel mismatch: C={C} vs dim={self.dim}"

        x = x.to(torch.float32).view(B, a, b, C)
        res = x

        # FFT
        X = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')                 # (B, a, b//2+1, C)

        # optional low-rank channel mixing: X @ (I + A B)
        if self.rank > 0:
            I = torch.eye(C, device=x.device, dtype=x.dtype)
            Mch = I + (self.lora_chA[block] @ self.lora_chB[block])       # (C, C)
            X = torch.matmul(X, Mch)                                      # (..., C)

        # identity-safe amplitude
        amp = 1.0 + self.amp_scale * torch.tanh(self.lora_amp_p[block].squeeze())  # (a, b//2+1, C)
        X = X * amp

        # iFFT
        Y = torch.fft.irfft2(X, s=(a, b), dim=(1, 2), norm='ortho')       # (B, a, b, C)

        # FiLM + residual scale
        Y = ssf_ada(Y, self.lora_ssf_scale[block], self.lora_ssf_shift[block])
        alpha = self.lora_alpha[block] if self.lora_alpha.dim() == 4 else self.lora_alpha
        out = res + alpha * Y
        return out.view(B, N, C)


class GlobalFilter2D_real(nn.Module):
    """
    Global 2D frequency filter for dense tasks (PEFT-friendly).
    x: (B, N, C), N = H*W. Pass spatial_size=(H, W).

    Upgrades:
      - identity-safe banded amplitude: amp = 1 + amp_scale * tanh(sum_k g_k * B_k)
      - optional edge-gated residual to focus sharpening at boundaries
    """
    def __init__(self, blocks, dim, h, w,
                 init_alpha=1.0, per_block_alpha=True,
                 num_bands=4, amp_scale=0.5,
                 per_channel_bands=True,
                 edge_gate=False, edge_temp=6.0):
        super().__init__()
        self.h, self.w, self.dim = h, w, dim
        self.num_bands = int(num_bands)
        self.amp_scale = float(amp_scale)
        self.per_channel_bands = bool(per_channel_bands)
        self.edge_gate = bool(edge_gate)
        self.edge_temp = float(edge_temp)

        # band gains (learnable); identity-safe via tanh in forward
        if self.per_channel_bands:
            # (blocks, K, C)
            self.lora_band_gains = nn.Parameter(torch.zeros(blocks, self.num_bands, dim))
        else:
            # (blocks, K)
            self.lora_band_gains = nn.Parameter(torch.zeros(blocks, self.num_bands))

        # FiLM after iFFT
        self.lora_ssf_scale, self.lora_ssf_shift = init_ssf_scale_shift(blocks, dim)

        # residual scale
        if per_block_alpha:
            self.lora_alpha = nn.Parameter(torch.full((blocks, 1, 1, 1), float(init_alpha)))
        else:
            self.lora_alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        # cache for radial band masks per (H, W//2+1)
        self.register_buffer('_cached_hw', torch.tensor([-1, -1]), persistent=False)
        self.register_buffer('_bands', None, persistent=False)  # (K, H, W//2+1)
        # simple Sobel kernels for edge gate
        if self.edge_gate:
            kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
            ky = kx.t()
            self.register_buffer('_kx', kx.view(1,1,3,3), persistent=False)
            self.register_buffer('_ky', ky.view(1,1,3,3), persistent=False)

    def _make_radial_bands(self, H, W2p1, device):
        if self._cached_hw[0].item() == H and self._cached_hw[1].item() == W2p1 and self._bands is not None:
            return self._bands
        yy = torch.linspace(-1, 1, H, device=device)
        xx = torch.linspace(0, 1, W2p1, device=device)  # rFFT half-plane
        Y, X = torch.meshgrid(yy, xx, indexing='ij')
        r = torch.sqrt(torch.clamp(Y*Y + X*X, 0, 1))
        edges = torch.linspace(0, 1, self.num_bands+1, device=device)
        masks = []
        for k in range(self.num_bands):
            lo, hi = edges[k], edges[k+1]
            masks.append(((r >= lo) & (r <= hi)).float())
        self._bands = torch.stack(masks, 0)  # (K,H,W2p1)
        self._cached_hw = torch.tensor([H, W2p1], device=device)
        return self._bands

    def forward(self, block, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        assert self.h == a and self.w == (b // 2 + 1), f"Filter grid {(self.h,self.w)} != rFFT grid {(a,b//2+1)}"
        assert C == self.dim

        x = x.to(torch.float32).view(B, a, b, C)
        res = x
        #breakpoint()

        X = torch.fft.rfft2(x, dim=(1,2), norm='ortho')               # (B,a,b//2+1,C)

        # --- banded, identity-safe amplitude ---
        Bm = self._make_radial_bands(a, b//2+1, x.device)             # (K,a,b//2+1)
        gains = torch.tanh(self.lora_band_gains[block])               # (K,) or (K,C)
        if gains.dim() == 1:
            amp = 1.0 + self.amp_scale * (Bm * gains.view(-1,1,1)).sum(0).unsqueeze(-1)  # (a,b//2+1,1)
        else:
            # per-channel bands
            # (K,H,W2p1) x (K,C) -> (H,W2p1,C)
            amp = 1.0 + self.amp_scale * torch.einsum('khw,kc->hwc', Bm, gains)

        X = X * amp                                                   # amplitude only

        Y = torch.fft.irfft2(X, s=(a,b), dim=(1,2), norm='ortho')     # (B,a,b,C)
        Y = ssf_ada(Y, self.lora_ssf_scale[block], self.lora_ssf_shift[block])

        # --- optional edge-gated residual (focus changes near boundaries) ---
        if self.edge_gate:
            # crude edge map from features: gradient mag of channel-avg
            g = x.mean(-1, keepdim=True).permute(0,3,1,2)             # (B,1,a,b)
            gx = F.conv2d(g, self._kx, padding=1)
            gy = F.conv2d(g, self._ky, padding=1)
            mag = torch.sqrt(gx*gx + gy*gy)                           # (B,1,a,b)
            gate = torch.sigmoid(self.edge_temp * (mag / (mag.mean()+1e-6)))
            Y = Y * gate.permute(0,2,3,1)                             # broadcast over C

        alpha = self.lora_alpha[block] if self.lora_alpha.dim()==4 else self.lora_alpha
        out = res + alpha * Y
        return out.view(B, N, C)


class CompatLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, x_tasks: dict = None) -> Tensor:
        return super().forward(input), None


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., lora=False, tasks=None, mtlora=None, layer_idx=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if mtlora.FC1_ENABLED:
            self.fc1 = MTLoRALinear(in_features, hidden_features, r=mtlora.R_PER_TASK_LIST[layer_idx],
                                    lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=(
                                        tasks if (lora or mtlora.INTERMEDIATE_SPECIALIZATION) else None),
                                    trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.fc1 = CompatLinear(in_features, hidden_features)
        self.act = act_layer()
        if mtlora.FC2_ENABLED:
            self.fc2 = MTLoRALinear(hidden_features, out_features, r=mtlora.R_PER_TASK_LIST[layer_idx],
                                    lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=(
                                        tasks if (lora or mtlora.INTERMEDIATE_SPECIALIZATION) else None),
                                    trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.fc2 = CompatLinear(hidden_features, out_features)
        self.tasks = tasks
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_tasks=None, masks=None, masks_tasks=None):
        total_reg_loss = 0
        if masks is None:
            x, fc1_lora_tasks, reg_loss = self.fc1(x, x_tasks, masks=None)
        else:
            x, fc1_lora_tasks, reg_loss = self.fc1(x, x_tasks, masks=masks, masks_tasks=masks_tasks)
        total_reg_loss += reg_loss
        x = self.act(x)
        x = self.drop(x)
        if fc1_lora_tasks is not None:
            for task in self.tasks:
                fc1_lora_tasks[task] = self.act(fc1_lora_tasks[task])
                fc1_lora_tasks[task] = self.drop(fc1_lora_tasks[task])
        if masks is None:
            x, fc2_lora_tasks, reg_loss = self.fc2(x, fc1_lora_tasks, masks=None)
        else:
            x, fc2_lora_tasks, reg_loss = self.fc2(x, fc1_lora_tasks, masks=masks, masks_tasks=masks_tasks)
        total_reg_loss += reg_loss
        x = self.drop(x)
        if fc2_lora_tasks is not None:
            for task in self.tasks:
                fc2_lora_tasks[task] = self.drop(fc2_lora_tasks[task])
        return x, fc2_lora_tasks, total_reg_loss


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., lora=False, tasks=None, mtlora=None, layer_idx=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.lora = lora

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            # 2*Wh-1 * 2*Ww-1, nH
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        if mtlora.QKV_ENABLED:
            self.qkv = MTLoRALinear(dim, dim * 3, r=mtlora.R_PER_TASK_LIST[layer_idx],
                                    lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=None, bias=qkv_bias,
                                    trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.qkv = CompatLinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if mtlora.PROJ_ENABLED:
            self.proj = MTLoRALinear(dim, dim, r=mtlora.R_PER_TASK_LIST[layer_idx],
                                     lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=(
                tasks if (lora or mtlora.INTERMEDIATE_SPECIALIZATION) else None),
                trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.proj = CompatLinear(dim, dim)

        self.tasks = tasks
        self.has_tasks = (lora or mtlora.INTERMEDIATE_SPECIALIZATION)
        #if self.has_tasks:
            #self.q_alpha = 100
            #self.k_alpha = 100
            #self.v_alpha = 100
            #self._gate_eps = 40  # keeps gates close to 1.0
            #self.center_exp_gates = False
            #self.lora_q_beta = nn.ParameterDict({t: nn.Parameter(torch.zeros(self.num_heads)) for t in self.#tasks})
            #self.lora_k_beta = nn.ParameterDict({t: nn.Parameter(torch.zeros(self.num_heads)) for t in self.#tasks})
            #self.lora_v_beta = nn.ParameterDict({t: nn.Parameter(torch.zeros(self.num_heads)) for t in self.#tasks})
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _exp_gate(self, beta, alpha, log_cap=None, center=False):
        logg = alpha * beta
        gamma = torch.exp(logg)               # > 0, multiplicative
        if center:
            gamma = gamma / (gamma.mean() + 1e-8)  # geometric-mean ≈ 1
        return gamma

    def forward(self, x, mask=None, masks=None, masks_tasks=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        total_reg_loss = 0
        if masks is None:
            qkv, _, reg_loss = self.qkv(x)
        else:
            qkv, _, reg_loss = self.qkv(x, masks=masks, masks_tasks=masks_tasks)
        total_reg_loss += reg_loss  # <<< accumulate qkv reg loss

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C //
                          self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            # Wh*Ww,Wh*Ww,nH
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        """x_tasks = None
        if self.has_tasks:
            x_tasks = {}
            for t in self.tasks:
                gv = (1.0 + self._gate_eps * self.lora_v_beta[t]).view(1, -1, 1, 1)
                gq = (1.0 + self._gate_eps * self.lora_q_beta[t]).view(1, -1, 1, 1)
                gk = (1.0 + self._gate_eps * self.lora_k_beta[t]).view(1, -1, 1, 1)
                #print('gv', gv)
                #print('gq', gq)
                #print('gk', gk)

                q_t = q * gq
                k_t = k * gk
                v_t = v * gv

                attn_t = (q_t @ k_t.transpose(-2, -1)) + relative_position_bias.unsqueeze(0)

                if mask is not None:
                    nW = mask.shape[0]
                    attn = attn_t.view(B_ // nW, nW, self.num_heads, N,
                                    N) + mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, self.num_heads, N, N)
                    attn = self.softmax(attn)
                else:
                    attn = self.softmax(attn_t)

                attn = self.attn_drop(attn)
                x_t = (attn @ v_t).transpose(1, 2).reshape(B_, N, C)
                x_tasks[t] = x_t """

        if masks is None:
            x, x_proj_lora_tasks, reg_loss = self.proj(x)
        else:
            x, x_proj_lora_tasks, reg_loss = self.proj(x, masks=masks, masks_tasks=masks_tasks)

        total_reg_loss += reg_loss

        x = self.proj_drop(x)
        if x_proj_lora_tasks is not None:
            for task in self.tasks:
                x_proj_lora_tasks[task] = self.proj_drop(
                    x_proj_lora_tasks[task])
        return x, x_proj_lora_tasks, total_reg_loss

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, lora=False, tasks=None, mtlora=None, layer_idx=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.tasks = tasks
        self.lora = lora
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, lora=lora, tasks=tasks, mtlora=mtlora, layer_idx=layer_idx)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, lora=lora, tasks=tasks, mtlora=mtlora, layer_idx=layer_idx)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x, masks=None, masks_tasks=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        
        assert L == H * W, "input feature has wrong size"

        total_reg_loss = 0
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            x_windows = window_partition(x, self.window_size)

        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        if masks is None:
            attn_windows, attn_windows_lora_tasks, reg_loss = self.attn(
                x_windows, mask=self.attn_mask)
        else:
            attn_windows, attn_windows_lora_tasks, reg_loss = self.attn(
                x_windows, mask=self.attn_mask, masks=masks, masks_tasks=masks_tasks)

        total_reg_loss += reg_loss

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = x
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                # <<< roll back masks too (and masks_tasks)
            else:
                x = WindowProcessReverse.apply(
                    attn_windows, B, H, W, C, self.shift_size, self.window_size)
                # <<< masks already handled via explicit reverse + roll above
        else:
            shifted_x = x
            x = shifted_x
        
        if attn_windows_lora_tasks is not None:
            for task in self.tasks:
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task].view(
                    -1, self.window_size, self.window_size, C)
                attn_windows_lora_tasks[task] = window_reverse(
                    attn_windows_lora_tasks[task], self.window_size, H, W)
                if self.shift_size > 0:
                    attn_windows_lora_tasks[task] = torch.roll(attn_windows_lora_tasks[task], shifts=(
                        self.shift_size, self.shift_size), dims=(1, 2))
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task].view(
                    B, H * W, C)
                attn_windows_lora_tasks[task] = shortcut + self.drop_path(attn_windows_lora_tasks[task])

        x = x.view(B, H * W, C)
        # <<< flatten masks like x (and masks_tasks)

        x = shortcut + self.drop_path(x)

        # FFN
        if masks is None:
            mlp_result, mlp_lora_tasks, reg_loss = self.mlp(
                self.norm2(x), {task: self.norm2(attn_windows_lora_tasks[task]) for task in self.tasks} if attn_windows_lora_tasks is not None else None)
        else:
            mlp_result, mlp_lora_tasks, reg_loss = self.mlp(
                self.norm2(x), {task: self.norm2(attn_windows_lora_tasks[task]) for task in self.tasks} if attn_windows_lora_tasks is not None else None, masks=masks, masks_tasks=masks_tasks)
        
        total_reg_loss += reg_loss

        if mlp_lora_tasks is None:
            if masks is None:
                return x + self.drop_path(mlp_result), None, total_reg_loss
            else:
                return x + self.drop_path(mlp_result), None, masks, masks_tasks, total_reg_loss
        else:
            if attn_windows_lora_tasks is None:
                for task in self.tasks:
                    mlp_lora_tasks[task] = self.drop_path(mlp_lora_tasks[task])
            else:
                for task in self.tasks:
                    mlp_lora_tasks[task] = attn_windows_lora_tasks[task] + \
                        self.drop_path(mlp_lora_tasks[task])
            
            if masks is None:
                return x + self.drop_path(mlp_result), mlp_lora_tasks, total_reg_loss
            else:
                return x + self.drop_path(mlp_result), mlp_lora_tasks, masks, masks_tasks, total_reg_loss


        def extra_repr(self) -> str:
            return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

        def flops(self):
            flops = 0
            H, W = self.input_resolution
            # norm1
            flops += self.dim * H * W
            # W-MSA/SW-MSA
            nW = H * W / self.window_size / self.window_size
            flops += nW * self.attn.flops(self.window_size * self.window_size)
            # mlp
            flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
            # norm2
            flops += self.dim * H * W
            return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, layer_idx=0, mtlora=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        if mtlora.DOWNSAMPLER_ENABLED:
            self.reduction = MTLoRALinear(4 * dim, 2 * dim, r=mtlora.R_PER_TASK_LIST[layer_idx],
                                          lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=None, bias=False,
                                          trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.reduction = CompatLinear(4 * dim, 2 * dim, bias=False)

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
        x, _, reg_loss = self.reduction(x)

        return x, reg_loss

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, tasks=None, mtlora=None, layer_idx=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.tasks = tasks

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 lora=(i == depth - 1),
                                 tasks=tasks,
                                 mtlora=mtlora,
                                 layer_idx=layer_idx)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, layer_idx=layer_idx, mtlora=mtlora)
        else:
            self.downsample = None

    def forward(self, x, masks=None, masks_tasks=None):
        total_reg_loss = 0
        for blk in self.blocks:
            if masks is None:
                x, tasks_lora, reg_loss = blk(x)
            else:
                x, tasks_lora, masks, masks_tasks, reg_loss = blk(
                    x, masks=masks, masks_tasks=masks_tasks
                )
            total_reg_loss += reg_loss        

        if self.downsample is not None:
            # features only
            x, _ = self.downsample(x)
            if tasks_lora is not None:
                for task in self.tasks:
                    tasks_lora[task], reg_loss_t = self.downsample(tasks_lora[task].float())
                    total_reg_loss += reg_loss_t

        if masks is None:
            return x, tasks_lora, total_reg_loss
        else:
            return x, tasks_lora, masks, masks_tasks, total_reg_loss

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


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
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * \
            (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerMTLoRA(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 basic_layer=BasicLayer, tasks=None, mtlora=None, ** kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.tasks = tasks
        self.mtlora = mtlora

        # Print lora params:
        if mtlora is not None:
            print("\nMTLoRA params:")
            print(mtlora)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                # stochastic depth decay rule
                                                sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = basic_layer(dim=int(embed_dim * 2 ** i_layer),
                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                  patches_resolution[1] // (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(
                                    depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (
                                    i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                fused_window_process=fused_window_process,
                                tasks=tasks,
                                mtlora=self.mtlora,
                                layer_idx=i_layer)
            self.layers.append(layer)

        #self.freq_shared = nn.ModuleList()
        #self.freq_tasks  = nn.ModuleDict({t: nn.ModuleList() for t in self.tasks})

        H0, W0 = self.patches_resolution  # grid from PatchEmbed
        for i_layer in range(self.num_layers):
            # Resolution & channels AFTER layer i_layer (i.e., after PatchMerging if present)
            if i_layer < self.num_layers - 1:
                H_out = H0 // (2 ** (i_layer + 1))
                W_out = W0 // (2 ** (i_layer + 1))
                C_out = int(self.embed_dim * 2 ** (i_layer + 1))  # doubled by PatchMerging
            else:
                H_out = H0 // (2 ** i_layer)
                W_out = W0 // (2 ** i_layer)
                C_out = int(self.embed_dim * 2 ** i_layer)

            # rFFT grid width is W_out//2 + 1
            #self.freq_shared.append(
            #    GlobalFilter2D_real(blocks=1, dim=C_out, h=H_out, w=W_out//2 + 1, init_alpha=0.6, per_block_alpha=True)
            #)
            #for t in self.tasks:
            #    self.freq_tasks[t].append(
            #        GlobalFilter2D_real(blocks=1, dim=C_out, h=H_out, w=W_out//2 + 1, init_alpha=0.6, per_block_alpha=True)
            #    )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward_features(self, x, return_stages=False, flatten_ft=False, masks=None, print_masks=False):
        if print_masks:
            visualize_tensors(x, masks, num_samples=6)
        """ breakpoint()
        org_img = x
        show_frequency_bands_grid(
            batch_image=org_img,
            num_samples=3,                          # must be < B
            save_path="freq_panels.png",
            k_bands=((0.00,0.02),(0.02,0.06),(0.06,0.12),(0.12,0.25),(0.25,0.50)),
            band_thresh_pct=92,                     # same percentile for all bands
            denorm="auto",                          # handles ImageNet-normalized tensors too
            seed=0
        )
        breakpoint() """
        x = self.patch_embed(x)                      # x: [B, L0, C0]
        H0, W0 = self.patches_resolution            # patch grid from PatchEmbed
        masks_tasks = masks
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        if return_stages:
            out = []
        H, W = H0, W0  # track current feature map resolution
        total_reg_loss = 0.0
        for i, layer in enumerate(self.layers):
            if masks is None:
                x, tasks_lora, reg_loss = layer(x)
            else:
                x, tasks_lora, masks, masks_tasks, reg_loss = layer(x, masks=masks, masks_tasks=masks_tasks)

            total_reg_loss += reg_loss

            # === Update (H, W) to reflect the output of this layer (after its PatchMerging, if any) ===
            if i < self.num_layers - 1:
                # this layer performs PatchMerging → halves spatial dims
                H, W = H // 2, W // 2
            
            # === RUN SHARED FILTER on x ===
            #x = self.freq_shared[i](block=0, x=x, spatial_size=(H, W))

            # Ensure tasks_lora exists
            if tasks_lora is None:
                tasks_lora = {task: x for task in self.tasks}

            # === RUN TASK-SPECIFIC FILTERS on tasks_lora ===
            #for t in self.tasks:
            #    tasks_lora[t] = self.freq_tasks[t][i](block=0, x=tasks_lora[t], spatial_size=(H, W))
                #breakpoint()
            
            if return_stages:
                out.append((x, tasks_lora))

        if return_stages:
            if masks is not None:
                return out, total_reg_loss, masks, masks_tasks
            else:
                return out, total_reg_loss
        else:
            if flatten_ft:
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
            if masks is not None:
                return out, total_reg_loss, masks, masks_tasks
            else:
                return out, total_reg_loss

    def forward(self, x, return_stages=False, flatten_ft=False, masks=None):
        masks_tasks = masks
        if masks is None:
            x, reg_loss = self.forward_features(x, return_stages, flatten_ft, masks=None)
        else:
            x, reg_loss, masks, masks_tasks = self.forward_features(x, return_stages, flatten_ft, masks=masks)
        x = self.head(x)
        return x, reg_loss, masks, masks_tasks

    def flops(self, images=None, logger=None, detailed=False):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * \
            self.patches_resolution[0] * \
            self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
