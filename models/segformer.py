# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

from .base_decode_head import BaseDecodeHead
import torch.nn.functional as F

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

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class GlobalFilter2D_real(nn.Module):
    """
    AFNO-style global frequency filter with learnable residual scale.
    Expects x: (B, N, C) with N = H*W. Use spatial_size=(H, W).
    """
    def __init__(self, blocks, dim, h, w, init_alpha=0.1, per_block_alpha=True, r_threshold=0.5):
        super().__init__()
        self.lora_complex_weight = nn.Parameter(
            torch.randn(blocks, h, w, dim, dtype=torch.float32) * 0.02
        )
        self.h, self.w = h, w
        self.lora_ssf_scale, self.lora_ssf_shift = init_ssf_scale_shift(blocks, dim)
        if per_block_alpha:
            self.lora_alpha = nn.Parameter(
                torch.full((blocks, 1, 1, 1), float(init_alpha))
            )
        else:
            self.lora_alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        # Fixed LOW / HIGH masks on the rFFT grid (buffers)
        yy = torch.linspace(-1, 1, self.h)
        xx = torch.linspace(0, 1, self.w)  # rFFT half-plane
        Y, X = torch.meshgrid(yy, xx, indexing='ij')
        r = torch.sqrt(torch.clamp(Y * Y + X * X, 0, 1))  # (H, W2p1)

        low  = (r <= float(r_threshold)).float()
        high = 1.0 - low

        self.register_buffer("mask_low",  low.unsqueeze(0).unsqueeze(0),  persistent=False)  # (1,1,H,W2p1)
        self.register_buffer("mask_high", high.unsqueeze(0).unsqueeze(0), persistent=False)  # (1,1,H,W2p1)

        # --- NEW: edge-focused high-frequency mask (subset of 'high') ---
        # We emphasize *very* high radial frequencies as "edge band".
        edge_start = 0.8  # fixed; can be made a hyper-param if you want
        edge = torch.clamp((r - edge_start) / (1.0 - edge_start + 1e-6), 0, 1)
        edge = edge * high  # only within high band
        self.register_buffer("mask_edge", edge.unsqueeze(0).unsqueeze(0), persistent=False)  # (1,1,H,W2p1)

    def forward(self, block, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        assert self.h == a and self.w == (b // 2 + 1), \
            f"Filter grid {(self.h,self.w)} != rFFT grid {(a,b//2+1)}"

        x = x.to(torch.float32).view(B, a, b, C)
        res = x

        # frequency-domain filtering
        X = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')                # (B,a,w,C)
        W = self.lora_complex_weight[block].squeeze()                   # (a,w,C)
        Y = torch.fft.irfft2(X * W, s=(a, b), dim=(1, 2), norm='ortho') # (B,a,b,C)

        Y = ssf_ada(Y, self.lora_ssf_scale[block], self.lora_ssf_shift[block])
        alpha = self.lora_alpha[block] if self.lora_alpha.dim() == 4 else self.lora_alpha
        out = res + alpha * Y
        return out.reshape(B, N, C)

class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self,  **kwargs):
        super(SegFormerHead, self).__init__(
            input_transform='multiple_select', in_index=[0, 1, 2, 3], **kwargs)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = kwargs['channels']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        H_out, W_out = 56, 56  # all branches upsampled to this
        backbone_channels = [18, 36, 72, 144]
        self.num_branches = len(backbone_channels)
        last_inp_channels = sum(backbone_channels)
        self.lora_freqs = nn.ModuleList([
            GlobalFilter2D_real(
                blocks=1,
                dim=c,
                h=H_out,
                w=W_out // 2 + 1,
                init_alpha=0.6,
                per_block_alpha=True
            )
            for c in backbone_channels
        ]) 

        # cross-task consensus parameters (per branch)
        # how much to move toward low-frequency consensus / edge consensus
        self.lora_eug_alpha_low  = nn.Parameter(torch.zeros(self.num_branches))
        self.lora_eug_alpha_high = nn.Parameter(torch.zeros(self.num_branches))
        self.eug_cap_low  = 0.5   # cap for low-band consensus
        self.eug_cap_high = 0.3   # cap for edge-band alignment

        # NEW: learnable anchor scaling for low-band and edge-band
        self.anchor_scale_low  = nn.Parameter(torch.ones(self.num_branches))
        self.anchor_scale_edge = nn.Parameter(torch.ones(self.num_branches))

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, donors=None):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x0 = c1
        x1u = F.interpolate(c2, (x0_h, x0_w), mode='bilinear', align_corners=False)
        x2u = F.interpolate(c3, (x0_h, x0_w), mode='bilinear', align_corners=False)
        x3u = F.interpolate(c4, (x0_h, x0_w), mode='bilinear', align_corners=False)

        branches = [x0, x1u, x2u, x3u]
        #breakpoint()

        # upsample donors to the same resolution
        donors_up = None
        if donors is not None and len(donors) > 0:
            donors_up = []
            for d in donors:  # each d is [d0, d1, d2, d3] for one donor task
                d0u = d[0] if (d[0].size(2) == x0_h and d[0].size(3) == x0_w) \
                      else F.interpolate(d[0], (x0_h, x0_w),
                                         mode='bilinear', align_corners=False)
                d1u = F.interpolate(d[1], (x0_h, x0_w), mode='bilinear', align_corners=False)
                d2u = F.interpolate(d[2], (x0_h, x0_w), mode='bilinear', align_corners=False)
                d3u = F.interpolate(d[3], (x0_h, x0_w), mode='bilinear', align_corners=False)
                donors_up.append([d0u, d1u, d2u, d3u])

        #breakpoint()

        out_branches = []
        pre_cross = []
        for b_idx, feat in enumerate(branches):
            B, C, H, W = feat.shape  # (B,C,56,56)

            # --- 1) Task-specific global filter (per-resolution, per-bin, per-channel) ---
            tok = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            tok = self.lora_freqs[b_idx](0, tok, spatial_size=(H, W))
            feat = tok.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            pre_cross.append(feat)

            # --- 2) Cross-task frequency consensus: low-band structure + edge-band alignment ---
            if donors_up is not None and len(donors_up) > 0:
                # current task spectrum
                F_rec = torch.fft.rfft2(feat.float(), dim=(2, 3), norm='ortho')  # (B,C,H,W2p1)

                # donor spectra for THIS branch
                donor_branch = [d[b_idx] for d in donors_up]  # list of (B,C,H,W)
                F_don = [torch.fft.rfft2(db.float(), dim=(2, 3), norm='ortho')
                         for db in donor_branch]
                F_avg = torch.stack(F_don, dim=0).mean(dim=0)  # (B,C,H,W2p1)

                # masks from the same filter (radial low, high, edge)
                mL = self.lora_freqs[b_idx].mask_low   # (1,1,H,W2p1)
                mE = self.lora_freqs[b_idx].mask_edge  # (1,1,H,W2p1)

                # ---------- Low-frequency consensus (structure alignment) ----------
                # We align *amplitudes* in the low band, preserving phase of this task
                mL_hw = mL  # (1,1,H,W2p1), broadcasts over (B,C)
                L_rec = mL_hw * F_rec
                L_avg = mL_hw * F_avg

                mag_rec = torch.abs(L_rec)         # (B,C,H,W2p1)
                mag_avg = torch.abs(L_avg)         # (B,C,H,W2p1)
                phase_rec = L_rec / (mag_rec + 1e-6)

                gammaL = self.anchor_scale_low[b_idx]
                sL = torch.sigmoid(self.lora_eug_alpha_low[b_idx]) * self.eug_cap_low

                # amplitude-level convex mix toward (scaled) donor average
                mag_out = mag_rec + sL * (gammaL * mag_avg - mag_rec)
                L_out = phase_rec * mag_out        # only non-zero in low band

                # start F_out from base spectrum, replace low band with L_out
                F_out = (1.0 - mL_hw) * F_rec + L_out

                # ---------- Edge-band alignment (strong edge alignment) ----------
                # Focus on very high radial frequencies (mask_edge ⊂ high)
                mE_hw = mE  # (1,1,H,W2p1)
                gammaE = self.anchor_scale_edge[b_idx]
                sH = torch.sigmoid(self.lora_eug_alpha_high[b_idx]) * self.eug_cap_high

                # move only the edge band toward (scaled) donor average
                Delta_edge = mE_hw * (gammaE * F_avg - F_rec)  # (B,C,H,W2p1)
                F_out = F_out + sH * Delta_edge

                # back to spatial domain
                feat = torch.fft.irfft2(F_out, s=(H, W), dim=(2, 3), norm='ortho')

            out_branches.append(feat)

        c1, c2, c3, c4 = out_branches
        #breakpoint()

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],
                     mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],
                     mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],
                     mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3])

        #breakpoint()

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        #breakpoint()

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
