# ------------------------------------------------------------------------------

# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Minor changes made by Simon Vandenhende
# ------------------------------------------------------------------------------

# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
#
# Original file:
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun(sunk@mail.ustc.edu.cn)
# Minor changes made by Simon Vandenhende
#
# Modifications:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np
import math
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from termcolor import colored
# from .sync_bn.inplace_abn.bn import InPlaceABNSync
from ptflops import get_model_complexity_info

# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
BatchNorm2d = functools.partial(nn.BatchNorm2d)
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)



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

        
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        return x

    def init_weights(self, pretrained='',):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            print('Using pretrained weights from location {}'.format(pretrained))
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
            #    print('=> loading {} from pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class HighResolutionFuse(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionFuse, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=0.1),
            nn.ReLU(inplace=False))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x


class GlobalFilter2D_real(nn.Module):
    """
    AFNO-style global frequency filter with learnable residual scale.
    PARAM-EFFICIENT version: radial-bin filter (annular bands).

    Expects x: (B, N, C) with N = H*W. Use spatial_size=(H, W).
    Keeps the same flow:
        rfft2 -> (X * W) -> irfft2 -> SSF -> residual + alpha
    """
    def __init__(
        self,
        blocks,
        dim,
        h,
        w,
        init_alpha=0.1,
        per_block_alpha=True,
        r_threshold=0.5,
        num_bins=16,              # <<< compactness knob (e.g., 8/16/32)
        init_std=0.02,
        identity_bias=True,       # W = 1 + delta if True, else W = delta
        bin_mode="linear",        # "linear" or "sqrt" spacing in radius
    ):
        super().__init__()
        self.h, self.w = h, w
        self.num_bins = int(num_bins)
        self.identity_bias = bool(identity_bias)

        # --- NEW: learn only per-(radial bin, channel) weights ---
        # Shape: (blocks, num_bins, dim)  <<<< drastically fewer params than (blocks, h, w, dim)
        self.lora_bin_weight = nn.Parameter(
            torch.randn(blocks, self.num_bins, dim, dtype=torch.float32) * init_std
        )

        # SSF + alpha (unchanged)
        self.lora_ssf_scale, self.lora_ssf_shift = init_ssf_scale_shift(blocks, dim)
        if per_block_alpha:
            self.lora_alpha = nn.Parameter(torch.full((blocks, 1, 1, 1), float(init_alpha)))
        else:
            self.lora_alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        # rFFT grid radius (buffers) + masks (kept for your head)
        yy = torch.linspace(-1, 1, self.h)
        xx = torch.linspace(0, 1, self.w)  # rFFT half-plane
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        r = torch.sqrt(torch.clamp(Y * Y + X * X, 0, 1))  # (H, W2p1)

        low  = (r <= float(r_threshold)).float()
        high = 1.0 - low
        self.register_buffer("mask_low",  low.unsqueeze(0).unsqueeze(0),  persistent=False)  # (1,1,H,W2p1)
        self.register_buffer("mask_high", high.unsqueeze(0).unsqueeze(0), persistent=False)  # (1,1,H,W2p1)

        edge_start = 0.8
        edge = torch.clamp((r - edge_start) / (1.0 - edge_start + 1e-6), 0, 1)
        edge = edge * high
        self.register_buffer("mask_edge", edge.unsqueeze(0).unsqueeze(0), persistent=False)  # (1,1,H,W2p1)

        # --- NEW: precompute radial bin indices for each (u,v) ---
        # bin_idx: (H, W2p1) with values in [0, num_bins-1]
        if bin_mode == "sqrt":
            r_eff = torch.sqrt(r)  # pushes more resolution toward low-frequencies
        else:
            r_eff = r

        # boundaries for bucketize -> num_bins bands
        # boundaries length = num_bins-1, output in [0, num_bins-1]
        boundaries = torch.linspace(0.0, 1.0, self.num_bins + 1)[1:-1]
        bin_idx = torch.bucketize(r_eff.contiguous(), boundaries, right=False).to(torch.long)
        self.register_buffer("bin_idx", bin_idx, persistent=False)

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
        Xf = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # (B,a,w,C), complex

        # Build W(u,v,c) from per-bin weights via lookup
        # delta_bins: (num_bins, C)
        delta_bins = self.lora_bin_weight[block]  # (K,C)
        # W: (a,w,C)
        W = delta_bins[self.bin_idx]              # (a,w,C)

        if self.identity_bias:
            W = 1.0 + W

        Y = torch.fft.irfft2(Xf * W, s=(a, b), dim=(1, 2), norm="ortho")  # (B,a,b,C)

        # SSF + residual
        Y = ssf_ada(Y, self.lora_ssf_scale[block], self.lora_ssf_shift[block])
        alpha = self.lora_alpha[block] if self.lora_alpha.dim() == 4 else self.lora_alpha
        out = res + alpha * Y
        return out.reshape(B, N, C)


class HighResolutionHead(nn.Module):
    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHead, self).__init__()

        backbone_channels = [18, 36, 72, 144]
        self.num_branches = len(backbone_channels)
        last_inp_channels = sum(backbone_channels)

        H_out, W_out = 56, 56  # all branches upsampled to this

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

        self.last_layer = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels * 4,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels * 4, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(last_inp_channels * 4, num_outputs,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, donors=None):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x0  = x[0]
        x1u = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear', align_corners=False)
        x2u = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear', align_corners=False)
        x3u = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear', align_corners=False)
        branches = [x0, x1u, x2u, x3u]  # each (B,C_b,H,W)
        breakpoint()

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

        x0, x1u, x2u, x3u = out_branches
        x = torch.cat([x0, x1u, x2u, x3u], dim=1)
        x = self.last_layer(x)
        return x