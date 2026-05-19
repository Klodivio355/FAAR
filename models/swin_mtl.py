# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details).
# --------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Dict, Optional

def save_semseg_feature_grid(
    x,
    pre_features,
    post_features,
    save_path,
    batch_indices=None,
    save_individual=False,
):
    """
    x:             Tensor [B, 3, H, W]
    pre_features:  dict with key 'semseg' -> Tensor [B, 1, H, W]
    post_features: dict with key 'semseg' -> Tensor [B, 1, H, W]
    save_path:     path for the big grid figure, e.g. "features_semseg.png"
    batch_indices: iterable of batch indices to plot (max 3). If None, uses [0, 1, 2] up to B.
    save_individual: if True, also saves each image / post / pre individually in ./viz3
    """

    assert "human_parts" in pre_features and "human_parts" in post_features, \
        "pre_features and post_features must contain key 'semseg'"

    assert x.dim() == 4 and x.size(1) == 3, "x must be [B, 3, H, W]"
    B, _, H, W = x.shape

    # ---- choose batch indices (max 3) ----
    if batch_indices is None:
        max_rows = min(B, 3)
        batch_indices = list(range(max_rows))
    else:
        batch_indices = list(batch_indices)
        # keep only valid indices
        batch_indices = [b for b in batch_indices if 0 <= b < B]
        assert len(batch_indices) > 0, "No valid batch indices provided"
        if len(batch_indices) > 3:
            batch_indices = batch_indices[:3]  # limit to 3

    num_rows = len(batch_indices)
    num_cols = 3  # image | semseg_post | semseg_pre

    x_cpu = x.detach().cpu()
    semseg_pre = pre_features["human_parts"].detach().cpu()
    semseg_post = post_features["human_parts"].detach().cpu()

    # create directory for individual saves if needed
    if save_individual:
        os.makedirs("./viz6", exist_ok=True)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3 * num_cols, 3 * num_rows),
        squeeze=False,
    )

    for row_idx, b in enumerate(batch_indices):
        # ---- column 0: original image ----
        img = x_cpu[b]
        img = img.clamp(0, 1)
        img_np = img.permute(1, 2, 0).numpy()  # [H, W, 3]

        ax_img = axes[row_idx, 0]
        ax_img.imshow(img_np)
        ax_img.axis("off")
        if row_idx == 0:
            ax_img.set_title("image")

        # optional save
        if save_individual:
            plt.imsave(f"./viz6/img_b{b}.png", img_np)

        # ---- column 1: semseg post ----
        post = semseg_post[b, 0]  # [H, W]
        post_min, post_max = post.min(), post.max()
        post_norm = (post - post_min) / (post_max - post_min + 1e-5)
        post_np = post_norm.numpy()

        ax_post = axes[row_idx, 1]
        ax_post.imshow(post_np, cmap="magma")
        ax_post.axis("off")
        if row_idx == 0:
            ax_post.set_title("human_parts post")

        if save_individual:
            plt.imsave(f"./viz6/human_parts_post_b{b}.png", post_np, cmap="magma")

        # ---- column 2: semseg pre ----
        pre = semseg_pre[b, 0]  # [H, W]
        pre_min, pre_max = pre.min(), pre.max()
        pre_norm = (pre - pre_min) / (pre_max - pre_min + 1e-5)
        pre_np = pre_norm.numpy()

        ax_pre = axes[row_idx, 2]
        ax_pre.imshow(pre_np, cmap="magma")
        ax_pre.axis("off")
        if row_idx == 0:
            ax_pre.set_title("human_parts pre")

        if save_individual:
            plt.imsave(f"./viz6/human_parts_pre_b{b}.png", pre_np, cmap="magma")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def get_head(task, backbone_channels, num_outputs, config=None, multiscale=True):
    """Return the decoder head"""
    head_type = config.MODEL.DECODER_HEAD.get(task, "hrnet")

    if head_type == "hrnet":
        print(
            f"Using hrnet for task {task} with backbone channels {backbone_channels}")
        from models.seg_hrnet import HighResolutionHead

        return HighResolutionHead(backbone_channels, num_outputs)
    elif head_type == "updecoder":
        print(f"Using updecoder for task {task}")
        from models.updecoder import Decoder

        return Decoder(
            backbone_channels,
            num_outputs,
            args=types.SimpleNamespace(
                **{
                    "num_deconv": 3,
                    "num_filters": [32, 32, 32],
                    "deconv_kernels": [2, 2, 2],
                }
            ),
        )
    elif head_type == "segformer":
        print(
            f"Using segformer for task {task} with {config.MODEL.SEGFORMER_CHANNELS} channels"
        )
        from models.segformer import SegFormerHead

        return SegFormerHead(
            in_channels=backbone_channels,
            channels=config.MODEL.SEGFORMER_CHANNELS,
            num_classes=num_outputs,
        )
    else:
        if not multiscale:
            from models.aspp_single import DeepLabHead
        else:
            from models.aspp import ASPP
        print(f"Using ASPP for task {task}")
        return ASPP(backbone_channels, num_outputs)


class DecoderGroup(nn.Module):
    def __init__(self, tasks, num_outputs, channels, out_size, config, multiscale=True):
        super(DecoderGroup, self).__init__()
        self.tasks = tasks
        self.num_outputs = num_outputs
        self.channels = channels
        self.decoders = nn.ModuleDict()
        self.out_size = out_size
        self.multiscale = multiscale
        for task in self.tasks:
            self.decoders[task] = get_head(
                task,
                self.channels,
                self.num_outputs[task],
                config=config,
                multiscale=self.multiscale,
            )

    def forward(self, x):
        """
        x: dict[str, Tensor] mapping each task name to its image/feature representation
        (whatever each decoder expects). Donors will be a list: [other_task1, other_task2].
        """
        result = {}
        post_branch = {}
        pre_branch = {}
        task_order = list(self.tasks)  # fixed order for reproducibility
        for task in task_order:
            # donors as a *list* of other tasks' reps, e.g. [x['depth'], x['normals']] for task='seg'
            donors = [x[t] for t in task_order if t != task]
            #breakpoint()
            y  = self.decoders[task](x[task], donors=donors)  # decoder must accept donors=[...]
            # upsample/output normalize if needed

            """ for i in range(len(pre_cross)):
                pre_cross[i] = F.interpolate(
                    pre_cross[i], size=self.out_size, mode="bilinear", align_corners=False
                )
            for i in range(len(out_branches)):
                out_branches[i] = F.interpolate(
                    out_branches[i], size=self.out_size, mode="bilinear", align_corners=False
                ) """

            y = F.interpolate(y, self.out_size, mode="bilinear", align_corners=False)
            result[task] = y
            #pre_cross_all = torch.cat(pre_cross, dim=1)      # [B, sum_C_pre, H, W]
            #out_branches_all = torch.cat(out_branches, dim=1)  # [B, sum_C_out, H, W]

            # 2) make a single "global" 2D map by reducing over channels
            #pre_cross_global = pre_cross_all.abs().mean(dim=1, keepdim=True)     # [B, 1, H, W]
            #out_branches_global = out_branches_all.abs().mean(dim=1, keepdim=True)
            #pre_branch[task] = pre_cross_global
            #post_branch[task] = out_branches_global
        return result#, pre_branch, post_branch


class Downsampler(nn.Module):
    def __init__(self, dims, channels, input_res, bias=False, enabled=True):
        super(Downsampler, self).__init__()
        self.dims = dims
        self.input_res = input_res
        self.enabled = enabled
        if self.enabled:
            self.downsample_0 = torch.nn.Conv2d(
                dims[0], channels[0], 1, bias=bias)
            self.downsample_1 = torch.nn.Conv2d(
                dims[1], channels[1], 1, bias=bias)
            self.downsample_2 = torch.nn.Conv2d(
                dims[2], channels[2], 1, bias=bias)
            self.downsample_3 = torch.nn.Conv2d(
                dims[3], channels[3], 1, bias=bias)

    def forward(self, x):
        s_3 = (
            x[3]
            .view(-1, self.input_res[3], self.input_res[3], self.dims[3])
            .permute(0, 3, 1, 2)
        )

        s_2 = (
            x[2]
            .view(-1, self.input_res[2], self.input_res[2], self.dims[2])
            .permute(0, 3, 1, 2)
        )
        s_1 = (
            x[1]
            .view(-1, self.input_res[1], self.input_res[1], self.dims[1])
            .permute(0, 3, 1, 2)
        )
        s_0 = (
            x[0]
            .view(-1, self.input_res[0], self.input_res[0], self.dims[0])
            .permute(0, 3, 1, 2)
        )

        if self.enabled:
            return [
                self.downsample_0(s_0),
                self.downsample_1(s_1),
                self.downsample_2(s_2),
                self.downsample_3(s_3),
            ]
        else:
            return [s_0, s_1, s_2, s_3]


class MultiTaskSwin(nn.Module):
    def __init__(self, encoder, config):
        super(MultiTaskSwin, self).__init__()

        self.backbone = encoder
        self.num_outputs = config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT
        self.tasks = config.TASKS
        if hasattr(self.backbone, "patch_embed"):
            patches_resolution = self.backbone.patch_embed.patches_resolution
            self.embed_dim = self.backbone.embed_dim
            num_layers = self.backbone.num_layers
            self.dims = [
                int((self.embed_dim * 2 ** ((i + 1) if i < num_layers - 1 else i)))
                for i in range(num_layers)
            ]
            self.input_res = [
                patches_resolution[0] // (2 **
                                          ((i + 1) if i < num_layers - 1 else i))
                for i in range(num_layers)
            ]
            self.window_size = self.backbone.layers[0].blocks[0].window_size
            self.img_size = self.backbone.patch_embed.img_size
        else:
            self.input_res = [28, 14, 7, 7]

            self.dims = [192, 384, 768, 768]
            self.window_size = config.MODEL.SWIN.WINDOW_SIZE
            self.img_size = config.DATA.IMG_SIZE

        self.channels = (
            config.MODEL.DECODER_CHANNELS
            if config.MODEL.DECODER_DOWNSAMPLER
            else self.dims
        )
        self.mtlora = config.MODEL.MTLORA

        if self.mtlora.ENABLED:
            self.downsampler = nn.ModuleDict(
                {
                    task: Downsampler(
                        dims=self.dims,
                        channels=self.channels,
                        input_res=self.input_res,
                        bias=False,
                    )
                    for task in self.tasks
                }
            )
        else:
            self.downsampler = Downsampler(
                dims=self.dims,
                channels=self.channels,
                input_res=self.input_res,
                bias=False,
            )

        self.per_task_downsampler = config.MODEL.PER_TASK_DOWNSAMPLER
        if self.per_task_downsampler:
            self.downsampler = nn.ModuleDict(
                {
                    task: Downsampler(
                        dims=self.dims,
                        channels=self.channels,
                        input_res=self.input_res,
                        bias=False,
                        enabled=config.MODEL.DECODER_DOWNSAMPLER,
                    )
                    for task in self.tasks
                }
            )
        else:
            self.downsampler = Downsampler(
                dims=self.dims,
                channels=self.channels,
                input_res=self.input_res,
                bias=False,
            )
        self.decoders = DecoderGroup(
            self.tasks,
            self.num_outputs,
            channels=self.channels,
            out_size=self.img_size,
            config=config,
            multiscale=True,
        )

    def forward(self, x, masks=None, training=False):
        
        shared_representation, reg_loss, masks, masks_tasks = self.backbone(x, return_stages=True, masks=None)
        
        if self.mtlora.ENABLED:
            shared_ft = {task: [] for task in self.tasks}
            for _, tasks_shared_rep in shared_representation:
                for task, shared_rep in tasks_shared_rep.items():
                    shared_ft[task].append(shared_rep)
            for task in self.tasks:
                shared_ft[task] = self.downsampler[task](shared_ft[task])
        else:
            if self.per_task_downsampler:
                shared_ft = {
                    task: self.downsampler[task](shared_representation)
                    for task in self.tasks
                }
            else:
                shared_representation = self.downsampler(shared_representation)
                shared_ft = {
                    task: shared_representation for task in self.tasks}
    
        result = self.decoders(shared_ft)
        if x.size(0) > 1:
            """ save_semseg_feature_grid(
                x=x,
                pre_features=pre_feat,
                post_features=post_feat,
                save_path="semseg_features_grid.png",
                batch_indices=[4,7,8],
                save_individual=True,   # also dumps ./viz3/img_b*.png etc.
            ) """
            #breakpoint()
        
        if training:
            return result, reg_loss
        else:
            return result, reg_loss

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = False

    def unfreeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """
        Freeze all backbone params, then unfreeze only those whose names contain 'film'.
        Prints the re-enabled parameter names and a count summary.
        """
        # 1) Freeze everything in the backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2) Re-enable FiLM params within the backbone
        enabled = []
        for name, p in self.backbone.named_parameters():
            if 'film' in name.lower():
                p.requires_grad = True
                enabled.append(f"backbone.{name}")

        # 3) Report
        if enabled:
            print("[freeze_backbone] Re-enabled FiLM parameters:")
            for n in enabled:
                print(f"  - {n}")
        else:
            print("[freeze_backbone] No backbone parameters matched 'film'; backbone remains fully frozen.")

        total = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[freeze_backbone] Trainable (backbone) params: {trainable} / {total}")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
