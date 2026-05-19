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


import os
import torch
import torch.distributed as dist
from torch import inf
import errno

from PIL import Image
import numpy as np
import cv2
import imageio
import scipy.io as sio
import torch.nn.functional as F
from models.lora import map_old_state_dict_weights
from models.lora8 import MTLoRALinear
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt


def enforce_zero_tail_all(model):
    """Zero out rows>k and cols>k for each LoRA adapter and mask their grads so they stay zero."""
    for _, m in model.named_modules():
        if not isinstance(m, MTLoRALinear):
            continue

        # remove old hooks if we re-enforce
        if hasattr(m, "_mask_hooks"):
            for h in m._mask_hooks:
                h.remove()
        m._mask_hooks = []

        # Shared adapter
        if hasattr(m, "lora_shared_A"):
            R = m.lora_shared_A.shape[0]
            k = min(getattr(m, "R_active_shared", R), R)
            if k < R:
                m.lora_shared_A.data[k:, :].zero_()
                m.lora_shared_B.data[:, k:].zero_()

                def hook_A(grad, kk=k):
                    grad = grad.clone()
                    grad[kk:, :] = 0
                    return grad
                def hook_B(grad, kk=k):
                    grad = grad.clone()
                    grad[:, kk:] = 0
                    return grad
                m._mask_hooks += [
                    m.lora_shared_A.register_hook(hook_A),
                    m.lora_shared_B.register_hook(hook_B),
                ]

        # Per-task adapters
        if hasattr(m, "lora_tasks_A"):
            for t, A in m.lora_tasks_A.items():
                Rt = A.shape[0]
                k_t = min(getattr(m, f"R_active_task__{t}", Rt), Rt)
                if k_t < Rt:
                    m.lora_tasks_A[t].data[k_t:, :].zero_()
                    m.lora_tasks_B[t].data[:, k_t:].zero_()

                    def hook_A_t(grad, kk=k_t):
                        grad = grad.clone()
                        grad[kk:, :] = 0
                        return grad
                    def hook_B_t(grad, kk=k_t):
                        grad = grad.clone()
                        grad[:, kk:] = 0
                        return grad
                    m._mask_hooks += [
                        m.lora_tasks_A[t].register_hook(hook_A_t),
                        m.lora_tasks_B[t].register_hook(hook_B_t),
                    ]


def show_global_class_low_high_by_indices(
    class_maps,                         # torch.Tensor [B, 21, H, W] (logits or probabilities)
    indices=(0,1,2,3),                  # columns = len(indices)
    rgb_batch=None,                     # optional torch.Tensor [B,3,H,W] for top-row (natural images)
    save_path="global_class_low_high.png",
    # how to interpret/aggregate class maps
    from_logits=True,                   # if True -> softmax over classes before combining
    combine="union",                    # "union" | "max" | "sum" | "lse"
    tau=0.25,                           # temperature for "lse"
    # display (top row)
    overlay=False,                      # if True, overlay global map on RGB
    overlay_alpha=0.45,
    denorm="auto",                      # for rgb_batch: "auto" | "imagenet" | "none" | "custom"
    mean=None, std=None,
    # frequency bands (0..Nyquist = 0.5)
    low_band=(0.00, 0.06),
    high_band=(0.25, 0.50),
    # binary map tuning (from soft envelope)
    low_thresh_pct=95.0,
    high_thresh_pct=87.0,
    blur_sigma_rel=0.01,
    gamma=0.7,
    return_maps=False                   # if True -> returns dict with global/low/high maps
):
    """
    Builds a 3-row figure (Original, Low, High) using ONE global per-image map
    aggregated across ALL classes from class_maps [B,21,H,W]. No Grad-CAM here.
    """
    # ---------- helpers ----------
    def _to_np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _pnorm(x, lo=1, hi=99):
        a, b = np.percentile(x, [lo, hi])
        if b - a < 1e-9: return np.zeros_like(x)
        return np.clip((x - a) / (b - a), 0, 1)

    def _hann2d(H, W):
        hy = np.hanning(H)[:, None]; hx = np.hanning(W)[None, :]
        return hy * hx

    def _gauss_blur(gray, rel_sigma):
        if rel_sigma <= 0: return gray
        sigma = max(0.5, rel_sigma * min(gray.shape))
        k = int(np.ceil(6*sigma)) | 1
        x = np.arange(k) - k//2
        ker = np.exp(-(x**2)/(2*sigma**2)); ker /= ker.sum()
        tmp = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 1, gray)
        out = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 0, tmp)
        return out

    def _majority3(bw):
        ker = np.ones((3,3), dtype=np.float32)
        from numpy.lib.stride_tricks import sliding_window_view as swv
        pad = np.pad(bw, 1, mode='constant')
        win = swv(pad, (3,3))
        cnt = (win * ker).sum(axis=(-1,-2))
        return (cnt >= 5).astype(np.float32)

    def _denorm_rgb(BCHW):
        arr = _to_np(BCHW)                   # [B,3,H,W]
        arr = np.moveaxis(arr, 1, -1)        # [B,H,W,3]
        if arr.max() > 1.5 and arr.dtype.kind != 'f': arr = arr / 255.0
        looks_01 = (arr.min() >= 0.0) and (arr.max() <= 1.0)
        if denorm == "none":
            pass
        elif denorm == "imagenet":
            m = np.array([0.485,0.456,0.406]).reshape(1,1,1,3)
            s = np.array([0.229,0.224,0.225]).reshape(1,1,1,3)
            arr = arr * s + m
        elif denorm == "custom":
            if mean is None or std is None:
                raise ValueError("Provide mean/std for denorm='custom'.")
            m = np.array(mean).reshape(1,1,1,3)
            s = np.array(std).reshape(1,1,1,3)
            arr = arr * s + m
        else:  # "auto"
            if (arr.min() < -0.2) or (arr.max() > 1.2):
                if arr.min() >= -4 and arr.max() <= 4:
                    m = np.array([0.485,0.456,0.406]).reshape(1,1,1,3)
                    s = np.array([0.229,0.224,0.225]).reshape(1,1,1,3)
                    arr = arr * s + m
                else:
                    for c in range(3): arr[..., c] = _pnorm(arr[..., c], 1, 99)
                    return np.clip(arr, 0, 1)
        arr = np.clip(arr, 0, 1)
        if not looks_01:
            for c in range(3): arr[..., c] = _pnorm(arr[..., c], 1, 99)
        return np.clip(arr, 0, 1)

    def _combine_probs(P):  # P: [K,H,W] in [0,1]
        if combine == "max":
            agg = np.max(P, axis=0)
        elif combine == "sum":
            agg = np.sum(P, axis=0)
            agg = _pnorm(agg, 1, 99)
        elif combine == "lse":
            m = np.max(P, axis=0, keepdims=True)
            agg = tau * (np.log(np.exp((P - m)/max(tau,1e-6)).sum(axis=0) + 1e-12) + (m[0]/max(tau,1e-6)))
            agg = _pnorm(agg, 1, 99)
        else:  # "union" (default): 1 - Π_k (1 - p_k)
            agg = 1.0 - np.prod(1.0 - np.clip(P, 0.0, 1.0), axis=0)
        return np.clip(agg, 0, 1)

    # ---------- inputs ----------
    import torch
    assert isinstance(class_maps, torch.Tensor) and class_maps.ndim == 4, \
        "class_maps must be torch.Tensor [B,21,H,W]"
    B, K, H, W = class_maps.shape
    assert K == 21, "Expected 21 classes (adjust if different)."

    # probs over classes
    if from_logits:
        P = torch.softmax(class_maps, dim=1)        # [B,21,H,W]
    else:
        P = torch.clamp(class_maps, 0, 1)           # assume already probabilities

    P_np = _to_np(P)                                 # [B,21,H,W]

    RGB = None
    if rgb_batch is not None:
        assert isinstance(rgb_batch, torch.Tensor) and rgb_batch.ndim == 4 \
               and rgb_batch.shape[:1] == (B,) and rgb_batch.shape[2:] == (H, W), \
               "rgb_batch must be [B,3,H,W] matching B,H,W of class_maps"
        RGB = _denorm_rgb(rgb_batch)                 # [B,H,W,3]

    # frequency grid + masks
    hann = _hann2d(H, W)
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    fy_s = np.fft.fftshift(fy, axes=0); fx_s = np.fft.fftshift(fx, axes=1)
    R = np.sqrt(fx_s**2 + fy_s**2)

    lb0, lb1 = max(0.0, low_band[0]), min(0.5, low_band[1])
    hb0, hb1 = max(0.0, high_band[0]), 0.5
    low_mask  = (R >= lb0) & (R <  lb1)
    high_mask = (R >= hb0) & (R <= hb1 + 1e-12)

    # ---------- figure ----------
    ncols = len(indices)
    fig, axes = plt.subplots(3, ncols, figsize=(3.2*ncols, 9.2))
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    out_maps = []  # per column: dict with 'global','low_env','high_env'

    for ci, bidx in enumerate(indices):
        # global per-image map from all classes
        global_map = _combine_probs(P_np[bidx])           # [H,W] in [0,1]

        # FFT envelopes (on global map)
        F  = np.fft.fft2(global_map * hann)
        Fh = np.fft.fftshift(F)

        Fl = Fh * low_mask
        el = np.abs(np.fft.ifft2(np.fft.ifftshift(Fl)))
        el = _gauss_blur(el, blur_sigma_rel)
        el = (_pnorm(el, 1, 99)) ** gamma
        thr_l = np.percentile(el, float(low_thresh_pct))
        bin_l = _majority3((el >= max(thr_l, 1e-8)).astype(np.float32))

        Fh_ = Fh * high_mask
        eh = np.abs(np.fft.ifft2(np.fft.ifftshift(Fh_)))
        eh = _gauss_blur(eh, blur_sigma_rel)
        eh = (_pnorm(eh, 1, 99)) ** gamma
        thr_h = np.percentile(eh, float(high_thresh_pct))
        bin_h = _majority3((eh >= max(thr_h, 1e-8)).astype(np.float32))

        out_maps.append({"global": global_map.astype(np.float32),
                         "low_env": el.astype(np.float32),
                         "high_env": eh.astype(np.float32)})

        # ----- Row 1: original images (no overlay by default) -----
        ax = axes[0, ci]
        if RGB is not None:
            ax.imshow(RGB[bidx], interpolation="nearest")
            if overlay:
                ax.imshow(global_map, cmap="magma", alpha=overlay_alpha, interpolation="nearest")
            ax.set_title(f"idx={bidx}", fontsize=11)
        else:
            # fallback: show global map heatmap if no RGB available
            ax.imshow(global_map, cmap="magma", interpolation="nearest")
            ax.set_title(f"Global (idx={bidx})", fontsize=11)
        ax.axis("off")

        # ----- Row 2: Low (binary) -----
        ax = axes[1, ci]
        ax.imshow(bin_l, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_facecolor("black")
        if ci == 0: ax.set_ylabel("Low", fontsize=11)
        ax.axis("off")

        # ----- Row 3: High (binary) -----
        ax = axes[2, ci]
        ax.imshow(bin_h, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_facecolor("black")
        if ci == 0: ax.set_ylabel("High", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)

    if return_maps:
        return {"indices": list(indices),
                "bands": {"low": (lb0, lb1), "high": (hb0, hb1)},
                "maps": out_maps}

def update_lora_scores(model):
    # Call after .backward(); reads current grads and updates EMA per-slot scores
    for _, m in model.named_modules():
        if isinstance(m, MTLoRALinear):
            m.update_rank_scores_from_grads()

""" def print_current_caps(model, tol: float = 1e-12, show_tasks: bool = True):
    for name, m in model.named_modules():
        if isinstance(m, MTLoRALinear) and hasattr(m, "lora_shared_A"):
            R = m.lora_shared_A.shape[0]
            k_cap = getattr(m, "R_active_shared", R)
            nnz_rows = int((m.lora_shared_A.abs().sum(dim=1) > tol).sum().item())
            print(f"{name:60s}  cap={k_cap:2d}/{R:2d}  nnz_rows(A)={nnz_rows:2d}")
            if show_tasks and hasattr(m, "lora_tasks_A"):
                for t, A in m.lora_tasks_A.items():
                    Rt = A.shape[0]
                    k_cap_t = getattr(m, f"R_active_task__{t}", Rt)
                    nnz_rows_t = int((A.abs().sum(dim=1) > tol).sum().item())
                    print(f"  └─ task={t:12s} cap={k_cap_t:2d}/{Rt:2d}  nnz_rows(A)={nnz_rows_t:2d} ") """

""" def print_current_caps(model, tol: float = 1e-12, show_tasks: bool = True, suggest: bool = False):
    import torch

    for name, m in model.named_modules():
        if isinstance(m, MTLoRALinear) and hasattr(m, "lora_shared_A"):
            R = m.lora_shared_A.shape[0]
            k_cap = getattr(m, "R_active_shared", R)
            nnz_rows = int((m.lora_shared_A.abs().sum(dim=1) > tol).sum().item())
            line = f"{name:60s}  cap={k_cap:2d}/{R:2d}  nnz_rows(A)={nnz_rows:2d}"
            if suggest and hasattr(m, "ema_score_shared"):
                se = m.ema_score_shared.float()
                if se.numel() > 0 and se.sum() > 0:
                    order = torch.argsort(se, descending=True)
                    ce = se[order].cumsum(0) / se.sum().clamp_min(1e-12)
                    hit = (ce >= m.score_rho_shared).nonzero(as_tuple=True)[0]
                    K = int(hit[0].item() + 1) if hit.numel() else se.numel()
                    line += f"  suggested_K≈{K}"
            print(line)

            if show_tasks and hasattr(m, "lora_tasks_A"):
                for t, A in m.lora_tasks_A.items():
                    Rt = A.shape[0]
                    k_cap_t = getattr(m, f"R_active_task__{t}", Rt)
                    nnz_rows_t = int((A.abs().sum(dim=1) > tol).sum().item())
                    t_str = str(t)
                    t_line = f"  └─ task={t_str:12s} cap={k_cap_t:2d}/{Rt:2d}  nnz_rows(A)={nnz_rows_t:2d}"
                    if suggest and hasattr(m, f"ema_score_task__{t}"):
                        se_t = getattr(m, f"ema_score_task__{t}").float()
                        if se_t.numel() > 0 and se_t.sum() > 0:
                            import math as _math
                            order_t = torch.argsort(se_t, descending=True)
                            ce_t = se_t[order_t].cumsum(0) / se_t.sum().clamp_min(1e-12)
                            rho_t = getattr(m, f"score_rho_task__{t}")
                            hit_t = (ce_t >= rho_t).nonzero(as_tuple=True)[0]
                            Kt = int(hit_t[0].item() + 1) if hit_t.numel() else se_t.numel()
                            t_line += f"  suggested_K≈{Kt}"
                    print(t_line) """


def print_current_caps(model, tol: float = 1e-12, show_tasks: bool = True, suggest: bool = False):
    import torch

    total_erased_params = 0  # cumulative across all layers & tasks

    for name, m in model.named_modules():
        if isinstance(m, MTLoRALinear) and hasattr(m, "lora_shared_A"):
            A = m.lora_shared_A
            B = getattr(m, "lora_shared_B", None)  # may be None depending on your impl

            R = A.shape[0]
            k_cap = getattr(m, "R_active_shared", R)

            # how many rank-1 updates are erased for this shared LoRA
            erased_ranks = max(0, R - k_cap)

            # one rank-1 update = one row of A + one column of B
            params_per_rank = A.shape[1]
            if B is not None:
                params_per_rank += B.shape[0]

            erased_params_shared = erased_ranks * params_per_rank
            total_erased_params += erased_params_shared

            # your original row sparsity info
            nnz_rows = int((A.abs().sum(dim=1) > tol).sum().item())
            line = f"{name:60s}  cap={k_cap:2d}/{R:2d}  nnz_rows(A)={nnz_rows:2d}"
            if erased_params_shared > 0:
                line += f"  erased_params={erased_params_shared}"

            if suggest and hasattr(m, "ema_score_shared"):
                se = m.ema_score_shared.float()
                if se.numel() > 0 and se.sum() > 0:
                    order = torch.argsort(se, descending=True)
                    ce = se[order].cumsum(0) / se.sum().clamp_min(1e-12)
                    hit = (ce >= m.score_rho_shared).nonzero(as_tuple=True)[0]
                    K = int(hit[0].item() + 1) if hit.numel() else se.numel()
                    line += f"  suggested_K≈{K}"
            print(line)

            # per-task LoRAs
            if show_tasks and hasattr(m, "lora_tasks_A"):
                B_tasks = getattr(m, "lora_tasks_B", None)

                for t, A_t in m.lora_tasks_A.items():
                    Rt = A_t.shape[0]
                    k_cap_t = getattr(m, f"R_active_task__{t}", Rt)

                    # erased rank-1 updates for this task
                    erased_ranks_t = max(0, Rt - k_cap_t)

                    params_per_rank_t = A_t.shape[1]
                    B_t = None
                    if B_tasks is not None:
                        try:
                            B_t = B_tasks[t]  # works for dict / ParameterDict / ModuleDict style
                        except Exception:
                            B_t = None
                    if B_t is not None:
                        params_per_rank_t += B_t.shape[0]

                    erased_params_t = erased_ranks_t * params_per_rank_t
                    total_erased_params += erased_params_t

                    nnz_rows_t = int((A_t.abs().sum(dim=1) > tol).sum().item())
                    t_str = str(t)
                    t_line = (
                        f"  └─ task={t_str:12s} cap={k_cap_t:2d}/{Rt:2d}  nnz_rows(A)={nnz_rows_t:2d}"
                    )
                    if erased_params_t > 0:
                        t_line += f"  erased_params={erased_params_t}"

                    if suggest and hasattr(m, f"ema_score_task__{t}"):
                        se_t = getattr(m, f"ema_score_task__{t}").float()
                        if se_t.numel() > 0 and se_t.sum() > 0:
                            import math as _math
                            order_t = torch.argsort(se_t, descending=True)
                            ce_t = se_t[order_t].cumsum(0) / se_t.sum().clamp_min(1e-12)
                            rho_t = getattr(m, f"score_rho_task__{t}")
                            hit_t = (ce_t >= rho_t).nonzero(as_tuple=True)[0]
                            Kt = int(hit_t[0].item() + 1) if hit_t.numel() else se_t.numel()
                            t_line += f"  suggested_K≈{Kt}"
                    print(t_line)

    # total number of parameters that have been erased from the objective
    return total_erased_params


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def set_inferred_ranks_from_caps(model):
    """Use current training caps as fixed inference ranks (shared + per-task)."""
    for _, m in model.named_modules():
        if isinstance(m, MTLoRALinear):
            if hasattr(m, "R_active_shared"):
                m.infer_rank_shared = int(m.R_active_shared)
            if hasattr(m, "infer_rank_tasks"):
                for t in m.infer_rank_tasks.keys():
                    cap = getattr(m, f"R_active_task__{t}", None)
                    if cap is not None:
                        m.infer_rank_tasks[t] = int(cap)

def clear_inferred_ranks(model):
    """Unset fixed ranks so training goes back to sampling within caps."""
    for _, m in model.named_modules():
        if isinstance(m, MTLoRALinear):
            m.infer_rank_shared = None
            if hasattr(m, "infer_rank_tasks"):
                for t in m.infer_rank_tasks.keys():
                    m.infer_rank_tasks[t] = None

def shrink_all_caps(model, margin=1, floor=1):
    with torch.no_grad():
        for _, m in model.named_modules():
            if isinstance(m, MTLoRALinear):
                m.shrink_active_cap(margin=margin, floor=floor)

def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger, backbone=False, quiet=False):
    resume_path = config.MODEL.RESUME if not backbone else config.MODEL.RESUME_BACKBONE
    logger.info(
        f"==============> Resuming form {resume_path}....................")
    if resume_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(resume_path, map_location='cpu')

    mtlora = config.MODEL.MTLORA
    mtlora_enabled = mtlora.ENABLED

    skip_decoder = config.TRAIN.SKIP_DECODER_CKPT

    model_state = {k: v for k, v in checkpoint["model"].items(
    ) if not k.startswith("decoders")} if skip_decoder else checkpoint["model"]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in model_state.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del model_state[k]

    if config.MODEL.UPDATE_RELATIVE_POSITION:
        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in model_state.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del model_state[k]

        # delete relative_coords_table since we always re-init it
        relative_position_index_keys = [
            k for k in model_state.keys() if "relative_coords_table" in k]
        for k in relative_position_index_keys:
            del model_state[k]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in model_state.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = model_state[k]
            relative_position_bias_table_current = model.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    # bicubic interpolate relative_position_bias_table if not match
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                        mode='bicubic')
                    model_state[k] = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)

        # bicubic interpolate absolute_pos_embed if not match
        absolute_pos_embed_keys = [
            k for k in model_state.keys() if "absolute_pos_embed" in k]
        for k in absolute_pos_embed_keys:
            # dpe
            absolute_pos_embed_pretrained = model_state[k]
            absolute_pos_embed_current = model.model_state()[k]
            _, L1, C1 = absolute_pos_embed_pretrained.size()
            _, L2, C2 = absolute_pos_embed_current.size()
            if C1 != C1:
                logger.warning(f"Error in loading {k}, passing......")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                        -1, S1, S1, C1)
                    absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                        0, 3, 1, 2)
                    absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                        0, 2, 3, 1)
                    absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                        1, 2)
                    model_state[k] = absolute_pos_embed_pretrained_resized

    if mtlora_enabled:
        mapping = {}
        trainable_layers = []
        if mtlora.QKV_ENABLED:
            trainable_layers.extend(["attn.qkv.weight", "attn.qkv.bias"])
        if mtlora.PROJ_ENABLED:
            trainable_layers.extend(["attn.proj.weight", "attn.proj.bias"])
        if mtlora.FC1_ENABLED:
            trainable_layers.extend(["mlp.fc1.weight", "mlp.fc1.bias"])
        if mtlora.FC2_ENABLED:
            trainable_layers.extend(["mlp.fc2.weight", "mlp.fc2.bias"])
        if mtlora.DOWNSAMPLER_ENABLED:
            trainable_layers.extend(["downsample.reduction.weight"])

        for k, v in model_state.items():
            last_three = ".".join(k.split(".")[-3:])
            prefix = ".".join(k.split(".")[:-3])
            if last_three in trainable_layers:
                weight_bias = last_three.split(".")[-1]
                layer_name = ".".join(last_three.split(".")[:-1])
                mapping[f"{prefix}.{layer_name}.{weight_bias}"] = f"{prefix}.{layer_name}.linear.{weight_bias}"
        if not len(mapping):
            print("No keys needs to be mapped for LoRA")
        model_state = map_old_state_dict_weights(
            model_state, mapping, "", config.MODEL.MTLORA.SPLIT_QKV)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if not quiet:
        if len(missing) > 0:
            logger.warning("=============Missing Keys==============")
            for k in missing:
                logger.warning(k)
        if len(unexpected) > 0:
            logger.warning("=============Unexpected Keys==============")
            for k in unexpected:
                logger.warning(k)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and not skip_decoder:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(
            f"=> loaded successfully '{resume_path}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(
        f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                    0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                    1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(
                f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_name = f'ckpt_epoch_{epoch}.pth'
    save_path = os.path.join(config.OUTPUT, save_name)
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    return save_path


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d)
                                for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, model=None, loss_dict=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            
            update_lora_scores(model)  
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def tens2image(tens, transpose=False):
    """Converts tensor with 2 or 3 dimensions to numpy array"""
    im = tens.cpu().detach().numpy()

    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    elif im.shape[-1] == 1:
        im = np.squeeze(im)
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    if transpose:
        if im.ndim == 3:
            im = im.transpose((1, 2, 0))
    return im


def normalize(arr, t_min=0, t_max=255):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()
    for i in arr:
        temp = (((i - arr.min())*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    res = np.array(norm_arr)
    return res


def save_imgs_mtl(batch_imgs, batch_labels, batch_predictions, path, id):
    import torchvision

    imgs = tens2image(batch_imgs, transpose=True)
    labels = {task: tens2image(label, transpose=True)
              for task, label in batch_labels.items()}
    predictions = {task: tens2image(prediction)
                   for task, prediction in batch_predictions.items()}

    Image.fromarray(normalize(imgs, 0, 255).astype(
        np.uint8)).save(f'{path}/{id}_img.png')

    for task in labels.keys():
        if task == "semseg":
            print(np.sum(labels[task] != 255))
            labels[task] = labels[task] != 255
            predictions[task] = predictions[task] != 225
            batch_imgs = 255*(batch_imgs-torch.min(batch_imgs)) / \
                (torch.max(batch_imgs)-torch.min(batch_imgs))
            semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8),
                                                               batch_predictions[task][0].to(torch.bool), colors="blue", alpha=0.5)
            Image.fromarray(semseg.numpy().transpose((1, 2, 0))
                            ).save(f'{path}/{id}_{task}_pred.png')
            semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8),
                                                               batch_labels[task][0].to(torch.bool), colors="blue", alpha=0.5)
            Image.fromarray(semseg.numpy().transpose((1, 2, 0))
                            ).save(f'{path}/{id}_{task}_gt.png')
        else:
            labels[task] = normalize(labels[task], 0, 255)
            predictions[task] = normalize(predictions[task], 0, 255)

            Image.fromarray(labels[task].astype(np.uint8)).save(
                f'{path}/{id}_{task}_gt.png')
            Image.fromarray(predictions[task].astype(np.uint8)).save(
                f'{path}/{id}_{task}_pred.png')

def _to_numpy(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _infer_hw_from_tokens(n_tokens: int, expected: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    if expected is not None:
        return expected
    root = int(round(n_tokens ** 0.5))
    if root * root == n_tokens:
        return (root, root)
    for h in range(int(np.sqrt(n_tokens)), 0, -1):
        if n_tokens % h == 0:
            return (h, n_tokens // h)
    return (n_tokens, 1)


def _ensure_HW_C(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], int]:
    a = _to_numpy(arr)
    if a.ndim == 2:
        N, C = a.shape
        H, W = _infer_hw_from_tokens(N)
        a = a.reshape(H, W, C)
        return a, (H, W), C
    if a.ndim == 3:
        if a.shape[0] in (1, 3):
            C, H, W = a.shape
            a = np.moveaxis(a, 0, -1)
            return a, (H, W), C
        else:
            H, W, C = a.shape
            return a, (H, W), C
    if a.ndim == 4:
        b0 = a[0]
        return _ensure_HW_C(b0)
    raise ValueError(f"Unsupported feature shape: {a.shape}")


def _minmax01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    xmin = x.min()
    xmax = x.max()
    if xmax - xmin < eps:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin + eps)


def _pca_reduce_to_3(hw_c: np.ndarray) -> np.ndarray:
    H, W, C = hw_c.shape
    X = hw_c.reshape(-1, C).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = U[:, :3] * S[:3]
    img3 = comps.reshape(H, W, 3)
    for i in range(3):
        img3[..., i] = _minmax01(img3[..., i])
    return img3


def feature_to_image(
    feat: Union[np.ndarray, "torch.Tensor"],
    hw_hint: Optional[Tuple[int, int]] = None,
    reduce_to: str = "auto",
) -> np.ndarray:
    hw_c, (H, W), C = _ensure_HW_C(feat)
    if C == 3 and reduce_to in ("auto", "rgb"):
        img = _minmax01(hw_c)
        return img
    if C == 1:
        return _minmax01(hw_c[..., 0])
    if reduce_to == "l2":
        img = np.linalg.norm(hw_c, axis=-1)
        return _minmax01(img)
    if reduce_to == "mean":
        img = hw_c.mean(axis=-1)
        return _minmax01(img)
    if reduce_to in ("pca3", "auto"):
        try:
            return _pca_reduce_to_3(hw_c)
        except Exception:
            img = np.linalg.norm(hw_c, axis=-1)
            return _minmax01(img)
    raise ValueError(f"Unknown reduce_to: {reduce_to}")


def _draw_window_grid(ax, H: int, W: int, window_size: Optional[int]):
    if window_size is None or window_size <= 0:
        return
    for x in range(0, W + 1, window_size):
        ax.vlines(x - 0.5, -0.5, H - 0.5)
    for y in range(0, H + 1, window_size):
        ax.hlines(y - 0.5, -0.5, W - 0.5)


def _save_single_image(img: np.ndarray, title: str, out_path: str, window_size: Optional[int] = None):
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, interpolation="nearest")
    else:
        plt.imshow(img, interpolation="nearest")
    plt.title(title)
    ax = plt.gca()
    H, W = img.shape[:2]
    _draw_window_grid(ax, H, W, window_size)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _resize_nn(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    in_h, in_w = img.shape[:2]
    out_h, out_w = out_hw
    ys = (np.arange(out_h) * (in_h / out_h)).astype(int)
    xs = (np.arange(out_w) * (in_w / out_w)).astype(int)
    ys = np.clip(ys, 0, in_h - 1)
    xs = np.clip(xs, 0, in_w - 1)
    if img.ndim == 2:
        return img[ys[:, None], xs[None, :]]
    else:
        return img[ys[:, None], xs[None, :], :]


def save_stage_images(
    stage_features: Dict[str, Union[np.ndarray, "torch.Tensor"]],
    stage_shapes: Dict[str, Tuple[int, int]],
    outdir: str = "swin_stage_viz",
    original_img_path: Optional[str] = None,
    reduce_to: str = "auto",
    window_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    os.makedirs(outdir, exist_ok=True)
    results: Dict[str, np.ndarray] = {}

    orig_img = None
    if original_img_path is not None and os.path.exists(original_img_path):
        try:
            import imageio.v2 as imageio
            orig_img = imageio.imread(original_img_path)
            if orig_img.dtype != np.float32 and orig_img.dtype != np.float64:
                orig_img = orig_img.astype(np.float32) / 255.0
        except Exception:
            orig_img = None

    for stage_name, feat in stage_features.items():
        hw = stage_shapes.get(stage_name, None)
        img = feature_to_image(feat, hw_hint=hw, reduce_to=reduce_to)
        results[stage_name] = img

        fig_path = os.path.join(outdir, f"{stage_name}.png")
        title = f"{stage_name} | shape {img.shape}"
        _save_single_image(img, title, fig_path, window_size=window_size)

        if orig_img is not None:
            H0, W0 = orig_img.shape[:2]
            if img.ndim == 3:
                gray = img.mean(axis=-1)
            else:
                gray = img
            gray_up = _resize_nn(gray, (H0, W0))
            plt.figure()
            plt.imshow(orig_img, interpolation="nearest")
            plt.imshow(gray_up, interpolation="nearest", alpha=0.5)
            plt.title(f"{stage_name} overlay")
            plt.axis("off")
            plt.tight_layout()
            overlay_path = os.path.join(outdir, f"{stage_name}_overlay.png")
            plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
            plt.close()

    return results



def show_low_high_from_images_by_indices(
    image_batch,                      # torch.Tensor [B,3,H,W]
    indices=(0,1,2,3),               # columns to show
    save_path="img_low_high_grid.png",

    # frequency bands in cycles/pixel (Nyquist = 0.5)
    # small fmin avoids DC bias; 1/64 targets ~64px blobs at the coarse end
    low_band=(0.002, 1/64),
    high_band=(0.25, 0.50),

    # thresholding (simple binarization)
    thr_type="percentile",           # "percentile" | "absolute"
    low_thr=95.0,                    # if percentile: e.g. 95; if absolute: in [0,1]
    high_thr=87.0,                   # if percentile: e.g. 87; if absolute: in [0,1]

    # map to threshold
    use_dominance=True,              # True: threshold Hl/(Hl+Hh) and Hh/(Hl+Hh); False: threshold raw envelopes
    normalize_env=True,              # only used if use_dominance=False (min–max to [0,1])

    # input handling
    denorm="auto",                   # "auto" | "imagenet" | "custom" | "none"
    mean=None, std=None,             # for denorm="custom"
):
    """
    3-row grid (Original | Low | High) with simple thresholded binary maps.
    Corrections:
      • Hann window before FFT
      • Low band starts >0 to avoid DC
      • Default thresholds apply to dominance maps for clearer coarse regions
    """
    # ---- helpers ----
    def _to_np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _ensure_BHWC_from_BCHW(bchw):
        arr = _to_np(bchw)                   # [B,3,H,W]
        assert arr.ndim == 4 and arr.shape[1] in (1,3), "image_batch must be [B,3,H,W]"
        arr = np.moveaxis(arr, 1, -1)        # -> [B,H,W,3]
        if arr.shape[-1] == 1: arr = np.repeat(arr, 3, axis=-1)
        return arr.astype(np.float64)

    def _maybe_denorm(BHWC):
        out = BHWC.copy()
        if out.max() > 1.5 and out.dtype.kind != 'f':
            out = out / 255.0
        looks_01 = (out.min() >= 0.0) and (out.max() <= 1.0)
        if denorm == "none":
            pass
        elif denorm == "imagenet":
            m = np.array([0.485,0.456,0.406]).reshape(1,1,1,3)
            s = np.array([0.229,0.224,0.225]).reshape(1,1,1,3)
            out = out * s + m
        elif denorm == "custom":
            if mean is None or std is None:
                raise ValueError("Provide mean/std for denorm='custom'.")
            m = np.array(mean).reshape(1,1,1,3)
            s = np.array(std).reshape(1,1,1,3)
            out = out * s + m
        else:
            if (out.min() < -0.2) or (out.max() > 1.2):
                m = np.array([0.485,0.456,0.406]).reshape(1,1,1,3)
                s = np.array([0.229,0.224,0.225]).reshape(1,1,1,3)
                out = out * s + m
        out = np.clip(out, 0, 1)
        if not looks_01:
            # mild per-channel normalization for display
            a = np.percentile(out, 1, axis=(1,2), keepdims=True)
            b = np.percentile(out, 99, axis=(1,2), keepdims=True)
            out = np.clip((out - a) / (b - a + 1e-9), 0, 1)
        return out

    def _luma(rgb):
        return 0.2989*rgb[...,0] + 0.5870*rgb[...,1] + 0.1140*rgb[...,2]

    # ---- prep ----
    import torch
    assert isinstance(image_batch, torch.Tensor) and image_batch.ndim == 4, \
        "image_batch must be torch.Tensor [B,3,H,W]"
    B, C, H, W = image_batch.shape

    BHWC = _ensure_BHWC_from_BCHW(image_batch)  # [B,H,W,3]
    BHWC = _maybe_denorm(BHWC)

    # frequency grid + masks
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    fy_s = np.fft.fftshift(fy, axes=0); fx_s = np.fft.fftshift(fx, axes=1)
    R = np.sqrt(fx_s**2 + fy_s**2)

    lb0, lb1 = max(0.0, low_band[0]),  min(0.5, low_band[1])
    hb0, hb1 = max(0.0, high_band[0]), min(0.5, high_band[1])
    low_mask  = (R >= lb0) & (R <  lb1)
    high_mask = (R >= hb0) & (R <= hb1 + 1e-12)

    # Hann window to reduce edge leakage
    hann = np.hanning(H)[:,None] * np.hanning(W)[None,:]

    # ---- figure ----
    ncols = len(indices)
    fig, axes = plt.subplots(3, ncols, figsize=(3.1*ncols, 9.0))
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    for ci, bidx in enumerate(indices):
        rgb  = BHWC[bidx]
        gray = _luma(rgb)

        # FFT (with taper)
        F  = np.fft.fft2(gray * hann)
        Fh = np.fft.fftshift(F)

        # band envelopes
        Fl = Fh * low_mask
        Hl = np.abs(np.fft.ifft2(np.fft.ifftshift(Fl)))  # low env

        Fh_ = Fh * high_mask
        Hh = np.abs(np.fft.ifft2(np.fft.ifftshift(Fh_))) # high env

        # choose maps to threshold
        if use_dominance:
            denom = (Hl + Hh + 1e-8)
            low_map  = Hl / denom
            high_map = Hh / denom
        else:
            if normalize_env:
                # min–max to [0,1] per image for each band
                Hl = (Hl - Hl.min()) / (Hl.max() - Hl.min() + 1e-9)
                Hh = (Hh - Hh.min()) / (Hh.max() - Hh.min() + 1e-9)
            low_map, high_map = Hl, Hh

        # simple thresholds -> binary
        if thr_type == "percentile":
            thr_l = np.percentile(low_map,  float(low_thr))
            thr_h = np.percentile(high_map, float(high_thr))
        elif thr_type == "absolute":
            thr_l = float(low_thr)
            thr_h = float(high_thr)
        else:
            raise ValueError("thr_type must be 'percentile' or 'absolute'.")

        bin_l = (low_map  >= thr_l).astype(np.float32)
        bin_h = (high_map >= thr_h).astype(np.float32)

        # Row 1: Original
        ax = axes[0, ci]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"idx={bidx}", fontsize=11)
        ax.axis("off")

        # Row 2: Low (binary)
        ax = axes[1, ci]
        ax.imshow(bin_l, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_facecolor("black")
        if ci == 0: ax.set_ylabel("Low", fontsize=11)
        ax.axis("off")

        # Row 3: High (binary)
        ax = axes[2, ci]
        ax.imshow(bin_h, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_facecolor("black")
        if ci == 0: ax.set_ylabel("High", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)