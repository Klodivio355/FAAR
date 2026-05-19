# --------------------------------------------------------
# MTLoRA
# GitHub: https://github.com/scale-lab/MTLoRA
# Built upon Microsoft LoRA (https://github.com/microsoft/LoRA)
#
# Original file:
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License
#
# Adapted file:
# Copyright (c) 2024 SCALE Lab, Brown University
# Licensed under the MIT License (see LICENSE for details)
# --------------------------------------------------------

"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self


import math
from typing import Optional, Dict, Union, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Dict, Union, Mapping, List, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: float, lora_dropout: float):
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else (lambda x: x)
        self.merged = False


class MTLoRALinear(LoRALayer):
    # LoRA implemented in a dense layer + in-training auto-rank hooks (EMA-based)
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = 0.0,
        ortho_weight: float = 0,
        ortho_detach_shared: bool = True,
        tasks=None,
        trainable_scale_shared: bool = False,
        trainable_scale_per_task: bool = False,
        shared_mode: str = 'matrix',
        eps_explore: float = 0.0,           # small prob to sample up to full R (keep-alive)
        score_beta: float = 0.95,           # EMA factor for performance-aware per-slot scores
        score_rho_shared: float = 0.90,     # cumulative threshold to propose K (shared)
        score_rho_task: float = 0.85,       # cumulative threshold to propose K (tasks)
        **kwargs,
    ):
        assert shared_mode in ['matrix', 'matrixv2', 'add', 'addition', 'lora_only']
        if shared_mode == 'add':
            shared_mode = 'addition'
        if shared_mode == 'lora_only':
            tasks = None
        has_tasks = tasks is not None
        if not has_tasks and shared_mode not in ['matrix']:
            shared_mode = 'matrix'

        if isinstance(r, int):
            r = {'shared': r}

        super().__init__(r=r['shared'], lora_alpha=lora_shared_scale, lora_dropout=lora_dropout)
        self.linear = nn.Linear(in_features, out_features, **kwargs)

        self.tasks = tasks
        self.shared_mode = shared_mode

        # ---------------- In-training Auto-Rank state (used by trainer) ----------------
        # Fixed inference ranks for eval; None => use current active cap.
        self.infer_rank_shared = None
        if has_tasks:
            self.infer_rank_tasks = {t: None for t in tasks}

        # Train-time overrides (optional); None => normal sampling.
        self.rank_override_shared = None
        if has_tasks:
            self.rank_override_tasks = {t: None for t in tasks}

        # Exploration keep-alive
        self.eps_explore = float(eps_explore)

        # EMA score config
        self.score_beta = float(score_beta)
        self.score_rho_shared = float(score_rho_shared)
        self._score_rho_task_default = float(score_rho_task)
        # -------------------------------------------------------------------------------

        if r['shared'] > 0:
            if has_tasks:
                self.ortho_weight = float(ortho_weight)
                self.ortho_detach_shared = bool(ortho_detach_shared)
                self.lora_tasks_A = nn.ParameterDict({
                    t: nn.Parameter(self.linear.weight.new_zeros((r[t], in_features)))
                    for t in tasks
                })
                self.lora_tasks_B = nn.ParameterDict({
                    t: nn.Parameter(self.linear.weight.new_zeros((out_features, r[t])))
                    for t in tasks
                })
                if trainable_scale_per_task:
                    self.lora_task_scale = nn.ParameterDict({
                        t: nn.Parameter(torch.tensor([lora_task_scale], dtype=torch.float32))
                        for t in tasks
                    })
                else:
                    self.lora_task_scale = {t: lora_task_scale[t] for t in tasks}


            if self.shared_mode == 'addition':
                assert has_tasks
                self.lora_norm = nn.LayerNorm(out_features)
            elif self.shared_mode in ['matrix', 'matrixv2']:
                self.lora_shared_A = nn.Parameter(self.linear.weight.new_zeros((r['shared'], in_features)))
                self.lora_shared_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r['shared'])))
            else:
                raise NotImplementedError

            if trainable_scale_shared:
                self.lora_shared_scale = nn.Parameter(torch.tensor([lora_shared_scale], dtype=torch.float32))
            else:
                self.lora_shared_scale = lora_shared_scale

            self.reset_parameters()

            # --------- [EMA scores + active caps] (performance-aware auto-rank) ---------
            if hasattr(self, "lora_shared_A"):
                R = int(self.lora_shared_A.shape[0])
                self.register_buffer("ema_score_shared", torch.zeros(R))   # per-slot importance (EMA)
                self.R_active_shared = R                                   # train-time cap; shrinks over epochs

            if has_tasks and hasattr(self, "lora_tasks_A"):
                # store per-task score buffers & caps as attributes
                for t, A in self.lora_tasks_A.items():
                    Rt = int(A.shape[0])
                    self.register_buffer(f"ema_score_task__{t}", torch.zeros(Rt))
                    setattr(self, f"R_active_task__{t}", Rt)
                    setattr(self, f"score_rho_task__{t}", self._score_rho_task_default)
            # ---------------------------------------------------------------------------

    def reset_parameters(self):
        if hasattr(self, "lora_shared_A"):
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_shared_B)  # standard LoRA: start ΔW≈0
        if hasattr(self, "lora_tasks_A"):
            for t in self.tasks:
                nn.init.kaiming_uniform_(self.lora_tasks_A[t], a=math.sqrt(5))
                nn.init.zeros_(self.lora_tasks_B[t])

    def _orthoreg(self, shared_res: torch.Tensor, task_res: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if self.ortho_detach_shared:
            shared_res = shared_res.detach()
        B, S, D = shared_res.shape
        s = F.normalize(shared_res.reshape(B*S, D), dim=-1, eps=eps)
        t = F.normalize(task_res.reshape(B*S, D), dim=-1, eps=eps)
        return (s * t).pow(2).mean()

    def merge(self):
        raise NotImplementedError

    # ---------------------- Forward with DyLoRA prefix + active caps ----------------------
    def forward(self, x, x_tasks=None, masks=None, masks_tasks=None):
        pretrained = self.linear(x)
        if self.r == 0:
            return pretrained, None

        x = self.lora_dropout(x)
        reg_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        # ---- per-token RMS scale from the frozen path (no grad, fp32) ----
        with torch.no_grad():
            s = pretrained.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

        # Shared adapter: choose prefix length b_shared
        R = self.lora_shared_A.shape[0]
        Rcap = getattr(self, "R_active_shared", R)

        if self.training:
            if self.rank_override_shared is not None:
                b_shared = int(self.rank_override_shared)
            else:
                if (self.eps_explore > 0.0) and (torch.rand(()) < self.eps_explore):
                    b_shared = torch.randint(1, R + 1, ()).item()
                else:
                    b_shared = torch.randint(1, Rcap + 1, ()).item()
        else:
            b_shared = self.infer_rank_shared if (self.infer_rank_shared is not None) else Rcap

        mask_shared = self.lora_shared_A.new_zeros(R)
        mask_shared[:b_shared] = 1.0
        A_eff = self.lora_shared_A * mask_shared.view(-1, 1)
        B_eff = self.lora_shared_B * mask_shared.view(1, -1)

        # shared residual
        lora = (x @ A_eff.transpose(0, 1) @ B_eff.transpose(0, 1)) * self.lora_shared_scale

        # Per-task adapters
        lora_tasks = None
        if self.tasks is not None:
            lora_tasks = {}
            for t in self.tasks:
                x_t = x if x_tasks is None else x_tasks[t]

                Rt = self.lora_tasks_A[t].shape[0]
                Rcap_t = getattr(self, f"R_active_task__{t}", Rt)

                if self.training:
                    forced = self.rank_override_tasks.get(t, None)
                    if forced is not None:
                        b_task = int(forced)
                    else:
                        if (self.eps_explore > 0.0) and (torch.rand(()) < self.eps_explore):
                            b_task = torch.randint(1, Rt + 1, ()).item()
                        else:
                            b_task = torch.randint(1, Rcap_t + 1, ()).item()
                else:
                    fixed_k = self.infer_rank_tasks.get(t, None)
                    b_task = fixed_k if (fixed_k is not None) else Rcap_t

                mask_t = self.lora_tasks_A[t].new_zeros(Rt)
                mask_t[:b_task] = 1.0
                A_eff_t = self.lora_tasks_A[t] * mask_t.view(-1, 1)
                B_eff_t = self.lora_tasks_B[t] * mask_t.view(1, -1)

                task_res = (x_t @ A_eff_t.transpose(0, 1) @ B_eff_t.transpose(0, 1)) * self.lora_task_scale[t]

                lora_tasks[t] = pretrained + task_res

                if self.ortho_weight > 0.0:
                    reg_loss = reg_loss + self.ortho_weight * self._orthoreg(lora, task_res)

        return pretrained + lora, lora_tasks, reg_loss
    
    @torch.no_grad()
    def update_rank_scores_from_grads(self):
        # ---- shared ----
        gA = getattr(self.lora_shared_A, "grad", None)
        gB = getattr(self.lora_shared_B, "grad", None)
        if (gA is not None) and (gB is not None):
            A32, B32 = self.lora_shared_A.detach().float(), self.lora_shared_B.detach().float()
            gA32, gB32 = gA.detach().float(), gB.detach().float()
            sA = (A32 * gA32).sum(dim=1).abs()
            sB = (B32 * gB32).sum(dim=0).abs()
            s  = 0.5 * (sA + sB)
            s  = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            if s.numel() > 0:
                s = torch.clamp(s, max=s.median().item() * 100.0 + 1e-12)
            self.ema_score_shared.mul_(self.score_beta).add_((1.0 - self.score_beta) * s.to(self.ema_score_shared.dtype))

        # ---- per-task ----
        if hasattr(self, "lora_tasks_A"):
            for t in self.lora_tasks_A.keys():
                gAt = getattr(self.lora_tasks_A[t], "grad", None)
                gBt = getattr(self.lora_tasks_B[t], "grad", None)
                if (gAt is None) or (gBt is None):
                    continue
                A32t, B32t = self.lora_tasks_A[t].detach().float(), self.lora_tasks_B[t].detach().float()
                gA32t, gB32t = gAt.detach().float(), gBt.detach().float()
                sAt = (A32t * gA32t).sum(dim=1).abs()
                sBt = (B32t * gB32t).sum(dim=0).abs()
                st  = 0.5 * (sAt + sBt)
                st  = torch.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0)
                if st.numel() > 0:
                    st = torch.clamp(st, max=st.median().item() * 100.0 + 1e-12)
                buf = getattr(self, f"ema_score_task__{t}")
                buf.mul_(self.score_beta).add_((1.0 - self.score_beta) * st.to(buf.dtype))

    @torch.no_grad()
    def shrink_active_cap(self, margin: int = 2, floor: int = 4):
        se = self.ema_score_shared.float()
        if se.numel() == 0 or se.sum() <= 0:
            return

        # 1) sort scores (desc) to identify best slots
        order = torch.argsort(se, descending=True)              # [R]
        se_sorted = se.index_select(0, order)
        ce = se_sorted.cumsum(0) / se_sorted.sum().clamp_min(1e-12)
        hit = (ce >= self.score_rho_shared).nonzero(as_tuple=True)[0]
        K = int(hit[0].item() + 1) if hit.numel() else se_sorted.numel()
        keep = max(floor, min(getattr(self, "R_active_shared", se.numel()), K + margin))

        # 2) move best 'keep' slots to the front (so prefix masking stays valid)
        R = self.lora_shared_A.shape[0]
        perm = torch.cat([order[:keep], order[keep:]], dim=0)   # [R]
        self.lora_shared_A.data = self.lora_shared_A.data.index_select(0, perm)
        self.lora_shared_B.data = self.lora_shared_B.data.index_select(1, perm)
        # reorder the score buffer too
        self.ema_score_shared.data = self.ema_score_shared.data.index_select(0, perm)

        # 3) hard-zero the tail and clamp the cap
        if keep < R:
            self.lora_shared_A.data[keep:].zero_()
            self.lora_shared_B.data[:, keep:].zero_()
        self.R_active_shared = keep

def mark_only_lora_as_trainable(
    model: nn.Module,
    bias: str = "none",
    freeze_patch_embed: bool = False,
    freeze_norm: bool = False,
    free_relative_bias: bool = False,
    freeze_downsample_reduction: bool = False,
) -> None:
    """Freeze all modules except LoRA's (per filters) and ALWAYS unfreeze FiLM params ('film' in name)."""

    def lora_filter(key): return "lora_" in key
    def patch_embed_filter(key): return not freeze_patch_embed and "patch_embed" in key
    def norm_filter(key): return not freeze_norm and "norm" in key
    def downsample_reduction_filter(key): return not freeze_downsample_reduction and "downsample.reduction" in key
    def relative_position_bias_filter(key): return not free_relative_bias and "relative_position_bias_table" in key
    def all_filters(key):
        return (
            lora_filter(key)
            or patch_embed_filter(key)
            or norm_filter(key)
            or downsample_reduction_filter(key)
            or relative_position_bias_filter(key)
        )

    print(f"LoRA bias mode: {bias}")
    print(f"LoRA Freeze patch_embed: {freeze_patch_embed}")
    print(f"LoRA Freeze norm: {freeze_norm}")
    print(f"LoRA Freeze downsample_reduction: {freeze_downsample_reduction}")
    print(f"LoRA Freeze relative_position_bias: {free_relative_bias}")

    # 1) Freeze everything except what all_filters() allows
    for n, p in model.named_parameters():
        if not all_filters(n):
            p.requires_grad = False

    # 3) Bias handling (does not override FiLM enabling)
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    elif bias != "none":
        raise NotImplementedError

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[mark_only_lora_as_trainable] Trainable params: {trainable} / {total}")


def merge_lora_weights(model) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str, split_qkv: bool = False) -> Dict:
    unmatched_keys = []
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            weights = state_dict.pop(
                full_checkpoint_name)
            last_four = ".".join(full_attribute_name.split(".")[-4:])
            if split_qkv and last_four in ["attn.qkv.linear.weight", "attn.qkv.linear.bias"]:
                w_q, w_k, w_v = torch.chunk(weights, chunks=3)
                weight_bias = last_four.split(".")[-1]
                full_attribute_name_without_suffix = ".".join(full_attribute_name.split(".")[
                    :-2])
                state_dict[f"{full_attribute_name_without_suffix}.q.linear.{weight_bias}"] = w_q
                state_dict[f"{full_attribute_name_without_suffix}.k.linear.{weight_bias}"] = w_k
                state_dict[f"{full_attribute_name_without_suffix}.v.linear.{weight_bias}"] = w_v
            else:
                state_dict[full_attribute_name] = weights
        else:
            unmatched_keys.append(checkpoint_name)
    if len(unmatched_keys) > 0:
        print(
            f"WARNING: The following keys from the checkpoint were not mapped: {unmatched_keys}")
    return state_dict