import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
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
        score_beta: float = 0.98,           # EMA factor for performance-aware per-slot scores
        score_rho_shared: float = 0.2,     # cumulative threshold to propose K (shared)
        score_rho_task: float = 0.2,       # cumulative threshold to propose K (tasks)
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

            # ======== ADDED: per-row log-magnitudes (positive via softplus) =========
            with torch.no_grad():
                base_norm = torch.norm(self.linear.weight, dim=1).clamp_min(1e-6)  # [out]
            self.shared_log_m = nn.Parameter(base_norm.log())                      # [out]
            if has_tasks:
                self.task_log_m = nn.ParameterDict({
                    t: nn.Parameter(base_norm.log().clone()) for t in tasks
                })
            # ========================================================================

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


    def forward(self, x, x_tasks=None, masks=None, masks_tasks=None):
        pretrained = self.linear(x)
        if self.r == 0:
            return pretrained, None

        reg_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
        x_drop = self.lora_dropout(x)

        # ====== SHARED BRANCH ======
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

        # Low-rank update
        deltaW_shared = (B_eff @ A_eff)  # [out,in]

        # Pretrained direction (frozen) and base row norms
        W0 = self.linear.weight.detach().float()                # [out,in], fp32 for stability
        V0 = F.normalize(W0, dim=1, eps=1e-6)                   # row-normalized
        base_norm = W0.norm(dim=1)                              # [out]

        # Scale LoRA delta (like lora_alpha) before forming direction
        v = (V0 + (self.lora_shared_scale * deltaW_shared).float())
        V_hat_shared = F.normalize(v, dim=1, eps=1e-6).to(self.linear.weight.dtype)  # proper grad through norm

        # Bounded, anchored magnitudes: m = base_norm * exp(β*(θ - θ0)), θ0 = log(base_norm)
        # This keeps m==base_norm at init and damps gradients by β.
        beta_m = 0.1
        if hasattr(self, "shared_log_m"):
            theta = self.shared_log_m                              # param, init ≈ log(base_norm)
            theta0 = base_norm.log().to(theta.dtype)
            m_eff = (base_norm * torch.exp(beta_m * (theta - theta0))).to(self.linear.weight.dtype)
        else:
            m_eff = base_norm.to(self.linear.weight.dtype)

        # DoRA weight and residual delta vs. frozen W0
        W_dora_shared = (m_eff.unsqueeze(1) * V_hat_shared)        # [out,in]
        deltaW_dora = (W_dora_shared - self.linear.weight.detach()).to(self.linear.weight.dtype)

        # Apply as residual (keeps LoRA-like training dynamics)
        lora_shared_res = F.linear(x_drop, deltaW_dora, bias=None)
        out = pretrained + lora_shared_res

        # Small regularizer to keep magnitudes near base (tunable)
        reg_loss = reg_loss + 1e-4 * ((m_eff / base_norm) - 1.0).pow(2).mean()

        # ====== PER-TASK BRANCHES (optional) ======
        lora_tasks = None
        if self.tasks is not None:
            lora_tasks = {}
            for t in self.tasks:
                x_t = x if x_tasks is None else x_tasks[t]
                x_t_drop = self.lora_dropout(x_t)

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

                deltaW_t = (B_eff_t @ A_eff_t)                     # [out,in]
                v_t = (V0 + (self.lora_task_scale[t] * deltaW_t).float())
                V_hat_t = F.normalize(v_t, dim=1, eps=1e-6).to(self.linear.weight.dtype)

                if hasattr(self, "task_log_m"):
                    theta_t = self.task_log_m[t]
                    theta0_t = base_norm.log().to(theta_t.dtype)
                    m_eff_t = (base_norm * torch.exp(beta_m * (theta_t - theta0_t))).to(self.linear.weight.dtype)
                else:
                    m_eff_t = m_eff  # fallback

                W_dora_t = (m_eff_t.unsqueeze(1) * V_hat_t)
                deltaW_dora_t = (W_dora_t - self.linear.weight.detach()).to(self.linear.weight.dtype)

                out_t = pretrained + F.linear(x_t_drop, deltaW_dora_t, bias=None)
                lora_tasks[t] = out_t

                # Keep your orthoreg behavior
                if self.ortho_weight > 0.0:
                    task_res = out_t - pretrained
                    reg_loss = reg_loss + self.ortho_weight * self._orthoreg(lora_shared_res, task_res)

        return out, lora_tasks, reg_loss

            
    
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
        # ====== SHARED ADAPTER ======
        if hasattr(self, "ema_score_shared"):
            se = self.ema_score_shared.float()
            if se.numel() > 0 and se.sum() > 0:
                # 1) sort scores (desc) and compute coverage
                order = torch.argsort(se, descending=True)              # [R]
                se_sorted = se.index_select(0, order)
                ce = se_sorted.cumsum(0) / se_sorted.sum().clamp_min(1e-12)
                hit = (ce >= self.score_rho_shared).nonzero(as_tuple=True)[0]
                K = int(hit[0].item() + 1) if hit.numel() else int(se_sorted.numel())

                # respect previous active cap and floor, add small margin
                cur_cap = getattr(self, "R_active_shared", se.numel())
                keep = max(floor, min(cur_cap, K + margin))

                # 2) move best 'keep' slots to the front (so prefix masking uses them)
                R = self.lora_shared_A.shape[0]
                perm = torch.cat([order[:keep], order[keep:]], dim=0)   # [R]
                self.lora_shared_A.data = self.lora_shared_A.data.index_select(0, perm)
                self.lora_shared_B.data = self.lora_shared_B.data.index_select(1, perm)
                self.ema_score_shared.data = self.ema_score_shared.data.index_select(0, perm)

                # 3) hard-zero tail and clamp the cap
                if keep < R:
                    self.lora_shared_A.data[keep:].zero_()
                    self.lora_shared_B.data[:, keep:].zero_()
                self.R_active_shared = keep

        # ====== TASK-SPECIFIC ADAPTERS ======
        if hasattr(self, "lora_tasks_A"):
            for t, A_t in self.lora_tasks_A.items():
                buf = getattr(self, f"ema_score_task__{t}", None)
                if buf is None:
                    continue
                st = buf.float()
                if st.numel() == 0 or st.sum() <= 0:
                    continue

                # 1) sort scores (desc) and compute coverage for this task
                order_t = torch.argsort(st, descending=True)            # [R_t]
                st_sorted = st.index_select(0, order_t)
                ct = st_sorted.cumsum(0) / st_sorted.sum().clamp_min(1e-12)
                rho_t = getattr(self, f"score_rho_task__{t}", self._score_rho_task_default)
                hit_t = (ct >= rho_t).nonzero(as_tuple=True)[0]
                Kt = int(hit_t[0].item() + 1) if hit_t.numel() else int(st_sorted.numel())

                cap_name = f"R_active_task__{t}"
                cur_cap_t = getattr(self, cap_name, st_sorted.numel())
                keep_t = max(floor, min(cur_cap_t, Kt + margin))

                # 2) permute A_t/B_t so best 'keep_t' are at the front
                Rt = A_t.shape[0]
                perm_t = torch.cat([order_t[:keep_t], order_t[keep_t:]], dim=0)
                self.lora_tasks_A[t].data = self.lora_tasks_A[t].data.index_select(0, perm_t)
                self.lora_tasks_B[t].data = self.lora_tasks_B[t].data.index_select(1, perm_t)
                buf.data = buf.data.index_select(0, perm_t)

                # 3) hard-zero tail and update this task's active cap
                if keep_t < Rt:
                    self.lora_tasks_A[t].data[keep_t:].zero_()
                    self.lora_tasks_B[t].data[:, keep_t:].zero_()
                setattr(self, cap_name, keep_t)



def mark_only_lora_as_trainable(
    model: nn.Module,
    bias: str = "none",
    freeze_patch_embed: bool = False,
    freeze_norm: bool = False,
    free_relative_bias: bool = False,
    freeze_downsample_reduction: bool = False
) -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            "none": all bias weights will be frozen,
            "lora_only": only bias weight for LoRA layers will be unfrozen,
            "all": all bias weights will be unfrozen.
    """
    # keep LoRA params trainable (A/B, scales, etc.)
    def lora_filter(key): 
        return "lora_" in key

    # === UPDATED: keep DoRA magnitude params trainable ===
    # supports both old name ("weight_magnitude") and new log-param names
    def weight_decomp_filter(key):
        return (
            ("weight_magnitude" in key) or      # legacy magnitude param
            ("shared_log_m" in key) or          # DoRA shared magnitude (log-param)
            ("task_log_m" in key) or            # DoRA per-task magnitudes (ParameterDict)
            key.endswith(".log_m")              # any other module-scoped '...log_m'
        )

    def patch_embed_filter(key): 
        return (not freeze_patch_embed) and ("patch_embed" in key)

    def norm_filter(key): 
        return (not freeze_norm) and ("norm" in key)

    def downsample_reduction_filter(key): 
        return (not freeze_downsample_reduction) and ("downsample.reduction" in key)

    def relative_position_bias_filter(key): 
        return (not free_relative_bias) and ("relative_position_bias_table" in key)

    def all_filters(key):
        return (
            lora_filter(key) or
            weight_decomp_filter(key) or
            patch_embed_filter(key) or
            norm_filter(key) or
            downsample_reduction_filter(key) or
            relative_position_bias_filter(key)
        )

    print(f"LoRA bias mode: {bias}")
    print(f"LoRA Freeze patch_embed: {freeze_patch_embed}")
    print(f"LoRA Freeze norm: {freeze_norm}")
    print(f"LoRA Freeze downsample_reduction: {freeze_downsample_reduction}")
    print(f"LoRA Freeze relative_position_bias: {free_relative_bias}")

    # freeze all params except those matched by the filters above
    for n, p in model.named_parameters():
        if not all_filters(n):
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and (m.bias is not None):
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def merge_lora_weights(model) -> None:
    """Merge DoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        # Only DoRALinear and DoRAQKVLinear have direct merge logic
        if isinstance(module, (LoRALinear)):
            module.merge()
        # MTLoRADecomposedLinear does not have a direct merge, consistent with original


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str, split_qkv: bool = False) -> Dict:
    # This function is for state dict mapping, DoRA doesn't change the *names* of the base weights in the state_dict
    # so no changes are strictly necessary here, as it maps from checkpoint names to attribute names.
    # New DoRA parameters (lora_magnitude) would simply be loaded if their names match.
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
