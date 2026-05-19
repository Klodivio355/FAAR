import numpy as np
import matplotlib.pyplot as plt

def show_frequency_bands_grid(
    batch_image,
    num_samples=3,                              # randomly pick this many images (< batch size)
    save_path="freq_bands_grid.png",            # one figure saved to disk
    # Five bands covering 0..Nyquist; if None -> auto equal-power per image
    k_bands=((0.00,0.02), (0.02,0.06), (0.06,0.12), (0.12,0.25), (0.25,0.50)),
    equal_power_if_none=True,
    # Display: binary masks built from soft maps
    band_thresh_pct=None,                       # scalar or list; if None -> sensible per-band defaults
    blur_sigma_rel=0.01,                        # Gaussian blur sigma relative to min(H,W) for soft maps
    gamma=0.7,                                  # <1 brightens weaker responses in soft maps
    # Input handling
    denorm="auto",                              # "auto" | "imagenet" | "custom" | "none"
    mean=None, std=None,                        # for denorm="custom"
    seed=None,
    return_softmaps=False                       # if True -> returns dict with soft maps for the chosen rows
):
    """
    Saves ONE figure with columns: [Original | Low | Mid-Low | Mid | Mid-High | High].
    Each band panel is a BINARY image (white on black) showing where that band is strong.
    Soft (0..1) band maps are computed internally; set return_softmaps=True to also get them back.

    Accepts batch_image as: [B,3,H,W], [B,H,W,3], [3,H,W], [H,W,3], or [H,W].
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

    def _ensure_batch_hwc(b):
        a = _to_np(b)
        if a.ndim == 4 and a.shape[-1] in (1,3):   # [B,H,W,C]
            if a.shape[-1] == 1: a = np.repeat(a, 3, axis=-1)
            return a.astype(np.float64)
        if a.ndim == 4 and a.shape[1] in (1,3):    # [B,C,H,W]
            a = np.moveaxis(a, 1, -1)              # -> [B,H,W,C]
            if a.shape[-1] == 1: a = np.repeat(a, 3, axis=-1)
            return a.astype(np.float64)
        if a.ndim == 3 and a.shape[0] in (1,3):    # [C,H,W] -> add batch
            return np.moveaxis(a[None, ...], 1, -1).astype(np.float64)  # [1,H,W,C]
        if a.ndim == 3 and a.shape[-1] in (1,3):   # [H,W,C] -> add batch
            return a[None, ...].astype(np.float64)
        if a.ndim == 2:                             # [H,W] -> add C,B
            return np.repeat(a[None, ..., None], 3, axis=-1).astype(np.float64)
        raise ValueError(f"Unsupported shape for batch_image: {a.shape}")

    def _pnorm(x, lo=1, hi=99):
        a, b = np.percentile(x, [lo, hi])
        if b - a < 1e-9: return np.zeros_like(x)
        return np.clip((x - a) / (b - a), 0, 1)

    def _maybe_denorm(BHWc):
        out = BHWc.copy()
        if out.max() > 1.5 and out.dtype.kind != 'f':  # 0..255
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
        else:  # "auto"
            if (out.min() < -0.2) or (out.max() > 1.2):
                if out.min() >= -4 and out.max() <= 4:  # assume ImageNet z-score
                    m = np.array([0.485,0.456,0.406]).reshape(1,1,1,3)
                    s = np.array([0.229,0.224,0.225]).reshape(1,1,1,3)
                    out = out * s + m
                else:
                    for c in range(3): out[..., c] = _pnorm(out[..., c], 1, 99)
                    return np.clip(out, 0, 1)
        out = np.clip(out, 0, 1)
        if not looks_01:
            for c in range(3): out[..., c] = _pnorm(out[..., c], 1, 99)
        return np.clip(out, 0, 1)

    def _hann2d(H, W):
        hy = np.hanning(H)[:, None]; hx = np.hanning(W)[None, :]
        return hy * hx

    def _luma(rgb):
        return 0.2989*rgb[...,0] + 0.5870*rgb[...,1] + 0.1140*rgb[...,2]

    def _gauss_blur(gray, rel_sigma):
        if rel_sigma <= 0: return gray
        # separable Gaussian via NumPy (no SciPy)
        sigma = max(0.5, rel_sigma * min(gray.shape))
        k = int(np.ceil(6*sigma)) | 1
        x = np.arange(k) - k//2
        ker = np.exp(-(x**2)/(2*sigma**2)); ker /= ker.sum()
        tmp = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 1, gray)
        out = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 0, tmp)
        return out

    def _majority3(bw):
        # 3x3 majority filter (keep if >=5 of 9 pixels are on)
        ker = np.ones((3,3), dtype=np.float32)
        from numpy.lib.stride_tricks import sliding_window_view as swv
        pad = np.pad(bw, 1, mode='constant')
        win = swv(pad, (3,3))
        cnt = (win * ker).sum(axis=(-1,-2))
        return (cnt >= 5).astype(np.float32)

    # ---------- prep batch ----------
    BHWc = _ensure_batch_hwc(batch_image)         # [B,H,W,3], float
    BHWc = _maybe_denorm(BHWc)                    # -> [0,1]
    B, H, W, _ = BHWc.shape

    # figure will assume a fixed #bands across rows:
    n_bands = 5 if k_bands is None else len(k_bands)
    band_names = (["Low","Mid-Low","Mid","Mid-High","High"] 
                  if n_bands == 5 else [f"Band {i+1}" for i in range(n_bands)])

    # thresholds: accept scalar or list; pad/trim to n_bands
    if band_thresh_pct is None:
        if n_bands == 5:
            band_thresh_list = [95, 93, 91, 89, 87]
        else:
            # linearly relax from 95 to 87 across bands
            band_thresh_list = list(np.linspace(95, 87, n_bands))
    else:
        try:
            # try iterating (list/tuple/np array)
            band_thresh_list = [float(v) for v in band_thresh_pct]
            if len(band_thresh_list) == 0:
                band_thresh_list = [90.0]
        except TypeError:
            # scalar
            band_thresh_list = [float(band_thresh_pct)]
        # pad/trim to n_bands
        if len(band_thresh_list) < n_bands:
            band_thresh_list += [band_thresh_list[-1]] * (n_bands - len(band_thresh_list))
        elif len(band_thresh_list) > n_bands:
            band_thresh_list = band_thresh_list[:n_bands]

    # pick rows
    if num_samples >= B:
        num_samples = max(1, B-1)  # keep at least one row even if B==1
    rng = np.random.default_rng(seed)
    sel = rng.choice(B, size=num_samples, replace=False)

    # ---------- figure ----------
    ncols = 1 + n_bands
    nrows = num_samples
    fig_w, fig_h = 3.1 * ncols, 3.1 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if nrows == 1: axes = np.expand_dims(axes, 0)

    # ---------- store soft maps if needed ----------
    softmaps_all = []          # list of (n_bands, H, W) per image
    bands_used_all = []        # list of band tuples per image

    # ---------- per selected image ----------
    for r, bi in enumerate(sel):
        rgb = BHWc[bi]
        gray = _luma(rgb)
        F = np.fft.fft2(gray * _hann2d(H, W))
        Fsh = np.fft.fftshift(F)

        fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
        fy_s = np.fft.fftshift(fy, axes=0); fx_s = np.fft.fftshift(fx, axes=1)
        R = np.sqrt(fx_s**2 + fy_s**2)            # radial freq 0..0.5
        P = np.abs(Fsh)**2

        # bands for this image
        if k_bands is None and equal_power_if_none:
            rflat, wflat = R.ravel(), P.ravel()
            order = np.argsort(rflat)
            rs, ws = rflat[order], wflat[order]
            csum = np.cumsum(ws); tot = csum[-1] + 1e-12
            # n_bands equal-power quantiles
            qs = [rs[np.searchsorted(csum, tot*(i/n_bands), side="left")] for i in range(1, n_bands)]
            edges = [0.0] + [float(q) for q in qs] + [0.5]
            bands = [(edges[i], edges[i+1]) for i in range(n_bands)]
        else:
            bands = list(k_bands)
            bands[0] = (max(0.0, bands[0][0]), min(0.5, bands[0][1]))
            for i in range(1, len(bands)-1):
                bands[i] = (max(0.0, bands[i][0]), min(0.5, bands[i][1]))
            bands[-1] = (max(0.0, bands[-1][0]), 0.5)

        # col 0: original
        ax = axes[r, 0]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title("Original", fontsize=11)
        ax.axis("off")

        # build soft+binary maps
        soft_maps = []
        for j, (fmin, fmax) in enumerate(bands, start=1):
            mask = (R >= fmin) & ((R <= fmax + 1e-12) if j == len(bands) else (R < fmax))
            Fb = Fsh * mask
            recon = np.fft.ifft2(np.fft.ifftshift(Fb))      # complex
            env = np.abs(recon)                              # energy envelope
            env = _gauss_blur(env, blur_sigma_rel)
            env = (_pnorm(env, 1, 99)) ** gamma             # soft map in [0,1]
            soft_maps.append(env)

            # binary for display (white on black)
            pct = band_thresh_list[j-1]                     # safe: list sized to n_bands
            thr = np.percentile(env, pct)
            binary = (env >= max(thr, 1e-8)).astype(np.float32)
            binary = _majority3(binary)

            # title per band
            title = band_names[j-1] if len(band_names) >= j else f"Band {j}"

            ax = axes[r, j]
            ax.imshow(binary, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_facecolor("black")
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        softmaps_all.append(np.stack(soft_maps, axis=0))  # (n_bands,H,W)
        bands_used_all.append(bands)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)

    if return_softmaps:
        # shape -> (num_samples, n_bands, H, W)
        soft = np.stack(softmaps_all, axis=0).astype(np.float32)
        return {
            "indices": sel.tolist(),
            "softmaps": soft,              # float32 0..1
            "bands_per_image": bands_used_all
        }

