import math
from typing import Any, Optional, Tuple, List, Dict
import torch
import torch.nn.functional as F

try: # optional dependency for progress bar
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from tokenfool.adapters.interfaces import HookableTransformerClassifier


def ATT(
    model: HookableTransformerClassifier,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    *,
    epsilon: float = 16 / 255,
    iters: int = 10,
    decay: float = 1.0,
    gamma: float = 0.5,
    lam: float = 0.0,
    # ViT defaults from the reference repo 
    weaken_factor: Tuple[float, float, float] = (0.45, 0.7, 0.65),
    keep_ratio: float = 10 / 12,  
    mu: Tuple[float, float, float] =(0.485, 0.456, 0.406),
    std: Tuple[float, float, float]=(0.229, 0.224, 0.225),
    targeted: bool = False,
    patch_out: bool = True,
    patch_size: int = 16,
    image_size: int = 224,
    gf_scale: float = 0.4,
    gf_offset: float = 0.4,
    progress: bool = False,
) -> torch.Tensor:
    """
    Perform the Adaptive Token Tuning (ATT) adversarial attack on a
    Transformer-based classifier.
    Taken and adapted from the official implementation at: 

    This implementation assumes a ViT/DeiT-style transformer architecture
    and requires a model adapter implementing HookableTransformerClassifier.

    Parameters
    ----------
    model : HookableTransformerClassifier
        Transformer classifier adapter exposing internal modules required
        for gradient hooks. Must implement:

            - hook_modules() -> Dict[str, List[nn.Module]]
            - att_feature_module() -> nn.Module

    x : torch.Tensor
        Input tensor of shape (B, C, H, W), normalized using (mu, std).

    y : torch.Tensor, optional
        Ground-truth labels of shape (B,). If None, model predictions are
        used as pseudo-labels.

    epsilon : float
        Maximum perturbation magnitude in pixel space.

    steps : int
        Number of attack iterations.

    decay : float
        Momentum decay factor used in the iterative update.

    gamma : float
        Base scaling factor applied to gradient modulation inside hooks.

    lam : float
        Adaptive scaling coefficient controlling variance-based gradient
        rescaling.

    weaken_factor : tuple[float, float, float]
        Multiplicative factors applied to gradients for attention,
        QKV projection, and MLP modules respectively.

    keep_ratio : float
        Fraction of transformer layers allowed to contribute gradients.
        Earlier layers are truncated.

    mu : tuple[float]
        Dataset mean used for input normalization.

    std : tuple[float]
        Dataset standard deviation used for input normalization.

    Returns
    -------
    x_adv : torch.Tensor
        Adversarial examples of shape (B, C, H, W).

    mask : None
        ATT does not produce a spatial perturbation mask.
    """
    required = ("hook_modules", "att_feature_module")
    missing = [m for m in required if not hasattr(model, m)]
    if missing:
        raise TypeError(f"ATT requires HookableTransformerClassifier methods: missing {missing}")

    device = x.device
    dtype = x.dtype
    x = x.to(device)

    # labels
    if y is None:
        with torch.no_grad():
            y = model.logits(x).argmax(dim=1)
    y = y.to(device)

    mu_t = torch.tensor(mu, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_sign = -1.0 if targeted else 1.0

    # hook targets
    mods = model.hook_modules()
    attn_drop_mods = mods["attn_drop"]
    qkv_mods = mods["qkv"]
    mlp_mods = mods["mlp"]

    num_blocks = len(attn_drop_mods)
    if not (len(qkv_mods) == len(mlp_mods) == num_blocks):
        raise TypeError("hook_modules() must return equally-sized per-block lists for attn_drop/qkv/mlp.")

    k_keep = int(round(keep_ratio * num_blocks))
    k_keep = max(0, min(num_blocks, k_keep))
    truncate_layers = torch.cat(
        (torch.ones(k_keep, device=device, dtype=x.dtype), torch.zeros(num_blocks - k_keep, device=device, dtype=x.dtype))
    )

    # shared hook state
    state: Dict[str, Any] = {
        "var_A": torch.tensor(0.0, device=device, dtype=dtype),
        "var_qkv": torch.tensor(0.0, device=device, dtype=dtype),
        "var_mlp": torch.tensor(0.0, device=device, dtype=dtype),
        "back_attn": num_blocks - 1,
        "im_fea": None,
        "im_grad": None,
    }

    # --- helpers
    def _ret_like(grad_in, new0):
        if len(grad_in) == 0:
            return tuple()
        out = [new0]
        for i in range(1, len(grad_in)):
            out.append(grad_in[i])
        return tuple(out)
    
    def _safe_gpf(out_grad: torch.Tensor, prev_var: torch.Tensor) -> torch.Tensor:
        if prev_var is None or float(prev_var.detach()) == 0.0:
            return torch.tensor(gamma, device=out_grad.device, dtype=out_grad.dtype)
        gpf = gamma + lam * (1.0 - torch.sqrt(torch.var(out_grad) / (prev_var + 1e-12)))
        return gpf.clamp(0.0, 1.0)
    
    def _attn_vit_extrema_scale(out_grad: torch.Tensor, gpf: torch.Tensor) -> torch.Tensor:
        # ViT attention-like path: [B, C, H, W]
        if out_grad.ndim != 4:
            return out_grad * gpf

        _, c, h, w = out_grad.shape
        flat = out_grad.detach()[0].reshape(c, h * w)

        max_all = flat.argmax(dim=1)
        min_all = flat.argmin(dim=1)

        max_h = torch.div(max_all, h, rounding_mode="floor")
        max_w = max_all % h
        min_h = torch.div(min_all, h, rounding_mode="floor")
        min_w = min_all % h

        c_idx = torch.arange(c, device=out_grad.device)

        out_grad[:, c_idx, max_h, :] *= gpf
        out_grad[:, c_idx, :, max_w] *= gpf
        out_grad[:, c_idx, min_h, :] *= gpf
        out_grad[:, c_idx, :, min_w] *= gpf
        return out_grad

    def _token_vit_extrema_scale(out_grad: torch.Tensor, gpf: torch.Tensor) -> torch.Tensor:
        # ViT token-like path: [B, N, C]
        if out_grad.ndim != 3:
            return out_grad * gpf

        _, _, c = out_grad.shape
        ref = out_grad.detach()[0]  # [N, C]
        max_all = ref.argmax(dim=0)
        min_all = ref.argmin(dim=0)
        c_idx = torch.arange(c, device=out_grad.device)

        out_grad[:, max_all, c_idx] *= gpf
        out_grad[:, min_all, c_idx] *= gpf
        return out_grad
    
    # --- hook functions 
    def attn_hook(module, grad_in, grad_out):
        if len(grad_in) == 0 or grad_in[0] is None:
            return None

        li = int(state["back_attn"])
        li = max(0, min(num_blocks - 1, li))  

        out_grad = grad_in[0] * truncate_layers[li] * weaken_factor[0]
        gpf = _safe_gpf(out_grad, state["var_A"])
        out_grad = _attn_vit_extrema_scale(out_grad, gpf)

        state["var_A"] = torch.var(out_grad.detach())
        state["back_attn"] = li - 1
        return _ret_like(grad_in, out_grad)

    def qkv_hook(module, grad_in, grad_out):
        if len(grad_in) == 0 or grad_in[0] is None:
            return None

        out_grad = grad_in[0] * weaken_factor[1]
        gpf = _safe_gpf(out_grad, state["var_qkv"])
        out_grad = _token_vit_extrema_scale(out_grad, gpf)

        state["var_qkv"] = torch.var(out_grad.detach())
        return _ret_like(grad_in, out_grad)

    def mlp_hook(module, grad_in, grad_out):
        if len(grad_in) == 0 or grad_in[0] is None:
            return None

        out_grad = grad_in[0] * weaken_factor[2]
        gpf = _safe_gpf(out_grad, state["var_mlp"])
        out_grad = _token_vit_extrema_scale(out_grad, gpf)

        state["var_mlp"] = torch.var(out_grad.detach())
        return _ret_like(grad_in, out_grad)

    # feature hooks 
    def fea_hook(module, inputs, output):
        if torch.is_tensor(output):
            state["im_fea"] = output
        elif isinstance(output, (tuple, list)) and len(output) and torch.is_tensor(output[0]):
            state["im_fea"] = output[0]
        else:
            state["im_fea"] = None

    def grad_hook(module, grad_in, grad_out):
        if isinstance(grad_out, (tuple, list)) and len(grad_out) and torch.is_tensor(grad_out[0]):
            state["im_grad"] = grad_out[0]
        elif torch.is_tensor(grad_out):
            state["im_grad"] = grad_out
        else:
            state["im_grad"] = None

    def _patch_index(size: int, img_size: int) -> torch.Tensor:
        p = int((img_size - size) / size + 1)
        q = p
        idx = torch.empty((p * q, size * size), dtype=torch.long, device=device)
        tmp = 0
        for q_ in range(q):
            for p_ in range(p):
                for i in range(size):
                    for j in range(size):
                        idx[tmp, i * size + j] = (q_ * size + i) * img_size + (p_ * size + j)
                tmp += 1
        return idx
    
    def _norm_patches(gf: torch.Tensor, index: torch.Tensor, patch: int, scale: float, offset: float) -> torch.Tensor:
        gf = gf.clone()
        flat = gf.view(gf.shape[0], -1)
        patch_area = patch * patch

        for b in range(flat.shape[0]):
            tmp = flat[b].take(index)     
            norm_tmp = tmp.mean(dim=-1)   # [num_patches]
            denom = (norm_tmp.max() - norm_tmp.min()).clamp_min(1e-12)
            scale_norm = scale * ((norm_tmp - norm_tmp.min()) / denom) + offset
            tmp_bi = scale_norm.repeat_interleave(patch_area)
            flat[b].put_(index.reshape(-1), tmp_bi)
        return flat.view_as(gf)

    def _build_patch_mask_from_refs(
        step_idx: int,
        gf_feat_ref: torch.Tensor,
        gf_grad_ref: torch.Tensor,
        patch_index: torch.Tensor,
    ) -> torch.Tensor:
        if gf_feat_ref.ndim != 3 or gf_grad_ref.ndim != 3:
            raise RuntimeError(
                f"ATT expects token-shaped feature refs [B, N, C], got "
                f"{tuple(gf_feat_ref.shape)} and {tuple(gf_grad_ref.shape)}"
            )
        if gf_feat_ref.shape != gf_grad_ref.shape:
            raise RuntimeError(
                f"ATT feature/gradient shapes must match, got "
                f"{tuple(gf_feat_ref.shape)} and {tuple(gf_grad_ref.shape)}"
            )

        prefix = int(model.num_prefix_tokens)
        feat_p = gf_feat_ref[:, prefix:, :]
        grad_p = gf_grad_ref[:, prefix:, :]

        num_patch_tokens = feat_p.shape[1]
        side = int(math.sqrt(num_patch_tokens))
        if side * side != num_patch_tokens:
            raise RuntimeError(
                f"ATT expects square patch-token grid after removing prefix tokens, got {num_patch_tokens}"
            )

        # ViT ATT map: feature-gradient product over patch tokens
        gf = (feat_p * grad_p).sum(dim=-1).view(feat_p.shape[0], side, side)

        gf = F.interpolate(
            gf.unsqueeze(1),
            size=(image_size, image_size),
            mode="nearest",
        ).squeeze(1)

        gf_patchs_t = _norm_patches(gf, patch_index, patch_size, gf_scale, gf_offset)
        gf_patchs_start = torch.ones_like(gf_patchs_t, device=device, dtype=dtype) * 0.99
        gf_delta = (gf_patchs_start - gf_patchs_t) / iters

        cpu_state = torch.random.get_rng_state()
        torch.manual_seed(step_idx)
        random_patch = torch.rand(
            image_size // patch_size,
            image_size // patch_size,
            device=device,
            dtype=dtype,
        )
        torch.random.set_rng_state(cpu_state)

        random_patch = (
            random_patch.repeat_interleave(patch_size, dim=0)
            .repeat_interleave(patch_size, dim=1)
        )

        thresh = gf_patchs_start - gf_delta * (step_idx + 1)
        gf_mask = torch.where(random_patch.unsqueeze(0) > thresh, 0.0, 1.0)
        return gf_mask.unsqueeze(1) 
    
    # Register hooks
    handles: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for m in attn_drop_mods:
            handles.append(m.register_full_backward_hook(attn_hook))
        for m in qkv_mods:
            handles.append(m.register_full_backward_hook(qkv_hook))
        for m in mlp_mods:
            handles.append(m.register_full_backward_hook(mlp_hook))

        feat_mod = model.att_feature_module()
        handles.append(feat_mod.register_forward_hook(fea_hook))
        handles.append(feat_mod.register_full_backward_hook(grad_hook))

        x_unnorm = x * std_t + mu_t
        perts = torch.zeros_like(x_unnorm, requires_grad=True)
        momentum = torch.zeros_like(x_unnorm)
        step_size = epsilon / iters
        patch_index = _patch_index(patch_size, image_size)

        if patch_out:
            model.zero_grad(set_to_none=True)
            init_x = x.detach().clone().requires_grad_(True)
            init_logits = model.logits(init_x)
            init_logits.backward(torch.ones_like(init_logits))

            if state["im_fea"] is None or state["im_grad"] is None:
                raise RuntimeError("ATT: feature/gradient hooks did not capture any data; check model adapter.")
            
            gf_feat_ref = state["im_fea"].detach()
            gf_grad_ref = state["im_grad"].detach()
            model.zero_grad(set_to_none=True)
        else:
            gf_feat_ref = None
            gf_grad_ref = None

        iterator = range(iters)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="Adaptive Token Tuning (ATT)")

        for i in iterator:
            # reset per-step state
            state["back_attn"] = num_blocks - 1
            state["var_A"] = torch.tensor(0.0, device=device, dtype=dtype)
            state["var_qkv"] = torch.tensor(0.0, device=device, dtype=dtype)
            state["var_mlp"] = torch.tensor(0.0, device=device, dtype=dtype)

            if perts.grad is not None:
                perts.grad.zero_()
            model.zero_grad(set_to_none=True)

            if patch_out:
                gf_mask = _build_patch_mask_from_refs(i, gf_feat_ref, gf_grad_ref, patch_index)
                adv_unnorm = x_unnorm + perts * gf_mask.detach()
            else:
                adv_unnorm = x_unnorm + perts

            adv = (adv_unnorm - mu_t) / std_t
            # forward + backward
            logits = model.logits(adv)
            loss = loss_sign *loss_fn(logits, y)
            loss.backward()

            grad = perts.grad
            if grad is None:
                raise RuntimeError("ATT: perts.grad is None; check that perts participates in the graph.")

            # normalize grad 
            grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-12)
            grad = grad + momentum * decay
            momentum = grad.detach()

            with torch.no_grad():
                perts.add_(step_size * grad.sign())
                perts.clamp_(-epsilon, epsilon)
                perts.copy_((x_unnorm + perts).clamp(0.0, 1.0) - x_unnorm)

        with torch.no_grad():
            x_adv = (x_unnorm + perts).clamp(0.0, 1.0)
            x_adv = (x_adv - mu_t) / std_t

        return x_adv.detach()

    finally:
        for h in handles:
            h.remove()