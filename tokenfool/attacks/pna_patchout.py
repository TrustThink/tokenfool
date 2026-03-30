from typing import Dict, List, Optional, Tuple

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from tokenfool.adapters.interfaces import HookableTransformerClassifier
from tokenfool.attacks.utils import clamp



def _infer_num_image_patches(
    height: int,
    width: int,
    patch_size: int,
) -> Tuple[int, int, int]:
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Input spatial size ({height}, {width}) must be divisible by patch_size={patch_size}."
        )
    gh = height // patch_size
    gw = width // patch_size
    return gh, gw, gh * gw


def _sample_patch_mask(
    batch_size: int,
    height: int,
    width: int,
    patch_size: int,
    num_patches: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    gh, gw, total = _infer_num_image_patches(height, width, patch_size)
    if num_patches <= 0:
        raise ValueError(f"num_patches must be > 0, got {num_patches}")
    if num_patches > total:
        raise ValueError(
            f"num_patches={num_patches} exceeds total number of patches={total}."
        )

    mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    for b in range(batch_size):
        patch_ids = torch.randperm(total, device=device)[:num_patches]
        rows = torch.div(patch_ids, gw, rounding_mode="floor")
        cols = patch_ids % gw
        for r, c in zip(rows.tolist(), cols.tolist()):
            h0 = r * patch_size
            w0 = c * patch_size
            mask[b, :, h0 : h0 + patch_size, w0 : w0 + patch_size] = 1.0
    return mask


def _zero_attention_grad_hook(module, grad_in, grad_out):
    if len(grad_in) == 0 or grad_in[0] is None:
        return None
    new0 = torch.zeros_like(grad_in[0])
    if len(grad_in) == 1:
        return (new0,)
    return (new0, *grad_in[1:])


def PNAPatchOut(
    model: HookableTransformerClassifier,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    *,
    epsilon: float = 16 / 255,
    iters: int = 10,
    patch_size: int = 16,
    num_patches: int = 130,
    lam: float = 0.1,
    mu: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    targeted: bool = False,
    momentum: float = 0.0,
    device: Optional[torch.device] = None,
    progress: bool = False,
) -> torch.Tensor:
    """
    Perform the PNA + PatchOut attack from:
    "Boosting Adversarial Transferability on Vision Transformer with PatchOut and Pay No Attention".
    Taken and adapted from the official implementation at
    https://github.com/zhipeng-wei/PNA-PatchOut/blob/master/our_method.py

    This attack:
      1. applies PNA by stopping gradients through attention-drop modules; and
      2. applies PatchOut by sampling a fresh random subset of image patches each iteration.

    Parameters
    ----------
    model : HookableTransformerClassifier
        Transformer classifier adapter exposing internal modules required for gradient hooks.
        Must implement:
        - hook_modules() -> Dict[str, List[nn.Module]]
    x : torch.Tensor
        Input tensor of shape (B, C, H, W), normalized using (mu, std).
    y : torch.Tensor, optional
        Ground-truth labels of shape (B,). If None, model predictions are used as pseudo-labels.
    epsilon : float
        Maximum perturbation magnitude in pixel space.
    iters : int
        Number of attack iterations.
    patch_size : int
        Spatial size of each image patch.
    num_patches : int
        Number of randomly sampled patches used per iteration.
    lam : float
        Weight applied to the L2 regularization term on the perturbation.
    mu : tuple[float]
        Dataset mean used for normalization.
    std : tuple[float]
        Dataset std used for normalization.
    targeted : bool
        If True, minimize cross-entropy toward y. Otherwise maximize it away from y.
    momentum : float
        Optional momentum factor for iterative sign updates.
    device : torch.device, optional
        Device for computation. Defaults to x.device.
    progress : bool
        If True, display a progress bar.

    Returns
    -------
    x_adv : torch.Tensor
        Adversarial examples of shape (B, C, H, W).
    """
    required = ("hook_modules",)
    missing = [m for m in required if not hasattr(model, m)]
    if missing:
        raise TypeError(
            f"PNAPatchOut requires HookableTransformerClassifier methods: missing {missing}"
        )

    if device is None:
        device = x.device

    x = x.to(device)
    dtype = x.dtype
    batch_size, channels, height, width = x.shape

    if y is None:
        with torch.no_grad():
            y = model.logits(x).argmax(dim=1)
    y = y.to(device)

    mu_t = torch.tensor(mu, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    lo = (0.0 - mu_t) / std_t
    hi = (1.0 - mu_t) / std_t
    eps = epsilon / std_t

    loss_fn = torch.nn.CrossEntropyLoss()
    direction = -1.0 if targeted else 1.0

    mods: Dict[str, List[torch.nn.Module]] = model.hook_modules()
    if "attn_probs_drop" not in mods:
        raise TypeError('hook_modules() must provide key "attn_probs_drop" for PNAPatchOut.')

    handles: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for m in mods["attn_probs_drop"]:
            handles.append(m.register_full_backward_hook(_zero_attention_grad_hook))

        delta = torch.zeros_like(x, device=device)
        velocity = torch.zeros_like(x, device=device)
        step_size = eps / float(iters)

        iterator = range(iters)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="PNA-PatchOut")

        for _ in iterator:
            patch_mask = _sample_patch_mask(
                batch_size,
                height,
                width,
                patch_size,
                num_patches,
                device=device,
                dtype=dtype,
            )

            delta = delta.detach().requires_grad_(True)
            x_adv = clamp(x + patch_mask * delta, lo, hi)

            logits = model.logits(x_adv)
            ce = loss_fn(logits, y)
            l2 = torch.linalg.norm(delta.view(batch_size, -1), dim=1).mean()

            loss = direction * ce + lam * l2

            model.zero_grad(set_to_none=True)
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            grad = delta.grad
            if grad is None:
                raise RuntimeError(
                    "PNAPatchOut: delta.grad is None; check model adapter and graph connectivity."
                )

            if momentum != 0.0:
                grad = grad / (
                    grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
                )
                velocity = momentum * velocity + grad
                grad = velocity

            with torch.no_grad():
                delta = delta + step_size * grad.sign()
                delta = clamp(delta, -eps, eps)
                delta = clamp(x + delta, lo, hi) - x

        x_adv = clamp(x + delta, lo, hi)
        return x_adv.detach()

    finally:
        for h in handles:
            h.remove()