from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F

try: # optional dependency for progress bar
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from tokenfool.attacks.utils import clamp, PCGrad
from tokenfool.adapters.interfaces import TransformerClassifier


def _infer_special_tokens_from_attn(N: int) -> int:

    for specials in (1, 2):
        P = N - specials
        if P > 0:
            g = int(math.isqrt(P))
            if g * g == P:
                return specials
    return 1


def PatchFool(
        model: TransformerClassifier,  
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mu: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        patch_size: int = 16,
        num_patch: int = 1,
        sparse_pixel_num: int = 0,
        patch_select: str = "Attn",
        attack_mode: str = "CE_loss",
        atten_select: int = 4,
        atten_loss_weight: float = 0.002,
        iters: int = 250,
        learnable_mask_stop: int = 200,
        lr: float = 0.22,
        step_size: int = 10,
        gamma: float = 0.95,
        mild_l_inf: float = 0.0,
        mild_l_2: float = 0.0,
        device: Optional[torch.device] = None,
        progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform the Patch-Fool adversarial attack on a Transformer-based classifier, taken and adapted from the original implementation at:
    https://github.com/GATECH-EIC/Patch-Fool/blob/main/main.py

    PatchFool is a patch-level adversarial attack designed for Vision
    Transformers. It selects one or more image patches and optimizes a 
    perturbation confined to those patches. Optionally, it supports sparse pixel selection within patches and attention-guided gradient alignment (PCGrad).

    The attack performs gradient ascent on cross-entropy loss, optionally
    combined with an attention-based loss, under optional L2 or L-inf 
    constraints in normalized input space.

    Parameters
    ----------
    model : TransformerClassifier
        Model implementing the TransformerClassifier protocol:
            - logits(x) -> (B, C)
            - logits_and_attn(x) -> (logits, attn_list)
        where each attention tensor has shape (B, heads, N, N).

    x : torch.Tensor
        Input tensor of shape (B, C, H, W), normalized using (mu, std).

    y : torch.Tensor, optional
        Ground-truth labels of shape (B,). If None, model predictions are used
        as pseudo-labels.

    mu : tuple[float]
        Channel-wise dataset mean used for normalization.

    std : tuple[float]
        Channel-wise dataset standard deviation used for normalization.

    patch_size : int
        Spatial size (in pixels) of each square patch.

    num_patch : int
        Number of patches to attack.

    sparse_pixel_num : int
        If > 0, enables sparse pixel attack within selected patches using a
        learnable mask that selects the top-k pixels.

    patch_select : {"Rand", "Saliency", "Attn"}
        Patch selection strategy:
        - "Rand": random patch indices.
        - "Saliency": largest gradient magnitude over patches.
        - "Attn": highest attention scores from a selected layer.

    attack_mode : {"CE_loss", "Attention"}
        - "CE_loss": optimize cross-entropy only.
        - "Attention": combine CE with attention alignment loss.

    atten_select : int
        Attention layer index used for patch selection (when
        `patch_select="Attn"`).

    atten_loss_weight : float
        Weight applied to attention-based loss terms.

    iters : int
        Number of attack optimization iterations.

    learnable_mask_stop : int
        Iteration at which learnable sparse mask stops updating.

    lr : float
        Learning rate for perturbation optimizer.

    step_size : int
        Step interval for learning rate scheduler.

    gamma : float
        Learning rate decay factor.

    mild_l_inf : float
        L-inf constraint radius. Applied per-channel in
        normalized space as epsilon / std. Set to 0 to disable.

    mild_l_2 : float
        L2 constraint radius. Applied per-channel in
        normalized space as radius / std. Set to 0 to disable.

    device : torch.device, optional
        Device for computation. Defaults to `x.device`.
    
    progress : bool
        If True, display a progress bar during attack iterations.

    Returns
    -------
    x_adv : torch.Tensor
        Adversarial examples of shape (B, C, H, W).

    mask : torch.Tensor
        Binary mask of shape (B, 1, H, W) indicating perturbed patch regions.

    """
    # parameter validation
    ph, pw = model.native_patch_size
    if attack_mode == "Attention" and (patch_size != ph or patch_size != pw):
        raise ValueError(
            f"Attention mode requires attack patch size to match native model patch size "
            f"{(ph, pw)}; got {(patch_size, patch_size)}."
        )
    
    if device is None:
        device = x.device
    x = x.to(device)

    # Use model prediction as gt
    if y is None:
        with torch.no_grad():
            y = model.logits(x).argmax(dim=1)
    y = y.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    mu_t = torch.tensor(mu, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    patch_num_per_line = int(x.size(-1) / patch_size)

    # -------------------------
    # Initial forward for patch selection
    # -------------------------
    delta: torch.Tensor = torch.zeros_like(x, device=device, requires_grad=True)

    logits0, atten0 = model.logits_and_attn(x + delta)
    loss0 = criterion(logits0, y) # initial loss

    N_tokens = atten0[0].size(-1)
    specials = _infer_special_tokens_from_attn(N_tokens)

    # -------------------------
    # Patch selection
    # -------------------------
    if patch_select == 'Rand':
        max_patch_index = torch.from_numpy(np.random.randint(
            0, patch_num_per_line * patch_num_per_line, (x.size(0), num_patch)
        )).to(device)
    elif patch_select == 'Saliency':
        filt = torch.ones((1, 3, patch_size, patch_size), device=device, dtype=x.dtype)
        grad = torch.autograd.grad(loss0, delta, retain_graph=False)[0]
        grad = torch.abs(grad)
        patch_grad = F.conv2d(grad, filt, stride=patch_size)
        patch_grad = patch_grad.view(patch_grad.size(0), -1)
        max_patch_index = patch_grad.argsort(descending=True)[:, :num_patch]
    elif patch_select == 'Attn':
        atten_layer = atten0[atten_select].mean(dim=1)   
        atten_layer = atten_layer.mean(dim=-2)          
        atten_layer = atten_layer[:, specials:]         
        max_patch_index = atten_layer.argsort(descending=True)[:, :num_patch]
    else:
        raise ValueError(f'Unknown patch_select: {patch_select}')

    # -------------------------
    # Build patch mask (and optional learnable mask)
    # -------------------------
    mask = torch.zeros((x.size(0), 1, x.size(2), x.size(3)), device=device, dtype=x.dtype)
    learnable_mask = None
    if sparse_pixel_num != 0:
        learnable_mask = mask.clone()

    for j in range(x.size(0)):
        index_list = max_patch_index[j]
        for index in index_list:
            row = (index // patch_num_per_line) * patch_size
            column = (index % patch_num_per_line) * patch_size
            row = int(row.item()) if torch.is_tensor(row) else int(row)
            column = int(column.item()) if torch.is_tensor(column) else int(column)

            if sparse_pixel_num != 0 and learnable_mask is not None:
                learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                    (patch_size, patch_size), device=device, dtype=x.dtype
                )
            mask[j, :, row:row + patch_size, column:column + patch_size] = 1

    # -------------------------
    # adv attack prep
    # -------------------------
    max_patch_index_matrix = max_patch_index[:, 0].to(device=device, dtype=torch.long)    
    max_patch_index_matrix = max_patch_index_matrix.repeat(N_tokens, 1)
    max_patch_index_matrix = max_patch_index_matrix.permute(1, 0).flatten().long()

    if mild_l_inf == 0:
        delta = (torch.rand_like(x) - mu_t) / std_t
    else:
        epsilon = mild_l_inf / std_t
        delta = 2 * epsilon * torch.rand_like(x) - epsilon + x

    delta.data = clamp(delta, (0 - mu_t) / std_t, (1 - mu_t) / std_t)
    original_img = x.clone()

    if sparse_pixel_num == 0:
        x = torch.mul(x, 1 - mask)
    else:
        if learnable_mask is not None:
            learnable_mask.requires_grad = True

    delta = delta.to(device)
    delta.requires_grad = True

    opt = torch.optim.Adam([delta], lr=lr)
    mask_opt = None
    if sparse_pixel_num != 0:
        if learnable_mask is None:
            raise ValueError("sparse_pixel_num > 0 but learnable_mask is None")
        mask_opt = torch.optim.Adam([learnable_mask], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    target_shift = specials

    iterator = range(iters)
    if progress and tqdm is not None:
        iterator = tqdm(iterator, desc="PatchFool")

    # -------------------------
    # Attack optimization loop
    # -------------------------
    for iter in iterator:
        opt.zero_grad(set_to_none=True)
        if sparse_pixel_num != 0 and mask_opt is not None:
            mask_opt.zero_grad(set_to_none=True)

        if sparse_pixel_num != 0 and learnable_mask is not None:
            if iter < learnable_mask_stop:
                sparse_mask = torch.zeros_like(mask)
                learnable_mask_temp = learnable_mask.view(learnable_mask.size(0), -1)
                sparse_mask_temp = sparse_mask.view(sparse_mask.size(0), -1)
                value, _ = learnable_mask_temp.sort(descending=True)
                threshold = value[:, sparse_pixel_num - 1].view(-1, 1)
                sparse_mask_temp[learnable_mask_temp >= threshold] = 1

                temp_mask = ((sparse_mask - learnable_mask).detach() + learnable_mask) * mask
            else:
                temp_mask = sparse_mask

            x_iter = original_img * (1 - sparse_mask)
            out, atten = model.logits_and_attn(x_iter + torch.mul(delta, temp_mask))
        else:
            out, atten = model.logits_and_attn(x + torch.mul(delta, mask))

        loss = criterion(out, y)

        if attack_mode == 'Attention':
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            ce_loss_grad_temp = grad.view(x.size(0), -1).detach().clone()

            if sparse_pixel_num != 0 and iter < learnable_mask_stop:
                if learnable_mask is None:
                    raise ValueError("sparse_pixel_num > 0 but learnable_mask is None")
                mask_grad = torch.autograd.grad(loss, learnable_mask, retain_graph=True)[0]

            range_list = range(len(atten) // 2)
            for atten_num in range_list:
                if atten_num == 0:
                    continue

                atten_map = atten[atten_num].mean(dim=1)          
                atten_map = atten_map.view(-1, atten_map.size(-1)) 
                atten_map = -torch.log(atten_map.clamp_min(1e-12))
                atten_loss = F.nll_loss(atten_map, max_patch_index_matrix + target_shift)

                atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]
                atten_grad_temp = atten_grad.view(x.size(0), -1)
                cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

                if sparse_pixel_num != 0 and iter < learnable_mask_stop:
                    if learnable_mask is None:
                        raise ValueError("sparse_pixel_num > 0 but learnable_mask is None")
                    mask_atten_grad = torch.autograd.grad(atten_loss, learnable_mask, retain_graph=True)[0]

                atten_grad = PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)

                if sparse_pixel_num != 0 and iter < learnable_mask_stop:
                    mask_atten_grad_temp = mask_atten_grad.view(mask_atten_grad.size(0), -1)
                    ce_mask_grad_temp = mask_grad.view(mask_grad.size(0), -1).detach().clone()
                    mask_cos_sim = F.cosine_similarity(mask_atten_grad_temp, ce_mask_grad_temp, dim=1)
                    mask_atten_grad = PCGrad(
                        mask_atten_grad_temp, ce_mask_grad_temp, mask_cos_sim, mask_atten_grad.shape
                    )
                    mask_grad = mask_grad + mask_atten_grad * atten_loss_weight

                grad = grad + atten_grad * atten_loss_weight

        else:
            # no attention loss
            if sparse_pixel_num != 0 and iter < learnable_mask_stop:
                if learnable_mask is None:
                    raise ValueError("sparse_pixel_num > 0 but learnable_mask is None")
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                mask_grad = torch.autograd.grad(loss, learnable_mask)[0]
            else:
                grad = torch.autograd.grad(loss, delta)[0]

        # gradient ascent on loss
        delta.grad = -grad
        opt.step()
        scheduler.step()

        if sparse_pixel_num != 0 and learnable_mask is not None and iter < learnable_mask_stop and mask_opt is not None:
            learnable_mask.grad = -mask_grad
            mask_opt.step()

            learnable_mask_temp = learnable_mask.view(x.size(0), -1)
            learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
            learnable_mask.data += 1e-6
            learnable_mask.data *= mask

        # l2 constraint
        if mild_l_2 != 0.0:
            radius = (mild_l_2 / std_t).view(1, x.size(1), 1, 1)  
            perturbation = (delta.detach() - original_img) * mask  
            l2 = torch.linalg.norm(perturbation.view(perturbation.size(0), perturbation.size(1), -1), dim=-1)  
            radius_bc = radius.view(1, x.size(1)).repeat(l2.size(0), 1)  
            scale = radius_bc / (l2 + 1e-12)
            scale[l2 < radius_bc] = 1.0
            scale = scale.view(scale.size(0), scale.size(1), 1, 1) 

            delta.data = original_img + perturbation * scale

        # l_inf constraint
        if mild_l_inf != 0.0:
            epsilon = mild_l_inf / std_t 
            delta.data = clamp(delta, original_img - epsilon, original_img + epsilon)

        delta.data = clamp(delta, (0 - mu_t) / std_t, (1 - mu_t) / std_t)

    # -------------------------
    # Create adversarial example
    # -------------------------
    with torch.no_grad():
        if sparse_pixel_num == 0:
            x_adv = x + torch.mul(delta, mask)
        else:
            # TODO
            raise NotImplementedError("Sparse pixel attack not fully implemented yet.")


    return x_adv, mask