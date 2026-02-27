import math
import numpy as np
import torch
import torch.nn.functional as F

from attacks.utils import clamp, PCGrad
from adapters.interfaces import TransformerClassifier


def _infer_special_tokens_from_attn(N: int) -> int:

    for specials in (1, 2):
        P = N - specials
        if P > 0:
            g = int(math.isqrt(P))
            if g * g == P:
                return specials
    return 1

#TODO: add tqdm
def PatchFool(
        model: TransformerClassifier,  
        x: torch.Tensor,
        y: torch.Tensor = None,
        *,
        mu=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        patch_size=16,
        num_patch=1,
        sparse_pixel_num=0,
        patch_select="Attn",
        attack_mode="CE_loss",
        atten_select=4,
        atten_loss_weight=0.002,
        iters=250,
        learnable_mask_stop=200,
        lr=0.22,
        step_size=10,
        gamma=0.95,
        mild_l_inf=0.0,
        mild_l_2=0.0,
        device=None,
):
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
    delta = torch.zeros_like(x, device=device, requires_grad=True)

    logits0, atten0 = model.logits_and_attn(x + delta)
    loss0 = criterion(logits0, y) # initial loss

    N_tokens = atten0[0].size(-1)
    specials = _infer_special_tokens_from_attn(N_tokens)

    # -------------------------
    # Patch selection
    # -------------------------
    if patch_select == 'Rand':
        max_patch_index = np.random.randint(
            0, patch_num_per_line * patch_num_per_line, (x.size(0), num_patch)
        )
        max_patch_index = torch.from_numpy(max_patch_index).to(device)
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

            if sparse_pixel_num != 0:
                learnable_mask.data[j, :, row:row + patch_size, column:column + patch_size] = torch.rand(
                    (patch_size, patch_size), device=device, dtype=x.dtype
                )
            mask[j, :, row:row + patch_size, column:column + patch_size] = 1

    # -------------------------
    # adv attack prep
    # -------------------------
    max_patch_index_matrix = max_patch_index[:, 0]
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
        learnable_mask.requires_grad = True

    delta = delta.to(device)
    delta.requires_grad = True

    opt = torch.optim.Adam([delta], lr=lr)
    mask_opt = None
    if sparse_pixel_num != 0:
        mask_opt = torch.optim.Adam([learnable_mask], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    target_shift = specials

    # -------------------------
    # Attack optimization loop
    # -------------------------
    for iter in range(iters):
        opt.zero_grad(set_to_none=True)
        if sparse_pixel_num != 0 and mask_opt is not None:
            mask_opt.zero_grad(set_to_none=True)

        if sparse_pixel_num != 0:
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
                grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
                mask_grad = torch.autograd.grad(loss, learnable_mask)[0]
            else:
                grad = torch.autograd.grad(loss, delta)[0]

        # gradient ascent on loss
        delta.grad = -grad
        opt.step()
        scheduler.step()

        if sparse_pixel_num != 0 and iter < learnable_mask_stop and mask_opt is not None:
            learnable_mask.grad = -mask_grad
            mask_opt.step()

            learnable_mask_temp = learnable_mask.view(x.size(0), -1)
            learnable_mask.data -= learnable_mask_temp.min(-1)[0].view(-1, 1, 1, 1)
            learnable_mask.data += 1e-6
            learnable_mask.data *= mask

        # (Your original code did NOT enforce mild_l_inf / mild_l_2 inside the loop;
        # if you want strict congruence, keep it that way. If you intended constraints,
        # you can add them here.)
        #TODO

    # -------------------------
    # Create adversarial example
    # -------------------------
    with torch.no_grad():
        if sparse_pixel_num == 0:
            x_adv = x + torch.mul(delta, mask)
        else:
            # TODO
            x_adv = None


    return x_adv, mask