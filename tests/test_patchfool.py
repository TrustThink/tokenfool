import math
import torch
import torch.nn as nn
import pytest

from tokenfool.attacks.patchfool import PatchFool, _infer_special_tokens_from_attn

class DummyViT(nn.Module):
    """
    Minimal differentiable model TransformerClassifier for tests
    """
    def __init__(self, num_classes=10, heads=2, grid=4, specials=1, layers=6):
        super().__init__()
        self.num_classes = num_classes
        self.heads = heads
        self.grid = grid
        self.specials = specials
        self.N = specials + grid * grid
        self.layers = layers
        self.proj = nn.Linear(3, num_classes)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        feat = x.mean(dim=(2, 3))
        return self.proj(feat)

    def logits_and_attn(self, x: torch.Tensor):
        logits = self.logits(x)
        B = x.size(0)
        device = x.device
        base = x.mean(dim=(2, 3), keepdim=False)  
        s = base.sum(dim=1, keepdim=True)         
        A = torch.sigmoid(s).view(B, 1, 1, 1) * torch.ones(B, self.heads, self.N, self.N, device=device)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)  

        attn_list = [A for _ in range(self.layers)]
        return logits, attn_list


def normalized_bounds(mu, std, device):
    mu_t = torch.tensor(mu, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    lo = (0 - mu_t) / std_t
    hi = (1 - mu_t) / std_t
    return lo, hi, std_t


def test_infer_special_tokens_basic():
    assert _infer_special_tokens_from_attn(197) == 1
    assert _infer_special_tokens_from_attn(198) == 2


@pytest.mark.parametrize("patch_select", ["Rand", "Saliency", "Attn"])
def test_patchfool_shapes_and_bounds(patch_select):
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, C, H, W = 2, 3, 64, 64
    x = torch.rand(B, C, H, W, device=device)
    model = DummyViT(grid=H // 16, specials=1).to(device)

    x_adv, mask = PatchFool(
        model,
        x,
        y=None,
        patch_size=16,
        num_patch=1,
        sparse_pixel_num=0,
        patch_select=patch_select,
        attack_mode="CE_loss",
        iters=3,          # fast
        lr=0.1,
        mild_l_inf=0.0,
        mild_l_2=0.0,
        device=device,
    )

    assert x_adv.shape == x.shape
    assert mask.shape == (B, 1, H, W)

    # bounds check
    lo, hi, _ = normalized_bounds((0.485,0.456,0.406), (0.229,0.224,0.225), device)
    assert torch.all(x_adv >= lo - 1e-6)
    assert torch.all(x_adv <= hi + 1e-6)


def test_patchfool_mask_locality_sparse0():
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, C, H, W = 1, 3, 64, 64
    x = torch.rand(B, C, H, W, device=device)
    model = DummyViT(grid=H // 16, specials=1).to(device)

    x_adv, mask = PatchFool(
        model,
        x,
        y=None,
        patch_size=16,
        num_patch=1,
        sparse_pixel_num=0,
        patch_select="Rand",
        attack_mode="CE_loss",
        iters=3,
        lr=0.1,
        device=device,
    )

    diff = (x_adv - x).abs()
    outside = diff * (1 - mask)
    assert outside.max().item() <= 1e-5


def test_patchfool_linf_constraint():
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, C, H, W = 1, 3, 64, 64
    x = torch.rand(B, C, H, W, device=device)
    model = DummyViT(grid=H // 16, specials=1).to(device)

    mild_l_inf = 0.05
    x_adv, mask = PatchFool(
        model,
        x,
        y=None,
        patch_size=16,
        num_patch=1,
        sparse_pixel_num=0,
        patch_select="Rand",
        attack_mode="CE_loss",
        iters=5,
        lr=0.2,
        mild_l_inf=mild_l_inf,
        mild_l_2=0.0,
        device=device,
    )

    _, _, std_t = normalized_bounds((0.485,0.456,0.406), (0.229,0.224,0.225), device)
    eps = mild_l_inf / std_t
    diff = (x_adv - x).abs()
    assert torch.all(diff <= eps + 1e-5)


def test_patchfool_l2_constraint():
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, C, H, W = 1, 3, 64, 64
    x = torch.rand(B, C, H, W, device=device)
    model = DummyViT(grid=H // 16, specials=1).to(device)

    mild_l_2 = 1.0
    x_adv, mask = PatchFool(
        model,
        x,
        y=None,
        patch_size=16,
        num_patch=1,
        sparse_pixel_num=0,
        patch_select="Rand",
        attack_mode="CE_loss",
        iters=5,
        lr=0.2,
        mild_l_inf=0.0,
        mild_l_2=mild_l_2,
        device=device,
    )

    _, _, std_t = normalized_bounds((0.485,0.456,0.406), (0.229,0.224,0.225), device)
    radius = (mild_l_2 / std_t).view(1, C)  

    pert = (x_adv - x) * mask
    l2 = torch.linalg.norm(pert.view(B, C, -1), dim=-1)  
    assert torch.all(l2 <= radius + 1e-4)