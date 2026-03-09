import torch
import torch.nn as nn
import pytest

from tokenfool.attacks.att import ATT

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class MockAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim)
        self.attn_drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn_drop(self.qkv(x))


class MockBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = MockAttn(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TinyBackbone(nn.Module):
    def __init__(self, image_size=32, patch_size=8, dim=8, num_classes=5, num_prefix_tokens=1):
        super().__init__()
        assert image_size % patch_size == 0
        grid = image_size // patch_size
        num_patches = grid * grid
        patch_dim = 3 * patch_size * patch_size

        self.num_prefix_tokens = num_prefix_tokens
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_dim, dim)
        self.prefix = nn.Parameter(torch.zeros(1, num_prefix_tokens, dim))
        self.blocks = nn.ModuleList([MockBlock(dim), MockBlock(dim)])
        self.head = nn.Linear(dim, num_classes)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        g = H // p
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, g * g, C * p * p)
        patch_tokens = self.patch_proj(patches)
        prefix = self.prefix.expand(B, -1, -1)
        return torch.cat([prefix, patch_tokens], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, 0])


class DummyHookableViT:
    def __init__(self, backbone: TinyBackbone):
        self.model = backbone

    @property
    def num_prefix_tokens(self) -> int:
        return self.model.num_prefix_tokens

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def hook_modules(self):
        return {
            "attn_drop": [blk.attn.attn_drop for blk in self.model.blocks],
            "qkv": [blk.attn.qkv for blk in self.model.blocks],
            "mlp": [blk.mlp for blk in self.model.blocks],
        }

    def att_feature_module(self):
        return self.model.blocks[-2]
    


@pytest.mark.parametrize("patch_out", [False, True])
@pytest.mark.parametrize("num_prefix_tokens", [1, 2])
def test_att_shapes_and_bounds(patch_out, num_prefix_tokens):
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, C, H, W = 2, 3, 32, 32
    x = torch.rand(B, C, H, W, device=device)

    backbone = TinyBackbone(
        image_size=H,
        patch_size=8,
        dim=32,
        num_classes=7,
        num_prefix_tokens=num_prefix_tokens,
    ).to(device)
    model = DummyHookableViT(backbone)

    x_adv = ATT(
        model,
        x,
        y=None,
        epsilon=8 / 255,
        iters=3,
        decay=1.0,
        patch_out=patch_out,
        patch_size=8,
        image_size=H,
        progress=False,
    )

    assert x_adv.shape == x.shape
    assert torch.isfinite(x_adv).all()

    mu_t = torch.tensor(MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    lo = (0 - mu_t) / std_t
    hi = (1 - mu_t) / std_t

    assert torch.all(x_adv >= lo - 1e-6)
    assert torch.all(x_adv <= hi + 1e-6)


def test_att_changes_input():
    torch.manual_seed(0)
    device = torch.device("cpu")

    x = torch.rand(1, 3, 32, 32, device=device)

    backbone = TinyBackbone(
        image_size=32,
        patch_size=8,
        dim=32,
        num_classes=5,
        num_prefix_tokens=1,
    ).to(device)
    model = DummyHookableViT(backbone)

    x_adv = ATT(
        model,
        x,
        epsilon=8 / 255,
        iters=3,
        patch_out=True,
        patch_size=8,
        image_size=32,
        progress=False,
    )

    diff = (x_adv - x).abs().max().item()
    assert diff > 0.0


def test_att_targeted_smoke():
    torch.manual_seed(0)
    device = torch.device("cpu")

    x = torch.rand(2, 3, 32, 32, device=device)
    target = torch.tensor([1, 2], device=device)

    backbone = TinyBackbone(
        image_size=32,
        patch_size=8,
        dim=32,
        num_classes=5,
        num_prefix_tokens=1,
    ).to(device)
    model = DummyHookableViT(backbone)

    x_adv = ATT(
        model,
        x,
        y=target,
        epsilon=8 / 255,
        iters=2,
        targeted=True,
        patch_out=True,
        patch_size=8,
        image_size=32,
        progress=False,
    )

    assert x_adv.shape == x.shape
    assert torch.isfinite(x_adv).all()


def test_att_respects_pixel_epsilon_in_unnormalized_space():
    torch.manual_seed(0)
    device = torch.device("cpu")

    x = torch.rand(1, 3, 32, 32, device=device)

    backbone = TinyBackbone(
        image_size=32,
        patch_size=8,
        dim=32,
        num_classes=5,
        num_prefix_tokens=1,
    ).to(device)
    model = DummyHookableViT(backbone)

    x_adv = ATT(
        model,
        x,
        epsilon=8 / 255,
        iters=3,
        patch_out=True,
        patch_size=8,
        image_size=32,
        progress=False,
    )

    mu_t = torch.tensor(MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x_pix = (x * std_t + mu_t).clamp(0, 1)
    x_adv_pix = (x_adv * std_t + mu_t).clamp(0, 1)

    diff = (x_adv_pix - x_pix).abs()
    assert torch.all(diff <= 8 / 255 + 1e-5)


def test_att_missing_hook_methods_raises():
    class BadModel:
        def logits(self, x):
            return x.mean(dim=(2, 3))

    x = torch.rand(1, 3, 32, 32)

    with pytest.raises(TypeError):
        ATT(BadModel(), x, iters=1, progress=False)