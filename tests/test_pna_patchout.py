import pytest
import torch

from tests.helpers import DummyHookableViT, TinyBackbone
from tokenfool.attacks.pna_patchout import (
    PNAPatchOut,
    _infer_num_image_patches,
    _sample_patch_mask,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def test_infer_num_image_patches_basic():
    gh, gw, total = _infer_num_image_patches(32, 32, 8)
    assert (gh, gw, total) == (4, 4, 16)


def test_infer_num_image_patches_requires_divisible_size():
    with pytest.raises(ValueError, match="divisible"):
        _infer_num_image_patches(30, 32, 8)


def test_sample_patch_mask_shape_and_count():
    torch.manual_seed(0)
    mask = _sample_patch_mask(
        batch_size=2,
        height=32,
        width=32,
        patch_size=8,
        num_patches=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask.shape == (2, 1, 32, 32)
    assert torch.all(mask.sum(dim=(1, 2, 3)) == 3 * 8 * 8)


@pytest.mark.parametrize("num_patches", [0, 17])
def test_sample_patch_mask_validates_patch_count(num_patches):
    with pytest.raises(ValueError):
        _sample_patch_mask(
            batch_size=1,
            height=32,
            width=32,
            patch_size=8,
            num_patches=num_patches,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_pna_patchout_shapes_and_bounds():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    model = DummyHookableViT(TinyBackbone())

    x_adv = PNAPatchOut(
        model,
        x,
        epsilon=8 / 255,
        iters=2,
        patch_size=8,
        num_patches=2,
        lam=0.1,
        progress=False,
    )

    assert x_adv.shape == x.shape
    assert torch.isfinite(x_adv).all()

    mu_t = torch.tensor(MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(STD).view(1, 3, 1, 1)
    lo = (0 - mu_t) / std_t
    hi = (1 - mu_t) / std_t
    assert torch.all(x_adv >= lo - 1e-6)
    assert torch.all(x_adv <= hi + 1e-6)


def test_pna_patchout_returns_tensor_same_shape():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    model = DummyHookableViT(TinyBackbone())

    x_adv = PNAPatchOut(
        model,
        x,
        epsilon=8 / 255,
        iters=2,
        patch_size=8,
        num_patches=2,
        progress=False,
    )

    assert x_adv.shape == x.shape
    assert torch.isfinite(x_adv).all()


def test_pna_patchout_targeted_smoke():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([1, 2])
    model = DummyHookableViT(TinyBackbone(num_classes=5))

    x_adv = PNAPatchOut(
        model,
        x,
        y=y,
        targeted=True,
        epsilon=8 / 255,
        iters=2,
        patch_size=8,
        num_patches=2,
        progress=False,
    )

    assert x_adv.shape == x.shape
    assert torch.isfinite(x_adv).all()


def test_pna_patchout_momentum_smoke():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    model = DummyHookableViT(TinyBackbone())

    x_adv = PNAPatchOut(
        model,
        x,
        epsilon=8 / 255,
        iters=2,
        patch_size=8,
        num_patches=2,
        momentum=0.9,
        progress=False,
    )

    assert x_adv.shape == x.shape


def test_pna_patchout_missing_hook_methods_raises():
    class BadModel:
        def logits(self, x):
            return x.mean(dim=(2, 3))

    with pytest.raises(TypeError, match="missing"):
        PNAPatchOut(BadModel(), torch.rand(1, 3, 32, 32), iters=1, progress=False)


def test_pna_patchout_requires_attn_probs_drop_key():
    class BadHookModel:
        def zero_grad(self, set_to_none=True):
            pass

        def logits(self, x):
            return x.mean(dim=(2, 3))

        def hook_modules(self):
            return {"attn_proj": [], "ffn": []}

    with pytest.raises(TypeError, match="attn_probs_drop"):
        PNAPatchOut(BadHookModel(), torch.rand(1, 3, 32, 32), iters=1, progress=False)