import copy

import pytest
import torch

from tests.helpers import FakeViT, NotAViT
from tokenfool.adapters.timm_vit import TimmViTAdapter


@pytest.fixture
def fake_vit():
    return FakeViT().eval()


@pytest.fixture
def fake_deit_distilled():
    return FakeViT(distilled=True).eval()


def test_adapter_deepcopies_model(fake_vit):
    original = fake_vit
    adapter = TimmViTAdapter(original)
    assert adapter.model is not original
    assert isinstance(adapter.model, FakeViT)


def test_logits_matches_logits_and_attn_logits(fake_vit):
    x = torch.randn(2, 3, 32, 32)
    adapter = TimmViTAdapter(fake_vit)

    got = adapter.logits(x)
    want, _ = adapter.logits_and_attn(x)

    assert got.shape == (2, 5)
    assert torch.allclose(got, want)


def test_tokens_and_attn_shapes(fake_vit):
    x = torch.randn(2, 3, 32, 32)
    adapter = TimmViTAdapter(fake_vit)

    tokens, attn = adapter.tokens_and_attn(x)

    assert tokens.shape[:2] == (2, 17)  # 1 cls + 16 spatial
    assert len(attn) == len(adapter.model.blocks)
    for a in attn:
        assert a.shape == (2, 2, 17, 17)
        assert a.requires_grad


def test_logits_and_attn_shapes(fake_vit):
    x = torch.randn(2, 3, 32, 32)
    adapter = TimmViTAdapter(fake_vit)

    logits, attn = adapter.logits_and_attn(x)

    assert logits.shape == (2, 5)
    assert len(attn) == len(adapter.model.blocks)
    for a in attn:
        assert a.shape == (2, 2, 17, 17)
        assert a.requires_grad


def test_attention_capture_is_live_for_grad(fake_vit):
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    adapter = TimmViTAdapter(fake_vit)

    _, attn = adapter.logits_and_attn(x)
    loss = -torch.log(
        attn[0].mean(dim=1).view(-1, attn[0].size(-1)).clamp_min(1e-12)
    ).mean()
    grad = torch.autograd.grad(loss, x, retain_graph=False)[0]

    assert grad is not None
    assert grad.shape == x.shape


def test_num_prefix_tokens_regular(fake_vit):
    adapter = TimmViTAdapter(fake_vit)
    assert adapter.num_prefix_tokens == 1


def test_num_prefix_tokens_distilled(fake_deit_distilled):
    adapter = TimmViTAdapter(fake_deit_distilled)
    assert adapter.num_prefix_tokens == 2


def test_native_patch_size(fake_vit):
    adapter = TimmViTAdapter(fake_vit)
    assert adapter.native_patch_size == (8, 8)


def test_hook_modules_contains_expected_keys(fake_vit):
    adapter = TimmViTAdapter(fake_vit)
    hooks = adapter.hook_modules()

    assert set(hooks) == {"attn_probs_drop", "attn_proj", "ffn"}
    assert len(hooks["attn_probs_drop"]) == len(adapter.model.blocks)
    assert len(hooks["attn_proj"]) == len(adapter.model.blocks)
    assert len(hooks["ffn"]) == len(adapter.model.blocks)


def test_att_feature_module_returns_last_block(fake_vit):
    adapter = TimmViTAdapter(fake_vit)
    mod = adapter.att_feature_module()

    assert type(mod) is type(adapter.model.blocks[-1])


def test_patching_is_idempotent(fake_vit):
    adapter = TimmViTAdapter(fake_vit)
    before = copy.deepcopy(adapter.model.state_dict())

    adapter._patch_if_needed()
    adapter._patch_if_needed()

    after = adapter.model.state_dict()
    assert before.keys() == after.keys()


def test_non_vit_model_constructs_but_patch_dependent_methods_fail():
    adapter = TimmViTAdapter(NotAViT())
    x = torch.randn(1, 4)

    with pytest.raises(Exception):
        adapter.tokens_and_attn(x)