import torch
import pytest
import timm

from tokenfool.adapters.timm_vit import TimmViTAdapter


@pytest.fixture
def model():
    return timm.create_model("vit_small_patch16_224", pretrained=False).eval()


@pytest.fixture
def adapter(model):
    return TimmViTAdapter(model)


@pytest.fixture
def x():
    return torch.randn(2, 3, 224, 224)


def test_adapter_deepcopies_model(model):
    adapter = TimmViTAdapter(model)
    assert adapter.model is not model


def test_patching_does_not_modify_original_model(model, x):
    adapter = TimmViTAdapter(model)

    orig_forward_func = model.forward.__func__
    orig_forward_self = model.forward.__self__

    orig_ff_func = model.forward_features.__func__
    orig_ff_self = model.forward_features.__self__

    _ = adapter.tokens(x)

    assert model.forward.__func__ is orig_forward_func
    assert model.forward.__self__ is orig_forward_self

    assert model.forward_features.__func__ is orig_ff_func
    assert model.forward_features.__self__ is orig_ff_self


def test_cls_index(adapter):
    assert adapter.cls_index == 0


def test_num_prefix_tokens_standard_vit(adapter):
    assert adapter.num_prefix_tokens == 1


def test_patch_happens_lazily(adapter):
    assert adapter._patched is False


def test_first_functional_call_patches_model(adapter, x):
    _ = adapter.tokens(x)
    assert adapter._patched is True


def test_patch_is_idempotent(adapter, x):
    _ = adapter.tokens(x)
    forward1 = adapter.model.forward
    forward_features1 = adapter.model.forward_features

    _ = adapter.tokens(x)

    assert adapter.model.forward is forward1
    assert adapter.model.forward_features is forward_features1


def test_logits_shape(adapter, x):
    y = adapter.logits(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == x.shape[0]
    assert y.ndim == 2


def test_logits_and_attn_shapes(adapter, x):
    y, attn = adapter.logits_and_attn(x)

    assert isinstance(y, torch.Tensor)
    assert y.ndim == 2

    assert isinstance(attn, list)
    assert len(attn) == len(adapter.model.blocks)

    for a in attn:
        assert isinstance(a, torch.Tensor)
        assert a.ndim == 4
        assert a.shape[0] == x.shape[0]


def test_logits_matches_logits_and_attn(adapter, x):
    y1 = adapter.logits(x)
    y2, _ = adapter.logits_and_attn(x)
    assert torch.allclose(y1, y2)


def test_tokens_shape(adapter, x):
    t = adapter.tokens(x)
    assert isinstance(t, torch.Tensor)
    assert t.shape[0] == x.shape[0]
    assert t.ndim == 3


def test_tokens_and_attn_shapes(adapter, x):
    t, attn = adapter.tokens_and_attn(x)

    assert isinstance(t, torch.Tensor)
    assert t.ndim == 3
    assert t.shape[0] == x.shape[0]

    assert isinstance(attn, list)
    assert len(attn) == len(adapter.model.blocks)

    for a in attn:
        assert isinstance(a, torch.Tensor)
        assert a.ndim == 4
        assert a.shape[0] == x.shape[0]


def test_tokens_matches_tokens_and_attn(adapter, x):
    t1 = adapter.tokens(x)
    t2, _ = adapter.tokens_and_attn(x)
    assert torch.allclose(t1, t2)


def test_attention_is_captured_for_each_block(adapter, x):
    _, attn = adapter.tokens_and_attn(x)
    assert all(a is not None for a in attn)


def test_attention_is_square_over_tokens(adapter, x):
    _, attn = adapter.tokens_and_attn(x)
    for a in attn:
        assert a.shape[-1] == a.shape[-2]


def test_hook_modules_keys(adapter, x):
    _ = adapter.tokens(x)
    hooks = adapter.hook_modules()
    assert set(hooks.keys()) == {"attn_probs_drop", "attn_proj", "ffn"}


def test_hook_modules_lengths_match_block_count(adapter, x):
    _ = adapter.tokens(x)
    hooks = adapter.hook_modules()
    n_blocks = len(adapter.model.blocks)

    assert len(hooks["attn_probs_drop"]) == n_blocks
    assert len(hooks["attn_proj"]) == n_blocks
    assert len(hooks["ffn"]) == n_blocks


def test_hook_modules_are_modules(adapter, x):
    _ = adapter.tokens(x)
    hooks = adapter.hook_modules()

    for group in hooks.values():
        for mod in group:
            assert isinstance(mod, torch.nn.Module)


def test_att_feature_module_returns_penultimate_block(adapter, x):
    _ = adapter.tokens(x)
    mod = adapter.att_feature_module()

    if len(adapter.model.blocks) >= 2:
        assert mod is adapter.model.blocks[-2]
    else:
        assert mod is adapter.model.blocks[-1]


def test_zero_grad(adapter, x):
    y = adapter.logits(x).sum()
    y.backward()

    adapter.zero_grad(set_to_none=True)

    for p in adapter.model.parameters():
        assert p.grad is None


def test_non_vit_model_raises_on_tokens():
    model = timm.create_model("resnet18", pretrained=False).eval()
    adapter = TimmViTAdapter(model)

    with pytest.raises(TypeError):
        adapter.tokens(torch.randn(1, 3, 224, 224))


def test_non_vit_model_raises_on_hook_modules():
    model = timm.create_model("resnet18", pretrained=False).eval()
    adapter = TimmViTAdapter(model)

    with pytest.raises(TypeError):
        adapter.hook_modules()


def test_distilled_deit_num_prefix_tokens():
    model = timm.create_model("deit_small_distilled_patch16_224", pretrained=False).eval()
    adapter = TimmViTAdapter(model)
    assert adapter.num_prefix_tokens == 2


@pytest.mark.parametrize(
    "model_name",
    [
        "vit_small_patch16_224",
        "deit_small_patch16_224",
        "deit_small_distilled_patch16_224",
    ],
)
def test_tokens_and_attn_supported_families(model_name):
    model = timm.create_model(model_name, pretrained=False).eval()
    adapter = TimmViTAdapter(model)
    x = torch.randn(1, 3, 224, 224)

    t, attn = adapter.tokens_and_attn(x)

    assert t.ndim == 3
    assert len(attn) == len(adapter.model.blocks)
    assert all(a is not None for a in attn)