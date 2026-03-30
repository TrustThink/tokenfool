import pytest
import torch

from tokenfool.adapters.timm_vit import TimmViTAdapter

pytestmark = pytest.mark.integration


def test_timm_vit_smoke_tokens_and_attn():
    timm = pytest.importorskip("timm")

    model = timm.create_model("vit_small_patch16_224", pretrained=False).eval()
    adapter = TimmViTAdapter(model)
    x = torch.randn(1, 3, 224, 224)

    tokens, attn = adapter.tokens_and_attn(x)

    assert tokens.shape == (1, 197, model.embed_dim)
    assert len(attn) == len(adapter.model.blocks)
    assert attn[0].shape[-2:] == (197, 197)
    assert adapter.num_prefix_tokens == 1
    assert adapter.native_patch_size == (16, 16)


def test_timm_deit_distilled_prefix_tokens():
    timm = pytest.importorskip("timm")

    model = timm.create_model("deit_small_distilled_patch16_224", pretrained=False).eval()
    adapter = TimmViTAdapter(model)
    x = torch.randn(1, 3, 224, 224)

    tokens, attn = adapter.tokens_and_attn(x)

    assert tokens.shape[1] == 198
    assert attn[0].shape[-1] == 198
    assert adapter.num_prefix_tokens == 2
    assert adapter.native_patch_size == (16, 16)