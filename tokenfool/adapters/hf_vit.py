from typing import Dict, List, Tuple, Optional, cast
import copy
import types

import torch
import torch.nn as nn

from tokenfool.adapters.interfaces import HookableTransformerClassifier


class _IdentityDropout(nn.Module):
    """
    Hookable stand-in for attention-probability dropout.

    We intentionally keep this as identity so the adapter does not change model
    semantics just to create a hook point.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HuggingFaceViTAdapter(HookableTransformerClassifier):
    """
    Adapter for Hugging Face ViT / DeiT image-classification models.

    Supported public model families:
      - ViTForImageClassification
      - DeiTForImageClassification
      - DeiTForImageClassificationWithTeacher

    Hook mapping:
      - attn_probs_drop: injected hookable module on attention probabilities
      - attn_proj:       self-attention query projection
      - ffn:             block.intermediate
    """

    def __init__(self, model: nn.Module):
        self.model = copy.deepcopy(model)
        self._validate_model()
        self._patch_attention_blocks()

    @property
    def cls_index(self) -> int:
        return 0
    
    @property
    def native_patch_size(self) -> tuple[int, int]:
        """
        Native image patch size used by the model's patch embedding layer.
        """
        base = self._base_model()
        embeddings = getattr(base, "embeddings", None)
        if embeddings is None:
            raise TypeError("Unsupported HF model: missing embeddings.")

        patch_embeddings = getattr(embeddings, "patch_embeddings", None)
        if patch_embeddings is None:
            raise TypeError("Unsupported HF model: missing embeddings.patch_embeddings.")

        patch_size = getattr(patch_embeddings, "patch_size", None)

        if isinstance(patch_size, int):
            return (patch_size, patch_size)

        if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            h, w = patch_size
            return int(h), int(w)

        # fallback: some configs expose it here
        config_patch_size = getattr(self.model.config, "patch_size", None)
        if isinstance(config_patch_size, int):
            return (config_patch_size, config_patch_size)
        if isinstance(config_patch_size, (tuple, list)) and len(config_patch_size) == 2:
            h, w = config_patch_size
            return int(h), int(w)

        raise TypeError("Could not determine native patch size for HF ViT/DeiT model.")


    @property
    def num_prefix_tokens(self) -> int:
        emb = self._base_model().embeddings
        return 2 if hasattr(emb, "distillation_token") else 1

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.model.zero_grad(set_to_none=set_to_none)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        self._clear_cached_attn()
        outputs = self.model(
            pixel_values=x,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Could not extract logits from Hugging Face model output.")
        return logits

    def logits_and_attn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self._clear_cached_attn()
        outputs = self.model(
            pixel_values=x,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = getattr(outputs, "logits", None)
        attentions = getattr(outputs, "attentions", None)

        if not isinstance(logits, torch.Tensor):
            raise TypeError("Could not extract logits from Hugging Face model output.")

        attn_list = self._normalize_attentions(attentions)
        if len(attn_list) == 0:
            attn_list = self._collect_cached_attn()

        return logits, attn_list

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        self._clear_cached_attn()
        outputs = self._base_model()(
            pixel_values=x,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if not isinstance(last_hidden_state, torch.Tensor):
            raise TypeError(
                "Could not extract token sequence from Hugging Face backbone output."
            )
        return last_hidden_state

    def tokens_and_attn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self._clear_cached_attn()
        outputs = self._base_model()(
            pixel_values=x,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        attentions = getattr(outputs, "attentions", None)

        if not isinstance(last_hidden_state, torch.Tensor):
            raise TypeError(
                "Could not extract token sequence from Hugging Face backbone output."
            )

        attn_list = self._normalize_attentions(attentions)
        if len(attn_list) == 0:
            attn_list = self._collect_cached_attn()

        return last_hidden_state, attn_list

    def hook_modules(self) -> Dict[str, List[nn.Module]]:
        attn_probs_drop_modules: List[nn.Module] = []
        attn_proj_modules: List[nn.Module] = []
        ffn_modules: List[nn.Module] = []

        for i, block in enumerate(self._blocks()):
            self_attn = self._self_attention_module(block)

            attn_probs_drop = self._get_attn_probs_drop_module(self_attn)
            attn_proj = self._get_attn_proj_module(self_attn)
            ffn = self._get_ffn_module(block)

            if attn_probs_drop is None:
                raise TypeError(
                    f"Block {i} is missing an injected/native attention-probability hook module."
                )
            if attn_proj is None:
                raise TypeError(
                    f"Block {i} is missing an attention projection module."
                )
            if ffn is None:
                raise TypeError(
                    f"Block {i} is missing a feed-forward module."
                )

            attn_probs_drop_modules.append(attn_probs_drop)
            attn_proj_modules.append(attn_proj)
            ffn_modules.append(ffn)

        return {
            "attn_probs_drop": attn_probs_drop_modules,
            "attn_proj": attn_proj_modules,
            "ffn": ffn_modules,
        }

    def att_feature_module(self) -> nn.Module:
        blocks = self._blocks()
        if not blocks:
            raise TypeError("Model has no transformer blocks.")
        return blocks[-2] if len(blocks) >= 2 else blocks[-1]

    def _validate_model(self) -> None:
        base = self._base_model()

        if not hasattr(base, "encoder"):
            raise TypeError("Unsupported HF model: missing encoder.")
        if not hasattr(base.encoder, "layer"):
            raise TypeError("Unsupported HF model: missing encoder.layer.")
        if not hasattr(base, "embeddings"):
            raise TypeError("Unsupported HF model: missing embeddings.")

        blocks = self._blocks()
        if len(blocks) == 0:
            raise TypeError("Unsupported HF model: no transformer blocks found.")

        block0 = blocks[0]
        self_attn = self._self_attention_module(block0)
        if self_attn is None:
            raise TypeError("Unsupported HF block layout: could not find self-attention module.")
        if self._get_attn_proj_module(self_attn) is None:
            raise TypeError("Unsupported HF block layout: could not find attention projection module.")
        if self._get_ffn_module(block0) is None:
            raise TypeError("Unsupported HF block layout: could not find feed-forward module.")

    def _base_model(self) -> nn.Module:
        vit = getattr(self.model, "vit", None)
        if isinstance(vit, nn.Module):
            return vit

        deit = getattr(self.model, "deit", None)
        if isinstance(deit, nn.Module):
            return deit

        raise TypeError(
            "Unsupported model type. Expected a Hugging Face ViT/DeiT image-classification model."
        )

    def _blocks(self) -> List[nn.Module]:
        base = self._base_model()
        encoder = getattr(base, "encoder", None)
        if encoder is None:
            raise TypeError("Unsupported HF model: missing encoder.")
        layer = getattr(encoder, "layer", None)
        if layer is None:
            raise TypeError("Unsupported HF model: missing encoder.layer.")
        return [cast(nn.Module, blk) for blk in layer]

    def _self_attention_module(self, block: nn.Module) -> Optional[nn.Module]:
        outer = getattr(block, "attention", None)
        if outer is None:
            return None
        inner = getattr(outer, "attention", None)
        return inner if isinstance(inner, nn.Module) else None

    def _get_attn_probs_drop_module(self, self_attn: nn.Module) -> Optional[nn.Module]:
        injected = getattr(self_attn, "tokenfool_attn_probs_drop", None)
        if isinstance(injected, nn.Module):
            return injected

        native = getattr(self_attn, "dropout", None)
        if isinstance(native, nn.Module):
            return native

        native2 = getattr(self_attn, "attention_dropout", None)
        if isinstance(native2, nn.Module):
            return native2

        return None

    def _get_attn_proj_module(self, self_attn: nn.Module) -> Optional[nn.Module]:
        qkv = getattr(self_attn, "qkv", None)
        if isinstance(qkv, nn.Module):
            return qkv

        query = getattr(self_attn, "query", None)
        if isinstance(query, nn.Module):
            return query

        return None

    def _get_ffn_module(self, block: nn.Module) -> Optional[nn.Module]:
        mlp = getattr(block, "mlp", None)
        if isinstance(mlp, nn.Module):
            return mlp

        intermediate = getattr(block, "intermediate", None)
        if isinstance(intermediate, nn.Module):
            return intermediate

        return None

    def _clear_cached_attn(self) -> None:
        for block in self._blocks():
            self_attn = self._self_attention_module(block)
            if self_attn is not None and hasattr(self_attn, "_tokenfool_last_attn_probs"):
                self_attn._tokenfool_last_attn_probs = None

    def _collect_cached_attn(self) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for block in self._blocks():
            self_attn = self._self_attention_module(block)
            if self_attn is None:
                continue
            attn = getattr(self_attn, "_tokenfool_last_attn_probs", None)
            if isinstance(attn, torch.Tensor):
                out.append(attn)
        return out

    def _normalize_attentions(self, attentions) -> List[torch.Tensor]:
        if attentions is None:
            return []
        out: List[torch.Tensor] = []
        for a in attentions:
            if isinstance(a, torch.Tensor):
                out.append(a)
        return out

    def _patch_attention_blocks(self) -> None:
        for block in self._blocks():
            self_attn = self._self_attention_module(block)
            if self_attn is None:
                raise TypeError("Unsupported HF block layout: missing self-attention module.")

            if not hasattr(self_attn, "tokenfool_attn_probs_drop"):
                self_attn.tokenfool_attn_probs_drop = _IdentityDropout()

            if getattr(self_attn, "_tokenfool_patched", False):
                continue

            self._patch_self_attention_forward(self_attn)
            self_attn._tokenfool_patched = True

    def _patch_self_attention_forward(self, self_attn: nn.Module) -> None:
        required = ("query", "key", "value")
        missing = [name for name in required if not hasattr(self_attn, name)]
        if missing:
            raise TypeError(
                f"Unsupported HF self-attention layout for patching: "
                f"{type(self_attn).__name__} missing {missing}"
            )

        original_forward = self_attn.forward

        def _transpose_for_scores_fallback(
            module_self: nn.Module, x: torch.Tensor
        ) -> torch.Tensor:
            native = getattr(module_self, "transpose_for_scores", None)
            if callable(native):
                return native(x)

            num_heads = getattr(module_self, "num_attention_heads", None)
            head_dim = getattr(module_self, "attention_head_size", None)

            if not isinstance(num_heads, int) or not isinstance(head_dim, int):
                raise TypeError(
                    f"{type(module_self).__name__} missing transpose_for_scores and/or "
                    f"integer num_attention_heads / attention_head_size"
                )

            new_shape = x.size()[:-1] + (num_heads, head_dim)
            x = x.view(new_shape)
            return x.permute(0, 2, 1, 3)

        def patched_forward(
            module_self: nn.Module,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            *args,
            **kwargs,
        ):
            query_layer = _transpose_for_scores_fallback(
                module_self, module_self.query(hidden_states)
            )
            key_layer = _transpose_for_scores_fallback(
                module_self, module_self.key(hidden_states)
            )
            value_layer = _transpose_for_scores_fallback(
                module_self, module_self.value(hidden_states)
            )

            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            head_dim = getattr(module_self, "attention_head_size", None)
            if not isinstance(head_dim, int):
                raise TypeError("HF self-attention missing integer attention_head_size.")
            attention_scores = attention_scores / (head_dim ** 0.5)

            attention_probs = torch.softmax(attention_scores, dim=-1)
            attention_probs = module_self.tokenfool_attn_probs_drop(attention_probs)

            native_dropout = None
            maybe_dropout = getattr(module_self, "dropout", None)
            if isinstance(maybe_dropout, nn.Module):
                native_dropout = maybe_dropout
            else:
                maybe_dropout = getattr(module_self, "attention_dropout", None)
                if isinstance(maybe_dropout, nn.Module):
                    native_dropout = maybe_dropout

            if native_dropout is not None:
                attention_probs = native_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            module_self._tokenfool_last_attn_probs = attention_probs

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

            all_head_size = getattr(module_self, "all_head_size", None)
            if not isinstance(all_head_size, int):
                num_heads = getattr(module_self, "num_attention_heads", None)
                if not isinstance(num_heads, int):
                    raise TypeError(
                        "HF self-attention missing integer all_head_size / num_attention_heads."
                    )
                all_head_size = num_heads * head_dim

            new_context_layer_shape = context_layer.size()[:-2] + (all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)

            returned_attentions = attention_probs if output_attentions else None
            return context_layer, returned_attentions

        module_forward = types.MethodType(patched_forward, self_attn)
        self_attn.forward = module_forward
        self_attn._tokenfool_original_forward = original_forward


        