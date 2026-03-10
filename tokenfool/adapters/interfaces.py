from typing import Protocol, List, Tuple, Dict
import torch
import torch.nn as nn


class TransformerClassifier(Protocol):
    """
    Encoder-only transformer classifier with accessible attention maps.
    """
    @property
    def cls_index(self) -> int:
        ...

    @property
    def num_prefix_tokens(self) -> int:
        """
        Number of non-patch prefix tokens at the front of the sequence.
        Examples:
            ViT/DeiT: 1
            DeiT distilled: 2
        """
        ...

    def zero_grad(self, set_to_none: bool = True) -> None:
        ...


    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C)"""

    def logits_and_attn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            logits: (B, C)
            attn_list: List of (B, heads, N, N)
        """

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D)"""

    def tokens_and_attn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """(B, N, D), attn_list"""



class HookableTransformerClassifier(TransformerClassifier, Protocol):
    """
    Transformer classifier that exposes internal modules requiered for gradient-hook based attacks.
    """

    def hook_modules(self) -> Dict[str, List[nn.Module]]:
        """
        Return modules that support gradient hooks.

        Required keys for ATT:
            "attn_drop"
            "qkv"
            "mlp"
        """

    def att_feature_module(self) -> nn.Module:
        ...