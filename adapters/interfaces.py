from typing import Protocol, List, Tuple
import torch


class TransformerClassifier(Protocol):
    """
    Encoder-only transformer classifier with accessible attention maps.
    """

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C)"""

    def logits_and_attn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            logits: (B, C)
            attn_list: List[L] of (B, heads, N, N)
        """

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D)"""

    def tokens_and_attn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """(B, N, D), attn_list"""

    @property
    def cls_index(self) -> int:
        ...