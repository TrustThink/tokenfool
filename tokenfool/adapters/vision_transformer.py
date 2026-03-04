from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import copy


@dataclass
class VisionTransformerClassifier:

    model: torch.nn.Module
    _patched: bool = False

    def __post_init__(self) -> None:
        # TODO: sanity check
        # deepcopy to avoid patching in place
        self.model = copy.deepcopy(self.model)

    @property
    def cls_index(self) -> int:
        return 0

    # -------------------------
    # Forward API
    # -------------------------
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.logits_and_attn(x)
        return y

    def logits_and_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self._patch_if_needed()
        logits, attn_list = self.model(x)
        if isinstance(logits, tuple):
            logits = (logits[0] + logits[1]) / 2
        return logits, attn_list

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        t, _ = self.tokens_and_attn(x)
        return t

    def tokens_and_attn(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self._patch_if_needed()
        tokens, attn_list = self.model.forward_features(x)
        return tokens, attn_list
    
    def hook_modules(self):
        self._patch_if_needed()
        blocks = self.model.blocks
        return {
            "attn_drop": [b.attn.attn_drop for b in blocks],
            "qkv": [b.attn.qkv for b in blocks],
            "mlp": [b.mlp for b in blocks],
        }

    # -------------------------
    # Internal patching
    # -------------------------
    def _patch_if_needed(self) -> None:
        if self._patched:
            return

        m = self.model

        required = ["blocks", "patch_embed", "cls_token", "pos_embed", "pos_drop", "norm", "head"]
        missing = [name for name in required if not hasattr(m, name)]
        if missing:
            raise TypeError(f"Model missing required attributes: {missing}")

        # -------------------------
        # Patch Attention.forward
        # -------------------------
        def attn_forward_with_capture(self, x: torch.Tensor):
            """
            Returns:
              out: (B, N, C)
              attn_probs: (B, heads, N, N)  [softmax probs PRE-dropout]
            """
            B, N, C = x.shape

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn_scores = (q @ k.transpose(-2, -1)) * self.scale
            attn_probs = attn_scores.softmax(dim=-1)  
            attn_drop = self.attn_drop(attn_probs)    

            out = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out, attn_probs

        # -------------------------
        # Patch Block forward to collect attentions
        # -------------------------
        def block_forward_collect(self, x: torch.Tensor, attn_list: List[torch.Tensor]):
            y = self.norm1(x)
            attn_out, attn = self.attn(y)

            if hasattr(self, "drop_path1"):
                x = x + self.drop_path1(attn_out)
            else:
                x = x + self.drop_path(attn_out)

            if hasattr(self, "drop_path2"):
                x = x + self.drop_path2(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))

            attn_list.append(attn.clone())
            return x

        # apply attention + block patches
        for blk in m.blocks:
            if not hasattr(blk, "attn"):
                raise TypeError("Expected each block to have .attn")
            blk.attn.forward = attn_forward_with_capture.__get__(blk.attn, type(blk.attn))
            blk.forward_collect = block_forward_collect.__get__(blk, type(blk))

        # -------------------------
        # Patch forward_features
        # -------------------------
        def forward_features_collect(self, x: torch.Tensor):
            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tok = self.cls_token.expand(B, -1, -1)

            # DeiT distilled has dist_token
            if getattr(self, "dist_token", None) is not None:
                dist_tok = self.dist_token.expand(B, -1, -1)
                x = torch.cat((cls_tok, dist_tok, x), dim=1)
            else:
                x = torch.cat((cls_tok, x), dim=1)

            x = self.pos_drop(x + self.pos_embed)

            attn_list: List[torch.Tensor] = []
            for blk in self.blocks:
                x = blk.forward_collect(x, attn_list)

            x = self.norm(x)
            return x, attn_list

        m.forward_features = forward_features_collect.__get__(m, type(m))

        # -------------------------
        # Patch forward
        # -------------------------
        def forward_collect(self, x: torch.Tensor):
            """
            Returns:
              logits: (B, C)  (distilled models averaged to tensor)
              attn_list: list of (B, heads, N, N)
            """
            tokens, attn_list = self.forward_features(x)

            if getattr(self, "dist_token", None) is not None:
                cls_tok = tokens[:, 0]
                dist_tok = tokens[:, 1]
                x_cls = self.head(cls_tok)
                if hasattr(self, "head_dist"):
                    x_dist = self.head_dist(dist_tok)
                    logits = (x_cls + x_dist) / 2
                else:
                    logits = x_cls
            else:
                logits = self.head(tokens[:, 0])

            return logits, attn_list

        m.forward = forward_collect.__get__(m, type(m))
        self._patched = True