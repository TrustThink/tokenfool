import torch
import torch.nn as nn


class FakePatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=16):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class FakeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class FakeBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FakeAttention(dim)
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x))
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class FakeViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        embed_dim=16,
        depth=2,
        num_classes=5,
        distilled=False,
    ):
        super().__init__()
        self.patch_embed = FakePatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        num_prefix = 2 if distilled else 1
        self.pos_embed = nn.Parameter(torch.zeros(1, n + num_prefix, embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList([FakeBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls, x), dim=1)
        else:
            dist = self.dist_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls, dist, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.dist_token is None:
            x = x[:, 0]
        else:
            x = (x[:, 0] + x[:, 1]) / 2
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


class TinyBackbone(nn.Module):
    def __init__(self, image_size=32, patch_size=8, dim=16, num_classes=5):
        super().__init__()
        assert image_size % patch_size == 0
        patch_dim = 3 * patch_size * patch_size
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_dim, dim)
        self.prefix = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([TinyBlock(dim), TinyBlock(dim)])
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


class TinyAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim)
        self.attn_drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn_drop(self.qkv(x))


class TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = TinyAttn(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class DummyHookableViT:
    def __init__(self, backbone: TinyBackbone):
        self.model = backbone

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.model.zero_grad(set_to_none=set_to_none)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def hook_modules(self):
        return {
            "attn_probs_drop": [blk.attn.attn_drop for blk in self.model.blocks],
            "attn_proj": [blk.attn.qkv for blk in self.model.blocks],
            "ffn": [blk.mlp for blk in self.model.blocks],
        }


class NotAViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)