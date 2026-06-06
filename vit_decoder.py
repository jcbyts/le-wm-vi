"""ViT-tiny transpose decoder for FOND-JEPA.

Maps a D-dim latent code to an image. This is a drop-in replacement
for ``ConvDecoder`` with the same call contract: ``decoder(code)``.

The attention below is deliberately explicit instead of fused SDPA or
``nn.MultiheadAttention``. FOND can backpropagate through the inner inference
loop with ``create_graph=True``, which needs double-backward through the decoder.
"""

import torch
from einops import rearrange
from torch import nn


class _Attn(nn.Module):
    """Explicit multi-head self-attention, double-backward safe."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.h = heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.h) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = rearrange(att @ v, "b h n d -> b n (h d)")
        return self.proj(out)


class _DecBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attn(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTDecoder(nn.Module):
    """Latent ``(B, D)`` -> image ``(B, C, hw, hw)``."""

    def __init__(
        self,
        latent_dim,
        dim=192,
        depth=6,
        heads=6,
        mlp_dim=768,
        img_ch=3,
        img_hw=64,
        patch_size=8,
        dropout=0.0,
        output_activation="sigmoid",
    ):
        super().__init__()
        assert img_hw % patch_size == 0, "img_hw must be divisible by patch_size"
        assert output_activation in ("sigmoid", "identity")
        self.dim = dim
        self.img_ch = img_ch
        self.img_hw = img_hw
        self.patch = patch_size
        self.output_activation = output_activation
        self.gh = img_hw // patch_size
        self.n_tok = self.gh * self.gh
        self.patch_px = patch_size * patch_size * img_ch

        self.latent_to_tokens = nn.Linear(latent_dim, self.n_tok * dim)
        self.pos = nn.Parameter(torch.randn(1, self.n_tok, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_DecBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.to_pixels = nn.Linear(dim, self.patch_px)

    def forward(self, code):
        b = code.size(0)
        x = self.latent_to_tokens(code).view(b, self.n_tok, self.dim)
        x = x + self.pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.to_pixels(x)
        x = rearrange(
            x,
            "b (gh gw) (ph pw c) -> b c (gh ph) (gw pw)",
            gh=self.gh,
            gw=self.gh,
            ph=self.patch,
            pw=self.patch,
            c=self.img_ch,
        )
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        return x
