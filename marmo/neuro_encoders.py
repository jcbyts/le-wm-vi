from __future__ import annotations

import warnings

import torch
from torch import nn
import torch.nn.functional as F


class AlexNetV1Encoder(nn.Module):
    """AlexNet early-visual feature encoder with a LeWM vector output.

    The Brain-Score V1 layer commitment for AlexNet is ``features.2``. This
    module keeps that spatial feature map internally and projects it back to the
    existing LeWM vector contract, so current SIGReg/readout tooling still works.
    """

    def __init__(
        self,
        img_ch: int,
        embed_dim: int,
        *,
        input_hw: int,
        pretrained: bool = True,
        freeze_frontend: bool = True,
        resize_hw: int = 224,
        feature_index: int = 2,
        pool_hw: int = 1,
        pixel_mode: str = "visioncore",
    ):
        super().__init__()
        self.img_ch = int(img_ch)
        self.embed_dim = int(embed_dim)
        self.input_hw = int(input_hw)
        self.resize_hw = int(resize_hw)
        self.feature_index = int(feature_index)
        self.pool_hw = int(pool_hw)
        self.pixel_mode = str(pixel_mode).lower()
        if self.pixel_mode not in {"unit", "visioncore", "auto"}:
            raise ValueError("pixel_mode must be unit, visioncore, or auto")
        if self.feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        if self.pool_hw <= 0:
            raise ValueError("pool_hw must be positive")

        try:
            from torchvision.models import AlexNet_Weights, alexnet

            weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
            model = alexnet(weights=weights)
        except Exception as exc:
            if pretrained:
                warnings.warn(
                    f"Could not load pretrained AlexNet weights ({exc}); falling back to random AlexNet.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                from torchvision.models import alexnet

                model = alexnet(weights=None)
            else:
                raise

        feature_layers = list(model.features.children())
        if self.feature_index >= len(feature_layers):
            raise ValueError(f"AlexNet feature_index={self.feature_index} exceeds {len(feature_layers) - 1}")
        self.features = nn.Sequential(*feature_layers[: self.feature_index + 1])
        if freeze_frontend:
            for param in self.features.parameters():
                param.requires_grad_(False)

        if self.img_ch == 3:
            self.input_adapter = nn.Identity()
        elif self.img_ch == 1:
            self.input_adapter = None
        else:
            self.input_adapter = nn.Conv2d(self.img_ch, 3, kernel_size=1, bias=False)
            nn.init.constant_(self.input_adapter.weight, 1.0 / float(self.img_ch))

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        with torch.no_grad():
            dummy_hw = self.resize_hw if self.resize_hw > 0 else self.input_hw
            dummy = torch.zeros(1, 3, dummy_hw, dummy_hw)
            feat = self.features(dummy)
            pooled = F.adaptive_avg_pool2d(feat, (self.pool_hw, self.pool_hw))
            pooled_dim = int(pooled.numel())
        self.projector = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.embed_dim),
        )

    def _to_unit_pixels(self, x: torch.Tensor) -> torch.Tensor:
        if self.pixel_mode == "unit":
            return x.clamp(0.0, 1.0)
        if self.pixel_mode == "visioncore":
            return (x + (127.0 / 255.0)).clamp(0.0, 1.0)
        # Auto mode is for quick experiments when older checkpoints/configs do
        # not record pixel normalization. It is intentionally not the default.
        if torch.amin(x.detach()).item() < -0.05:
            return (x + (127.0 / 255.0)).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)

    def _adapt_channels(self, x: torch.Tensor) -> torch.Tensor:
        if self.img_ch == 3:
            return self.input_adapter(x)
        if self.img_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return self.input_adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_unit_pixels(x.float())
        x = self._adapt_channels(x)
        if self.resize_hw > 0 and (x.shape[-2] != self.resize_hw or x.shape[-1] != self.resize_hw):
            x = F.interpolate(x, size=(self.resize_hw, self.resize_hw), mode="bilinear", align_corners=False)
        x = (x - self.imagenet_mean) / self.imagenet_std
        feat = self.features(x)
        pooled = F.adaptive_avg_pool2d(feat, (self.pool_hw, self.pool_hw)).flatten(1)
        return self.projector(pooled)


class VOneBlockEncoder(nn.Module):
    """VOneBlock frontend projected into the existing LeWM vector space.

    This uses the VOneNet package as an optional dependency instead of vendoring
    its GPL source into this repo. The default is deterministic VOneBlock
    activations (``noise_mode=None``), which is better suited for a world-model
    target than sampling frontend noise at every pass.
    """

    def __init__(
        self,
        img_ch: int,
        embed_dim: int,
        *,
        input_hw: int,
        resize_hw: int = 224,
        pool_hw: int = 4,
        pixel_mode: str = "visioncore",
        freeze_frontend: bool = True,
        simple_channels: int = 128,
        complex_channels: int = 128,
        ksize: int = 25,
        stride: int = 4,
        visual_degrees: float = 8.0,
        sf_corr: float = 0.75,
        sf_max: float = 9.0,
        sf_min: float = 0.0,
        noise_mode: str | None = None,
    ):
        super().__init__()
        self.img_ch = int(img_ch)
        self.embed_dim = int(embed_dim)
        self.input_hw = int(input_hw)
        self.resize_hw = int(resize_hw)
        self.pool_hw = int(pool_hw)
        self.pixel_mode = str(pixel_mode).lower()
        if self.pixel_mode not in {"unit", "visioncore", "auto"}:
            raise ValueError("pixel_mode must be unit, visioncore, or auto")
        if self.pool_hw <= 0:
            raise ValueError("pool_hw must be positive")

        try:
            from vonenet import get_model
        except Exception as exc:
            raise ImportError(
                "encoder_kind='voneblock' requires the VOneNet package. "
                "Install it with: /home/tejas/.conda/envs/env/bin/pip install -e /tmp/vonenet"
            ) from exc

        vone = get_model(
            model_arch=None,
            pretrained=False,
            map_location="cpu",
            image_size=self.resize_hw if self.resize_hw > 0 else self.input_hw,
            visual_degrees=float(visual_degrees),
            simple_channels=int(simple_channels),
            complex_channels=int(complex_channels),
            ksize=int(ksize),
            stride=int(stride),
            sf_corr=float(sf_corr),
            sf_max=float(sf_max),
            sf_min=float(sf_min),
            noise_mode=None if noise_mode in {None, "none", ""} else str(noise_mode),
        )
        self.vone_block = vone.module if hasattr(vone, "module") else vone
        self.vone_block.eval()
        if freeze_frontend:
            for param in self.vone_block.parameters():
                param.requires_grad_(False)

        if self.img_ch == 3:
            self.input_adapter = nn.Identity()
        elif self.img_ch == 1:
            self.input_adapter = None
        else:
            self.input_adapter = nn.Conv2d(self.img_ch, 3, kernel_size=1, bias=False)
            nn.init.constant_(self.input_adapter.weight, 1.0 / float(self.img_ch))

        with torch.no_grad():
            dummy_hw = self.resize_hw if self.resize_hw > 0 else self.input_hw
            dummy = torch.zeros(1, 3, dummy_hw, dummy_hw)
            feat = self.vone_block(dummy)
            pooled = F.adaptive_avg_pool2d(feat, (self.pool_hw, self.pool_hw))
            pooled_dim = int(pooled.numel())
        self.projector = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.embed_dim),
        )

    def _to_unit_pixels(self, x: torch.Tensor) -> torch.Tensor:
        if self.pixel_mode == "unit":
            return x.clamp(0.0, 1.0)
        if self.pixel_mode == "visioncore":
            return (x + (127.0 / 255.0)).clamp(0.0, 1.0)
        if torch.amin(x.detach()).item() < -0.05:
            return (x + (127.0 / 255.0)).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)

    def _adapt_channels(self, x: torch.Tensor) -> torch.Tensor:
        if self.img_ch == 3:
            return self.input_adapter(x)
        if self.img_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return self.input_adapter(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_unit_pixels(x.float())
        x = self._adapt_channels(x)
        if self.resize_hw > 0 and (x.shape[-2] != self.resize_hw or x.shape[-1] != self.resize_hw):
            x = F.interpolate(x, size=(self.resize_hw, self.resize_hw), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        feat = self.vone_block(x)
        pooled = F.adaptive_avg_pool2d(feat, (self.pool_hw, self.pool_hw)).flatten(1)
        return self.projector(pooled)


class VOneAlexNetEncoder(nn.Module):
    """VOneAlexNet early-feature encoder with a LeWM vector output.

    Feature taps:
      - ``feature_index=-2``: VOneBlock output
      - ``feature_index=-1``: 1x1 bottleneck output
      - ``feature_index>=0``: output after that AlexNet-backend feature layer

    The Brain-Score VOneAlexNet V1 commitment is the first backend ReLU, which
    corresponds to ``feature_index=1`` in the local VOneNet implementation.
    """

    def __init__(
        self,
        img_ch: int,
        embed_dim: int,
        *,
        input_hw: int,
        pretrained: bool = True,
        freeze_frontend: bool = True,
        resize_hw: int = 224,
        feature_index: int = 1,
        pool_hw: int = 1,
        pixel_mode: str = "visioncore",
        simple_channels: int = 128,
        complex_channels: int = 128,
        ksize: int = 25,
        stride: int = 4,
        visual_degrees: float = 8.0,
        sf_corr: float = 0.75,
        sf_max: float = 9.0,
        sf_min: float = 0.0,
        noise_mode: str | None = None,
    ):
        super().__init__()
        self.img_ch = int(img_ch)
        self.embed_dim = int(embed_dim)
        self.input_hw = int(input_hw)
        self.resize_hw = int(resize_hw)
        self.feature_index = int(feature_index)
        self.pool_hw = int(pool_hw)
        self.pixel_mode = str(pixel_mode).lower()
        if self.pixel_mode not in {"unit", "visioncore", "auto"}:
            raise ValueError("pixel_mode must be unit, visioncore, or auto")
        if self.feature_index < -2:
            raise ValueError("VOneAlexNet feature_index must be -2, -1, or a non-negative backend feature index")
        if self.pool_hw <= 0:
            raise ValueError("pool_hw must be positive")

        try:
            from vonenet import get_model
        except Exception as exc:
            raise ImportError(
                "encoder_kind='vonealexnet' requires the VOneNet package. "
                "Install it with: /home/tejas/.conda/envs/env/bin/pip install -e /tmp/vonenet"
            ) from exc

        kwargs = {}
        if not pretrained:
            kwargs = dict(
                image_size=self.resize_hw if self.resize_hw > 0 else self.input_hw,
                visual_degrees=float(visual_degrees),
                simple_channels=int(simple_channels),
                complex_channels=int(complex_channels),
                ksize=int(ksize),
                stride=int(stride),
                sf_corr=float(sf_corr),
                sf_max=float(sf_max),
                sf_min=float(sf_min),
                noise_mode=None if noise_mode in {None, "none", ""} else str(noise_mode),
            )
        model = get_model(model_arch="alexnet", pretrained=bool(pretrained), map_location="cpu", **kwargs)
        self.model = model.module if hasattr(model, "module") else model
        if hasattr(self.model.vone_block, "set_noise_mode"):
            self.model.vone_block.set_noise_mode(None if noise_mode in {None, "none", ""} else str(noise_mode))
        self.model.eval()
        if freeze_frontend:
            for param in self.model.parameters():
                param.requires_grad_(False)

        self.backend_features = self.model.model.features
        if self.feature_index >= len(self.backend_features):
            raise ValueError(
                f"VOneAlexNet feature_index={self.feature_index} exceeds "
                f"{len(self.backend_features) - 1}"
            )

        if self.img_ch == 3:
            self.input_adapter = nn.Identity()
        elif self.img_ch == 1:
            self.input_adapter = None
        else:
            self.input_adapter = nn.Conv2d(self.img_ch, 3, kernel_size=1, bias=False)
            nn.init.constant_(self.input_adapter.weight, 1.0 / float(self.img_ch))

        with torch.no_grad():
            dummy_hw = self.resize_hw if self.resize_hw > 0 else self.input_hw
            dummy = torch.zeros(1, 3, dummy_hw, dummy_hw)
            feat = self._extract_features(dummy)
            pooled = F.adaptive_avg_pool2d(feat, (self.pool_hw, self.pool_hw))
            pooled_dim = int(pooled.numel())
        self.projector = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.embed_dim),
        )

    def _to_unit_pixels(self, x: torch.Tensor) -> torch.Tensor:
        if self.pixel_mode == "unit":
            return x.clamp(0.0, 1.0)
        if self.pixel_mode == "visioncore":
            return (x + (127.0 / 255.0)).clamp(0.0, 1.0)
        if torch.amin(x.detach()).item() < -0.05:
            return (x + (127.0 / 255.0)).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)

    def _adapt_channels(self, x: torch.Tensor) -> torch.Tensor:
        if self.img_ch == 3:
            return self.input_adapter(x)
        if self.img_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return self.input_adapter(x)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.vone_block(x)
        if self.feature_index == -2:
            return x
        x = self.model.bottleneck(x)
        if self.feature_index == -1:
            return x
        return self.backend_features[: self.feature_index + 1](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_unit_pixels(x.float())
        x = self._adapt_channels(x)
        if self.resize_hw > 0 and (x.shape[-2] != self.resize_hw or x.shape[-1] != self.resize_hw):
            x = F.interpolate(x, size=(self.resize_hw, self.resize_hw), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        feat = self._extract_features(x)
        pooled = F.adaptive_avg_pool2d(feat, (self.pool_hw, self.pool_hw)).flatten(1)
        return self.projector(pooled)
