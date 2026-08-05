from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from repo_runtime import configure_repo_runtime

configure_repo_runtime(Path(__file__).resolve().parents[1])

from ultralytics.nn.modules import Conv


class HaarTransform(nn.Module):
    """Parameter-free Haar wavelet split into LL/LH/HL/HH sub-bands."""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        return torch.cat(
            (
                (x00 + x01 + x10 + x11) * 0.5,
                (x00 - x01 + x10 - x11) * 0.5,
                (x00 + x01 - x10 - x11) * 0.5,
                (x00 - x01 - x10 + x11) * 0.5,
            ),
            dim=1,
        )


class WTConv(nn.Module):
    """Bypass-fusion wavelet downsampling that preserves the standard semantic path."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p: int | None = None, act: bool = True):
        super().__init__()
        if s != 2:
            raise ValueError(f"WTConv is only defined for stride-2 downsampling, but got s={s}.")

        self.c1 = c1
        self.c2 = c2
        padding = 1 if p is None else p
        self.main_conv = Conv(c1, c2, k=k, s=s, p=padding, act=act)
        self.wavelet_split = HaarTransform()
        self.wavelet_proj = Conv(c1 * 3, c2, k=1, s=1, act=act)
        self.fusion_conv = Conv(c2 * 2, c2, k=1, s=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_feat = self.main_conv(x)
        subbands = self.wavelet_split(x)
        high_freq_feat = subbands[:, self.c1 :, :, :]
        high_freq_feat = self.wavelet_proj(high_freq_feat)
        return self.fusion_conv(torch.cat((main_feat, high_freq_feat), dim=1))


class WTRefine(nn.Module):
    """Resolution-preserving wavelet refinement for post-fusion neck features."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, act: bool = True):
        super().__init__()
        if s != 1:
            raise ValueError(f"WTRefine is only defined for stride-1 refinement, but got s={s}.")

        self.c1 = c1
        padding = 1 if p is None else p
        self.main_proj = Conv(c1, c2, k=1, s=1, act=act)
        self.wavelet_split = HaarTransform()
        self.wavelet_proj = Conv(c1 * 3, c2, k=1, s=1, act=act)
        self.wavelet_local = Conv(c2, c2, k=k, s=1, p=padding, act=act)
        self.fusion_conv = Conv(c2 * 2, c2, k=1, s=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_feat = self.main_proj(x)
        subbands = self.wavelet_split(x)
        high_freq_feat = subbands[:, self.c1 :, :, :]
        high_freq_feat = self.wavelet_proj(high_freq_feat)
        high_freq_feat = self.wavelet_local(high_freq_feat)
        high_freq_feat = F.interpolate(high_freq_feat, size=main_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.fusion_conv(torch.cat((main_feat, high_freq_feat), dim=1))


class ResidualWTConv(nn.Module):
    """Residualized WTConv downsampling for backbone ablations."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p: int | None = None, act: bool = True):
        super().__init__()
        if s != 2:
            raise ValueError(f"ResidualWTConv is only defined for stride-2 downsampling, but got s={s}.")

        self.shortcut = Conv(c1, c2, k=1, s=s, p=0, act=False)
        self.wavelet_down = WTConv(c1, c2, k=k, s=s, p=p, act=act)
        self.out_proj = Conv(c2, c2, k=1, s=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.shortcut(x) + self.wavelet_down(x))


class DWTDown(WTConv):
    """Backward-compatible alias for older YAMLs that still refer to DWTDown."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p: int | None = None, act: bool = True):
        super().__init__(c1, c2, k=k, s=s, p=p, act=act)


class ECABlock(nn.Module):
    """Efficient Channel Attention with negligible parameter overhead."""

    def __init__(self, c1: int, k_size: int = 3):
        super().__init__()
        if k_size % 2 == 0:
            raise ValueError(f"ECABlock expects an odd kernel size, but got {k_size}.")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.activation(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class SPDConv(nn.Module):
    """Space-to-depth downsampling followed by convolution to preserve local pixels."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p: int | None = None, act: bool = True):
        super().__init__()
        if s != 2:
            raise ValueError(f"SPDConv is only defined for stride-2 downsampling, but got s={s}.")
        self.proj = Conv(c1 * 4, c2, k=k, s=1, p=p, act=act)

    @staticmethod
    def space_to_depth(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
        return torch.cat(
            (
                x[:, :, 0::2, 0::2],
                x[:, :, 1::2, 0::2],
                x[:, :, 0::2, 1::2],
                x[:, :, 1::2, 1::2],
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.space_to_depth(x))


class DefectMorphGate(nn.Module):
    """Lightweight morphology gate for round, horizontal, vertical, and diffuse defects."""

    def __init__(self, c1: int, reduction: int = 4):
        super().__init__()
        hidden = max(c1 // reduction, 8)
        self.round_branch = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)
        self.horizontal_branch = nn.Conv2d(c1, c1, kernel_size=(1, 5), padding=(0, 2), groups=c1, bias=False)
        self.vertical_branch = nn.Conv2d(c1, c1, kernel_size=(5, 1), padding=(2, 0), groups=c1, bias=False)
        self.diffuse_branch = nn.Conv2d(c1, c1, kernel_size=3, padding=2, dilation=2, groups=c1, bias=False)
        self.mix = Conv(c1 * 4, c1, k=1, s=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        morphology = torch.cat(
            (
                self.round_branch(x),
                self.horizontal_branch(x),
                self.vertical_branch(x),
                self.diffuse_branch(x),
            ),
            dim=1,
        )
        refined = self.mix(morphology)
        return x + refined * self.gate(refined)


class WeightedConcat(nn.Module):
    """Learnable scalar weighting before feature concatenation."""

    def __init__(self, dimension: int = 1, n_inputs: int = 2):
        super().__init__()
        self.dimension = dimension
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        weights = torch.softmax(self.weights[: len(xs)], dim=0)
        return torch.cat([x * weights[i] for i, x in enumerate(xs)], dim=self.dimension)


