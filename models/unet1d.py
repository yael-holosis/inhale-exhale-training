"""A 1D U-Net for per-sample phase segmentation, sized for the edge device.

Shape follows U-Time (Perslev et al., NeurIPS 2019, arXiv:1910.11162) - a fully convolutional
encoder/decoder that emits one class per input sample, with no recurrence and no fixed input
length. Two departures, both for size:

- **Depthwise-separable convolutions** everywhere but the stem, which is where the parameter
  count goes in a U-Net of this depth.
- **Narrow.** The signal is one channel at 10 fps; a breath is 24-150 samples. There is nothing
  here that needs the width a sleep-staging net carries over 13 EEG channels.

Any length in, same length out. The input is padded up to a multiple of ``2 ** depth`` inside
`forward` and the output is cropped back, so a caller never has to know the depth.

Receptive field matters more than parameter count here and is reported by `receptive_field()`:
it must cover at least two breaths, and at 4 bpm a breath is 150 samples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv(nn.Module):
    """Depthwise k-tap + pointwise mix, batch-normed. ~(k + c_out) params per input channel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size // 2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.pointwise(self.depthwise(x))), inplace=True)


class Block(nn.Module):
    """Two convolutions at one resolution. The stem is dense because 1 -> c cannot be split."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stem: bool = False):
        super().__init__()
        if stem:
            first = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))
        else:
            first = SeparableConv(in_channels, out_channels, kernel_size)
        self.body = nn.Sequential(first, SeparableConv(out_channels, out_channels, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UNet1D(nn.Module):
    """Per-sample logits over the phase vocabulary.

    Args:
        in_channels: 1 for the waveform alone. A second channel is where an orientation cue
            would go - see README, "what the model is not given".
        n_classes: size of the vocabulary (`phase.labels.N_CLASSES`).
        channels: encoder widths, one per level. Its length is the depth.
        kernel_size: odd, so padding keeps the length exactly.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 4,
                 channels: tuple[int, ...] = (16, 24, 32, 48), bottleneck: int = 64,
                 kernel_size: int = 9, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.depth = len(channels)
        self.kernel_size = kernel_size
        self.channels = tuple(channels)

        widths = [in_channels, *channels]
        self.encoder = nn.ModuleList(
            Block(widths[level], widths[level + 1], kernel_size, stem=(level == 0))
            for level in range(self.depth))
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = Block(channels[-1], bottleneck, kernel_size)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        up = [bottleneck, *reversed(channels[1:])]
        self.decoder = nn.ModuleList(
            Block(up[level] + channels[self.depth - 1 - level],
                  channels[self.depth - 1 - level], kernel_size)
            for level in range(self.depth))
        self.head = nn.Conv1d(channels[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, in_channels, length) -> (batch, n_classes, length). Length is preserved."""
        length = x.shape[-1]
        stride = 2 ** self.depth
        pad = (-length) % stride
        if pad:
            # Replicate rather than zero: a zero-pad puts a step at the edge, and a step is what
            # the first layer is looking for.
            x = F.pad(x, (0, pad), mode="replicate")

        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.dropout(self.bottleneck(x))
        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)[..., :length]

    def receptive_field(self) -> int:
        """Samples of input one output sample can see.

        Tracked as (field, jump) through the encoder, the bottleneck and the decoder, because a
        convolution after a pool reaches `jump` times further than one before it. Must cover at
        least two breaths: at 4 bpm a breath is 150 samples.
        """
        taps = self.kernel_size - 1
        field, jump = 1, 1
        for _ in range(self.depth):                    # encoder: 2 convs, then pool
            field += 2 * taps * jump
            jump *= 2
        field += 2 * taps * jump                       # bottleneck
        for _ in range(self.depth):                    # decoder: upsample, then 2 convs
            jump //= 2
            field += 2 * taps * jump
        return field

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
