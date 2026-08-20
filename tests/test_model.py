"""The network: any length in, same length out, and small enough to deploy."""

import pytest
import torch

from models.unet1d import UNet1D
from phase.labels import N_CLASSES

EDGE_PARAMETER_BUDGET = 100_000
"""What "light enough for the device" means, stated so a widening cannot pass unnoticed."""


@pytest.mark.parametrize("length", [37, 128, 200, 201, 255, 600, 4001])
def test_output_length_matches_input_length(length):
    model = UNet1D(channels=(16, 24, 32), bottleneck=48)
    assert model(torch.randn(2, 1, length)).shape == (2, N_CLASSES, length)


def test_the_default_shape_fits_the_edge_budget():
    model = UNet1D(channels=(16, 24, 32), bottleneck=48)
    assert model.n_parameters() < EDGE_PARAMETER_BUDGET


def test_the_receptive_field_covers_two_slow_breaths():
    # A breath at 4 bpm is 150 samples at 10 fps, and the phase of one breath is only readable
    # against its neighbours.
    model = UNet1D(channels=(16, 24, 32), bottleneck=48)
    assert model.receptive_field() >= 2 * 150


def test_an_even_kernel_is_refused():
    with pytest.raises(ValueError):
        UNet1D(kernel_size=8)


def test_gradients_reach_the_stem():
    model = UNet1D(channels=(16, 24), bottleneck=32)
    model(torch.randn(1, 1, 64)).sum().backward()
    assert model.encoder[0].body[0][0].weight.grad is not None
