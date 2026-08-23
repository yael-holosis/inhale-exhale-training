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


def test_the_same_window_gives_the_same_answer_in_any_equal_length_batch():
    """The guarantee inference relies on: a batch-mate cannot change a window's answer.

    True in eval mode *as long as the batch is not padded* - BatchNorm then uses its running
    statistics, and nothing else in the network crosses the batch dimension.
    """
    model = UNet1D(channels=(16, 24, 32), bottleneck=48).eval()
    one = torch.randn(1, 1, 200)
    batch = torch.cat([one, torch.randn(3, 1, 200)])
    with torch.no_grad():
        assert torch.allclose(model(one)[0], model(batch)[0], atol=1e-6)


def test_padding_a_batch_changes_the_answer_near_the_pad_and_only_there():
    """Why inference must not mix lengths in one batch.

    A shorter window padded up to a longer one has the pad inside the receptive field of its own
    tail, so those outputs are not what the window alone would produce. Beyond half a receptive
    field from the join the two are identical - which is the rule: run one window at a time, or
    bucket by length.
    """
    # Seeded: the tail difference is a few 1e-4 on unlucky weights, so an unseeded net failed
    # the threshold below about one run in six.
    torch.manual_seed(0)
    model = UNet1D(channels=(16, 24, 32), bottleneck=48).eval()
    short = torch.randn(1, 1, 200)
    padded = torch.zeros(2, 1, 600)
    padded[0, :, :200] = short
    padded[1] = torch.randn(1, 1, 600)
    with torch.no_grad():
        alone = model(short)[0]
        together = model(padded)[0][:, :200]

    difference = (alone - together).abs().max(dim=0).values
    far = model.receptive_field() // 2
    assert difference[:200 - far].max() < 1e-4, "the far end must be unaffected"
    assert difference[-10:].max() > 1e-3, "the tail must be affected - that is the whole point"
