"""The dataset and the splits, on a synthetic set - no AWS, no shards from a real build."""

import numpy as np
import pandas as pd
import pytest

from tests import synthetic
from phase.dataset import WindowDataset, class_weights, time_warp
from phase.labels import EXHALE, INHALE, PHASES, STOP


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    return synthetic.build(root, n_patients=6, signals_per_patient=2, windows_per_signal=3), root


def test_an_item_is_one_channel_of_the_crop_length(built):
    manifest, root = built
    item = WindowDataset(manifest, root, crop=200)[0]
    assert item["x"].shape == (1, 200)
    assert item["y"].shape == (200,)
    assert item["mask"].all()


def test_a_short_window_is_padded_and_the_padding_is_masked(built):
    manifest, root = built
    item = WindowDataset(manifest, root, crop=320)[0]
    assert item["mask"].sum() == 200
    assert not item["mask"][200:].any()


def test_normalising_uses_the_window_only(built):
    manifest, root = built
    values = WindowDataset(manifest, root, crop=200)[0]["x"].numpy().ravel()
    assert abs(values.mean()) < 1e-5
    assert abs(values.std() - 1.0) < 1e-3


def test_a_flat_window_is_not_divided_by_its_own_noise(built):
    manifest, root = built
    dataset = WindowDataset(manifest, root, crop=200)
    assert np.allclose(dataset._normalise(np.zeros(50, dtype=np.float32)), 0.0)


def test_time_warp_keeps_labels_on_class_indices():
    values = np.linspace(0, 1, 20).astype(np.float32)
    target = np.array([INHALE] * 10 + [EXHALE] * 10)
    warped, labels = time_warp(values, target, (1.7, 1.7), np.random.default_rng(0))
    assert warped.size == labels.size
    # Nearest-neighbour, never interpolated: a blend of INHALE and EXHALE lands on STOP.
    assert set(labels.tolist()) <= {INHALE, EXHALE}


def test_class_weights_are_capped():
    manifest = pd.DataFrame({"n_unknown": [10_000], "n_inhale": [1], "n_exhale": [1],
                             "n_stop": [1]})
    weights = class_weights(manifest, PHASES, power=1.0, cap=10.0)
    assert weights.max() <= 10.0
    assert weights.min() >= 0.1


def test_the_manifest_is_rebuilt_from_every_shard_present(tmp_path):
    """A second run - the other environment, or more patients - extends the set.

    Read from the directory rather than from the run's own selection, so an index written by one
    run still names the shards a previous one left beside it.
    """
    from phase.building import read_windows

    synthetic.build(tmp_path, n_patients=2, signals_per_patient=1, windows_per_signal=2)
    rebuilt = read_windows(tmp_path)
    assert len(rebuilt) == 4
    assert set(rebuilt["env"]) == {synthetic.ENV}
    assert rebuilt["shard"].str.startswith(synthetic.ENV).all()


def test_rebuilding_the_manifest_preserves_an_existing_split(tmp_path):
    """Extending a dataset must not drop the record of where the existing windows went."""
    from phase.building import load_windows, read_windows, write_windows
    from phase.splits import SPLIT_COLUMN

    frame = synthetic.build(tmp_path, n_patients=4, signals_per_patient=1, windows_per_signal=2)
    frame[SPLIT_COLUMN] = ["train"] * (len(frame) - 2) + ["test"] * 2
    write_windows(tmp_path, frame)

    rebuilt = read_windows(tmp_path)
    assert SPLIT_COLUMN in rebuilt.columns
    assert rebuilt[SPLIT_COLUMN].value_counts().to_dict() == {"train": len(frame) - 2, "test": 2}
    assert set(load_windows(tmp_path).columns) >= {SPLIT_COLUMN, "RespirationWindowID"}
