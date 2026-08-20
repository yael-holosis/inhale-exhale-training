"""The dataset and the splits, on a synthetic set - no AWS, no shards from a real build."""

import numpy as np
import pandas as pd
import pytest

from tests import synthetic
from phase.dataset import WindowDataset, class_weights, time_warp
from phase.labels import EXHALE, INHALE, PHASES, STOP
from phase.splits import patient_folds, split_for


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


def test_no_patient_appears_on_both_sides_of_a_split(built):
    manifest, _ = built
    folded = patient_folds(manifest, n_folds=3, seed=0)
    splits = split_for(folded, fold=0, val_fraction=0.3, seed=0)
    train = set(splits["train"]["PatientID"])
    assert not train & set(splits["test"]["PatientID"])
    assert not train & set(splits["val"]["PatientID"])
    assert not set(splits["val"]["PatientID"]) & set(splits["test"]["PatientID"])


def test_every_window_lands_in_exactly_one_fold(built):
    manifest, _ = built
    folded = patient_folds(manifest, n_folds=3, seed=0)
    assert (folded["fold"] >= 0).all()
    assert folded.groupby("PatientID")["fold"].nunique().max() == 1


def test_more_folds_than_patients_is_refused(built):
    manifest, _ = built
    with pytest.raises(ValueError):
        patient_folds(manifest, n_folds=99)
