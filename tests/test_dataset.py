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


def test_a_figure_is_drawn_and_is_not_empty(tmp_path):
    """The plotting path, end to end, without a checkpoint or AWS."""
    from phase import figures

    manifest = synthetic.build(tmp_path, n_patients=2, signals_per_patient=1,
                               windows_per_signal=2)
    dataset = WindowDataset(manifest, tmp_path, crop=200)
    items = []
    for index in range(len(dataset)):
        values, reference = dataset._window(index)
        prediction = reference.copy()
        prediction[:20] = 0                       # a disagreement to draw
        items.append({"values": values, "reference": reference, "prediction": prediction,
                      "fps": 10.0, "title": f"window {index}",
                      "score": figures.score(prediction, reference)})
    path = figures.plot_windows(items[:2], tmp_path / "figure.png", "test")
    assert path.exists() and path.stat().st_size > 10_000


def test_spread_takes_the_worst_and_the_best():
    from phase import figures

    scored = [{"score": value} for value in (0.1, 0.5, 0.9, 0.3, 0.7)]
    chosen = figures.choose(scored, 3, figures.SPREAD)
    assert [item["score"] for item in chosen] == [0.1, 0.5, 0.9]
    assert [item["score"] for item in figures.choose(scored, 2, figures.WORST)] == [0.1, 0.3]
    assert [item["score"] for item in figures.choose(scored, 2, figures.BEST)] == [0.9, 0.7]


def test_choosing_more_than_there_are_is_not_an_error():
    from phase import figures

    assert len(figures.choose([{"score": 0.5}], 10, figures.SPREAD)) == 1
    assert figures.choose([], 5) == []


def test_a_window_keeps_its_own_length_when_there_is_no_crop(built):
    manifest, root = built
    dataset = WindowDataset(manifest, root, crop=None)
    values, _ = dataset._window(0)
    item = dataset[0]
    assert item["x"].shape == (1, values.size)
    assert item["mask"].all()


def test_collate_pads_a_ragged_batch_and_masks_the_padding():
    """Windows differ in length; nothing may be cropped to fit a tensor."""
    import torch

    from phase.dataset import collate

    batch = [{"x": torch.ones(1, n), "y": torch.full((n,), 2, dtype=torch.long),
              "mask": torch.ones(n, dtype=torch.bool), "row": index}
             for index, n in enumerate((200, 350, 250))]
    out = collate(batch)
    assert out["x"].shape == (3, 1, 350)
    assert out["mask"].sum().item() == 200 + 350 + 250
    # Every real sample is kept, and every padded one is masked out.
    for position, n in enumerate((200, 350, 250)):
        assert out["mask"][position, :n].all()
        assert not out["mask"][position, n:].any()
        assert (out["x"][position, 0, n:] == 0).all()


def test_the_model_takes_a_ragged_batch_after_collate():
    import torch

    from models.unet1d import UNet1D
    from phase.dataset import collate

    batch = [{"x": torch.randn(1, n), "y": torch.zeros(n, dtype=torch.long),
              "mask": torch.ones(n, dtype=torch.bool), "row": 0} for n in (200, 450)]
    out = collate(batch)
    logits = UNet1D(channels=(16, 24, 32), bottleneck=48)(out["x"])
    assert logits.shape == (2, 4, 450)


def test_bucketed_batches_are_all_one_length():
    """A batch drawn from one length group needs no padding at all."""
    import numpy as np

    from phase.dataset import LengthBucketSampler

    lengths = np.array([200] * 130 + [250] * 20 + [400] * 3)
    for batch in LengthBucketSampler(lengths, 64, shuffle=True, seed=0):
        assert len(set(lengths[batch].tolist())) == 1


def test_bucketing_covers_every_window_exactly_once():
    import numpy as np

    from phase.dataset import LengthBucketSampler

    lengths = np.array([200] * 130 + [250] * 20 + [400] * 3)
    seen = [index for batch in LengthBucketSampler(lengths, 64, seed=0) for index in batch]
    assert sorted(seen) == list(range(len(lengths)))


def test_dropping_the_last_batch_would_lose_whole_lengths_so_it_is_off_by_default():
    """Bucketed, a short batch is the whole of a rare length rather than a remainder.

    On the real set `drop_last` would discard every window of 300 samples and longer - the
    slowest, most irregular breathing, and the reason variable-length training exists.
    """
    import numpy as np

    from phase.dataset import LengthBucketSampler

    lengths = np.array([200] * 130 + [400] * 3)
    kept = {index for batch in LengthBucketSampler(lengths, 64, seed=0) for index in batch}
    assert len(kept) == len(lengths)

    dropped = {index for batch in LengthBucketSampler(lengths, 64, drop_last=True, seed=0)
               for index in batch}
    assert not any(lengths[index] == 400 for index in dropped), \
        "drop_last silently removed an entire length group"


def test_drift_stays_under_the_breathing_it_is_meant_to_sit_beneath():
    """`drift_scale` is a fraction of the window, not an absolute amplitude.

    Radar windows carry a std of ~3e-4, so an absolute 0.3 was 315x the breathing and erased it
    on every window - while keeping the phase labels that described it.
    """
    values = (1e-4 * np.sin(np.linspace(0, 12 * np.pi, 200))).astype(np.float32)
    target = np.zeros(200, dtype=np.int64)
    dataset = WindowDataset(pd.DataFrame(), ".", augment={"drift_prob": 1.0, "drift_scale": 0.3})

    drifted, _ = dataset._augment(values.copy(), target, np.random.default_rng(0))
    added = drifted - values
    assert added.std() < values.std(), (
        f"drift {added.std():.2e} must sit under the signal {values.std():.2e}")


def test_shards_are_read_from_the_subdirectory_and_from_the_old_flat_layout(tmp_path):
    """A dataset built before the shards moved must still load - `shard_path` decides, not the
    caller, so nothing has to know which layout it is looking at."""
    from phase.building import SHARDS_DIR, shard_path

    (tmp_path / SHARDS_DIR).mkdir()
    (tmp_path / SHARDS_DIR / "nested.npz").write_bytes(b"")
    (tmp_path / "flat.npz").write_bytes(b"")

    assert shard_path(tmp_path, "nested.npz") == tmp_path / SHARDS_DIR / "nested.npz"
    assert shard_path(tmp_path, "flat.npz") == tmp_path / "flat.npz"
    # Absent either way, the subdirectory is not silently preferred into a missing file.
    assert shard_path(tmp_path, "gone.npz") == tmp_path / "gone.npz"


def test_an_absent_class_leaves_the_balance_of_the_others_alone():
    """`stop` merged away: its zero count must not reweight the three classes still in play."""
    three = pd.DataFrame({"n_unknown": [300], "n_inhale": [200], "n_exhale": [500]})
    merged = three.assign(n_stop=[0])
    weights = class_weights(merged, PHASES, power=0.5, cap=10.0)
    alone = class_weights(three, ("unknown", "inhale", "exhale"), power=0.5, cap=10.0)
    assert weights[:3].tolist() == pytest.approx(alone.tolist())
    assert weights[3] == 1.0


def test_a_one_sample_sliver_does_not_sink_a_window_score():
    """Window 61 of signal 1570218: one labelled `unknown` sample took macro F1 to 0.64."""
    from phase import figures

    reference = np.r_[[0], np.tile(np.r_[np.full(20, INHALE), np.full(30, EXHALE)], 4)[:199]]
    prediction = reference.copy()
    prediction[0] = INHALE
    assert figures.score(prediction, reference) == pytest.approx(0.995)
    assert figures.macro_f1(prediction, reference) < 0.7

