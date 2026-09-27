"""The two label corrections. No AWS, no dataset - a correction only reads a target."""

import numpy as np
import pytest

from phase.building import SHARDS_DIR, load_windows, recorrect, write_windows
from phase.corrections import Corrections, blank_edges
from phase.labels import EXHALE, INHALE, STOP, UNKNOWN


def test_nothing_configured_is_the_identity():
    corrections = Corrections.from_config({})
    target = np.array([INHALE] * 5 + [EXHALE] * 5)
    out, forced = corrections.apply(target)
    assert not corrections.enabled and not forced
    assert out.tolist() == target.tolist()


def test_the_two_edge_spans_go_whole():
    target = np.array([INHALE] * 3 + [EXHALE] * 3 + [STOP] * 3 + [INHALE] * 3)
    assert blank_edges(target).tolist() == ([UNKNOWN] * 3 + [EXHALE] * 3 + [STOP] * 3
                                            + [UNKNOWN] * 3)


def test_a_window_opening_with_unknown_keeps_its_first_phase():
    """Nothing was truncated at that end - the labeller looked at it and declined it."""
    target = np.array([UNKNOWN] * 4 + [INHALE] * 3 + [EXHALE] * 3 + [STOP] * 3 + [UNKNOWN] * 2)
    assert blank_edges(target).tolist() == target.tolist()


def test_only_the_end_that_touches_is_blanked():
    target = np.array([INHALE] * 3 + [EXHALE] * 3 + [STOP] * 3 + [UNKNOWN] * 2)
    out = blank_edges(target)
    assert out[:3].tolist() == [UNKNOWN] * 3          # ran to sample 0, so it went
    assert out[3:9].tolist() == [EXHALE] * 3 + [STOP] * 3   # the trailing stop is intact


def test_an_interior_span_is_never_touched():
    target = np.array([INHALE] * 2 + [EXHALE] * 2 + [STOP] * 2 + [INHALE] * 2 + [EXHALE] * 2)
    out = blank_edges(target)
    assert out[2:8].tolist() == [EXHALE, EXHALE, STOP, STOP, INHALE, INHALE]


def test_a_span_spanning_the_whole_window_is_blanked():
    """It touches both boundaries, so both ends of it were cut."""
    assert blank_edges(np.full(6, INHALE)).tolist() == [UNKNOWN] * 6


def test_an_entirely_unknown_window_survives_blanking():
    target = np.zeros(8, dtype=np.int64)
    assert blank_edges(target).tolist() == [UNKNOWN] * 8


def test_window_over_the_threshold_becomes_entirely_unknown():
    corrections = Corrections.from_config({"all_unknown_above": 0.5})
    mostly = np.array([UNKNOWN] * 7 + [INHALE] * 3)
    out, forced = corrections.apply(mostly)
    assert forced and out.tolist() == [UNKNOWN] * 10

    keeps = np.array([UNKNOWN] * 4 + [INHALE] * 6)
    out, forced = corrections.apply(keeps)
    assert not forced and out.tolist() == keeps.tolist()


def test_the_threshold_is_measured_after_the_edge_spans():
    """Blanking the edges adds unknown, so it can be what pushes a window over. Tested so it
    stays that way rather than being read off the labelling as drawn."""
    corrections = Corrections.from_config({"blank_edge_spans": True, "all_unknown_above": 0.5})
    # 20% unknown as drawn, 70% once the two edge spans go.
    target = np.array([INHALE] * 3 + [EXHALE] * 3 + [UNKNOWN] * 2 + [STOP] * 2)
    out, forced = corrections.apply(target)
    assert forced and out.tolist() == [UNKNOWN] * 10


def test_the_input_is_never_modified():
    target = np.array([INHALE] * 3 + [EXHALE] * 3 + [STOP] * 3)
    Corrections.from_config({"blank_edge_spans": True}).apply(target)
    assert target.tolist() == [INHALE] * 3 + [EXHALE] * 3 + [STOP] * 3


def test_an_empty_window_survives():
    out, forced = Corrections.from_config({"all_unknown_above": 0.5}).apply(
        np.array([], dtype=np.int64))
    assert out.size == 0 and not forced


@pytest.mark.parametrize("cfg", [{"all_unknown_above": 1.0}, {"all_unknown_above": -0.1}])
def test_a_nonsense_setting_is_refused(cfg):
    with pytest.raises(ValueError):
        Corrections.from_config(cfg)


def test_describe_round_trips():
    corrections = Corrections.from_config({"blank_edge_spans": True, "all_unknown_above": 0.6})
    assert Corrections.from_config(corrections.describe()) == corrections


def test_recorrect_derives_a_dataset_without_the_database(tmp_path):
    """The point of keeping `targets_drawn`: a new corrections setting costs a file copy, not a
    round-trip per window."""
    from tests import synthetic

    source = tmp_path / "source"
    drawn = synthetic.build(source, n_patients=2, signals_per_patient=1, windows_per_signal=2)
    drawn["split"] = "train"
    write_windows(source, drawn)

    out = tmp_path / "blanked"
    frame, forced = recorrect(source, out, Corrections.from_config({"blank_edge_spans": True}),
                              log=lambda *_: None)

    assert len(frame) == len(drawn) and forced == 0
    assert frame["n_unknown"].sum() > drawn["n_unknown"].sum()
    # Split assignment is carried, not recomputed: the windows did not move, only the target.
    assert set(load_windows(out)["split"]) == {"train"}


def test_recorrect_refuses_a_shard_without_the_drawn_target(tmp_path):
    # Refused only where the source dataset had corrections of its own: the drawn target is
    # then genuinely gone.
    (tmp_path / SHARDS_DIR).mkdir(parents=True)
    np.savez_compressed(tmp_path / SHARDS_DIR / "ds_algo_signal_1.npz",
                        targets=np.zeros(4, dtype=np.int8),
                        offsets=np.array([0, 4]), analysis_fps=np.array([10.0]))
    with pytest.raises(KeyError, match="targets_drawn"):
        recorrect(tmp_path, tmp_path / "out", Corrections.from_config({}),
                  previous=Corrections.from_config({"blank_edge_spans": True}),
                  log=lambda *_: None)


def test_recorrect_falls_back_to_targets_where_nothing_was_corrected(tmp_path):
    """Every dataset built before this feature has corrections off, so its stored target is the
    labelling as drawn and no rebuild is needed to correct it."""
    from tests import synthetic

    source = tmp_path / "source"
    drawn = synthetic.build(source, n_patients=1, signals_per_patient=1, windows_per_signal=2)
    for shard in (source / SHARDS_DIR).glob("*.npz"):
        with np.load(shard, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files if name != "targets_drawn"}
        np.savez_compressed(shard, **arrays)

    frame, _ = recorrect(source, tmp_path / "out",
                         Corrections.from_config({"blank_edge_spans": True}),
                         previous=Corrections.from_config({}), log=lambda *_: None)
    assert frame["n_unknown"].sum() > drawn["n_unknown"].sum()


def test_merging_stop_turns_every_pause_into_exhale():
    corrections = Corrections.from_config({"merge_stop_into_exhale": True})
    target = np.array([UNKNOWN] * 2 + [INHALE] * 3 + [EXHALE] * 3 + [STOP] * 3 + [UNKNOWN] * 2)
    out, forced = corrections.apply(target)
    assert corrections.enabled and not forced
    assert out.tolist() == [UNKNOWN] * 2 + [INHALE] * 3 + [EXHALE] * 6 + [UNKNOWN] * 2


def test_the_edge_blank_reads_the_merged_spans():
    """A trailing exhale + stop is one exhale once merged, so it goes whole."""
    corrections = Corrections.from_config({"merge_stop_into_exhale": True,
                                           "blank_edge_spans": True})
    target = np.array([UNKNOWN] * 2 + [INHALE] * 3 + [EXHALE] * 3 + [STOP] * 3)
    out, _ = corrections.apply(target)
    assert out.tolist() == [UNKNOWN] * 2 + [INHALE] * 3 + [UNKNOWN] * 6

