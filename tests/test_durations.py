"""The duration metrics, which are what the network is judged on.

The cases that matter are the exclusions: a span the window boundary cut must not reach either
statistic, a span nobody matched must not be averaged in as a zero error, and a signal with two
breaths must not contribute a median.
"""

from __future__ import annotations

import numpy as np
import pytest

from phase import durations
from phase.labels import EXHALE, INHALE, STOP, UNKNOWN

FPS = 10.0


def window(*runs: tuple[int, int]) -> np.ndarray:
    """`(class, length)` pairs laid end to end."""
    return np.concatenate([np.full(length, value, dtype=np.int64) for value, length in runs])


def item(labelled: np.ndarray, model: np.ndarray, signal: int = 1,
         window_id: int = 1, env: str = "ds_algo") -> dict:
    return {durations.LABEL: labelled, durations.MODEL: model, "fps": FPS, "env": env,
            "signal": signal, "patient": "SL_001", "window_id": window_id}


def test_interior_spans_drops_only_what_the_boundary_cut():
    labels = window((INHALE, 5), (UNKNOWN, 2), (EXHALE, 6), (UNKNOWN, 3), (STOP, 4))
    kept = durations.interior_spans(labels)
    assert [span["phase"] for span in kept] == ["exhale"]


def test_interior_spans_keeps_a_window_that_opens_and_closes_unknown():
    labels = window((UNKNOWN, 2), (INHALE, 5), (EXHALE, 6), (UNKNOWN, 2))
    assert [span["phase"] for span in durations.interior_spans(labels)] == ["inhale", "exhale"]


def test_span_error_is_the_duration_difference_in_seconds():
    labelled = window((UNKNOWN, 2), (INHALE, 10), (UNKNOWN, 4))
    model = window((UNKNOWN, 2), (INHALE, 13), (UNKNOWN, 1))
    pairs, coverage = durations.span_errors([item(labelled, model)], iou_threshold=0.5)
    row = pairs[pairs["phase"] == "inhale"].iloc[0]
    assert row[f"{durations.LABEL}_sec"] == pytest.approx(1.0)
    assert row[f"{durations.MODEL}_sec"] == pytest.approx(1.3)
    assert row[durations.ERROR] == pytest.approx(0.3)
    assert int(coverage[coverage["phase"] == "inhale"]["matched"].iloc[0]) == 1


def test_a_boundary_cut_span_reaches_neither_side():
    # Identical arrays, so any surviving pair would be a zero error and would still shift the n.
    labels = window((INHALE, 10), (UNKNOWN, 2), (EXHALE, 8))
    pairs, coverage = durations.span_errors([item(labels, labels)], iou_threshold=0.5)
    assert pairs.empty
    assert coverage[[durations.LABEL, durations.MODEL, "matched"]].to_numpy().sum() == 0


def test_an_unmatched_span_carries_no_error_but_is_counted():
    labelled = window((UNKNOWN, 2), (INHALE, 10), (UNKNOWN, 20))
    model = window((UNKNOWN, 26), (INHALE, 4), (UNKNOWN, 2))
    pairs, coverage = durations.span_errors([item(labelled, model)], iou_threshold=0.5)
    assert pairs.empty
    inhale = coverage[coverage["phase"] == "inhale"].iloc[0]
    assert (int(inhale[durations.LABEL]), int(inhale[durations.MODEL]),
            int(inhale["matched"])) == (1, 1, 0)


def test_unknown_is_never_measured():
    """An unknown gap between the two phases is production's turn, not a break in the breath."""
    labelled = window((UNKNOWN, 2), (INHALE, 6), (UNKNOWN, 6), (EXHALE, 6), (UNKNOWN, 2))
    pairs, coverage = durations.span_errors([item(labelled, labelled)], iou_threshold=0.5)
    assert set(pairs["phase"]) == {"inhale", "exhale", durations.RATIO}
    assert "unknown" not in set(coverage["phase"])


def _breaths(n: int, inhale: int, exhale: int) -> np.ndarray:
    """`n` complete breaths, padded with unknown so nothing touches the boundary."""
    runs = [(UNKNOWN, 2)]
    for _ in range(n):
        runs += [(INHALE, inhale), (EXHALE, exhale), (STOP, 3)]
    return window(*runs, (UNKNOWN, 2))


def test_signal_median_pools_the_windows_of_one_signal():
    # Two windows of the same signal, the model one sample long on every inhale.
    items = [item(_breaths(3, 10, 15), _breaths(3, 11, 15), signal=7, window_id=index)
             for index in range(2)]
    table = durations.signal_medians(items, min_spans=3)
    inhale = table[table["phase"] == "inhale"].iloc[0]
    assert int(inhale[f"{durations.SPANS}_{durations.LABEL}"]) == 6
    assert inhale[f"{durations.MEDIAN}_{durations.LABEL}"] == pytest.approx(1.0)
    assert inhale[durations.ERROR] == pytest.approx(0.1)
    assert bool(inhale[durations.USABLE])


def test_a_thin_signal_is_marked_rather_than_dropped():
    table = durations.signal_medians([item(_breaths(2, 10, 15), _breaths(2, 10, 15))],
                                    min_spans=3)
    assert len(table) and not table[durations.USABLE].any()
    assert durations.signal_agreement(table)["signals_thin"]["inhale"] == 1


def test_the_two_instances_are_not_pooled_on_a_shared_signal_id():
    """Signal 4 in `ds_algo` and signal 4 in `ds_prod` are different recordings."""
    items = [item(_breaths(3, 10, 15), _breaths(3, 10, 15), signal=4, env="ds_algo"),
             item(_breaths(3, 20, 25), _breaths(3, 20, 25), signal=4, env="ds_prod")]
    table = durations.signal_medians(items, min_spans=3)
    assert len(table[table["phase"] == "inhale"]) == 2


def test_agreement_reports_bias_and_limits_over_the_matched_pairs():
    items = [item(_breaths(3, 10, 15), _breaths(3, 12, 15), signal=index, window_id=index)
             for index in range(4)]
    pairs, coverage = durations.span_errors(items, iou_threshold=0.5)
    stats = durations.span_agreement(pairs, coverage)
    assert stats.loc["inhale", "bias_sec"] == pytest.approx(0.2)
    assert stats.loc["inhale", "mae_sec"] == pytest.approx(0.2)
    # Every error identical, so the limits collapse onto the bias.
    assert stats.loc["inhale", "loa_low"] == pytest.approx(0.2)
    assert stats.loc["inhale", "relative_mae"] == pytest.approx(0.2)
    # Signed, against the labelled span's own length: 0.2 s long on a 1.0 s inhale.
    assert stats.loc["inhale", "relative_bias"] == pytest.approx(0.2)
    assert stats.loc["exhale", "bias_sec"] == pytest.approx(0.0)


def test_a_phase_neither_side_called_is_reported_as_absent_not_as_agreement():
    labels = window((UNKNOWN, 2), (INHALE, 10), (EXHALE, 12), (UNKNOWN, 2))
    pairs, coverage = durations.span_errors([item(labels, labels)], iou_threshold=0.5)
    stats = durations.span_agreement(pairs, coverage)
    assert stats.loc["stop", "n"] == 0
    assert np.isnan(stats.loc["stop", "bias_sec"])


def test_relative_bias_keeps_the_direction_the_absolute_version_drops():
    """One span called long and one called short by the same fraction cancel; the MAE does not."""
    long_call = item(_breaths(1, 10, 15), _breaths(1, 12, 15), signal=1, window_id=1)
    short_call = item(_breaths(1, 10, 15), _breaths(1, 8, 15), signal=2, window_id=2)
    pairs, coverage = durations.span_errors([long_call, short_call], iou_threshold=0.5)
    stats = durations.span_agreement(pairs, coverage)
    assert stats.loc["inhale", "relative_bias"] == pytest.approx(0.0)
    assert stats.loc["inhale", "relative_mae"] == pytest.approx(0.2)


def test_the_ratio_is_exhale_over_the_same_breath_s_inhale():
    labelled = window((UNKNOWN, 2), (INHALE, 10), (EXHALE, 20), (STOP, 3), (UNKNOWN, 2))
    model = window((UNKNOWN, 2), (INHALE, 10), (EXHALE, 15), (STOP, 8), (UNKNOWN, 2))
    pairs, coverage = durations.span_errors([item(labelled, model)], iou_threshold=0.5)
    ratio = pairs[pairs["phase"] == durations.RATIO].iloc[0]
    assert ratio[f"{durations.LABEL}_sec"] == pytest.approx(2.0)      # 2.0 s over 1.0 s
    assert ratio[f"{durations.MODEL}_sec"] == pytest.approx(1.5)      # 1.5 s over 1.0 s
    assert ratio[durations.ERROR] == pytest.approx(-0.5)
    assert int(coverage[coverage["phase"] == durations.RATIO]["matched"].iloc[0]) == 1


def test_a_breath_with_one_unmatched_span_carries_no_ratio():
    """A ratio built from one matched span and one guess is not a reading of that breath."""
    labelled = window((UNKNOWN, 2), (INHALE, 10), (EXHALE, 20), (UNKNOWN, 2))
    # The exhale is called far too short to match at IoU 0.5, so the breath has no ratio.
    model = window((UNKNOWN, 2), (INHALE, 10), (EXHALE, 4), (UNKNOWN, 18))
    pairs, coverage = durations.span_errors([item(labelled, model)], iou_threshold=0.5)
    assert pairs[pairs["phase"] == durations.RATIO].empty
    row = coverage[coverage["phase"] == durations.RATIO].iloc[0]
    assert (int(row[durations.LABEL]), int(row["matched"])) == (1, 0)


def test_the_signal_ratio_is_the_ratio_of_the_medians():
    """Not the median of the ratios - this is the recording-level number."""
    items = [item(_breaths(4, 10, 20), _breaths(4, 10, 15), signal=3, window_id=index)
             for index in range(2)]
    table = durations.signal_medians(items, min_spans=3)
    row = table[table["phase"] == durations.RATIO].iloc[0]
    assert row[f"{durations.MEDIAN}_{durations.LABEL}"] == pytest.approx(2.0)
    assert row[f"{durations.MEDIAN}_{durations.MODEL}"] == pytest.approx(1.5)
    assert bool(row[durations.USABLE])


def test_a_signal_ratio_is_unusable_when_either_phase_is():
    """Two breaths is under the floor, so neither median stands and nor does their ratio."""
    table = durations.signal_medians([item(_breaths(2, 10, 20), _breaths(2, 10, 15))],
                                     min_spans=3)
    row = table[table["phase"] == durations.RATIO].iloc[0]
    assert not bool(row[durations.USABLE])
