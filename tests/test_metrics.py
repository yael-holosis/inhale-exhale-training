"""The metrics, on hand-built sequences where the right answer is obvious."""

import numpy as np

from phase.labels import EXHALE, INHALE, STOP, UNKNOWN
from phase.metrics import event_level, per_sample, phase_durations

TRUTH = np.array([UNKNOWN, INHALE, INHALE, INHALE, UNKNOWN, EXHALE, EXHALE, EXHALE, STOP, STOP])


def test_a_perfect_prediction_scores_one():
    scores = per_sample(TRUTH, TRUTH)
    assert scores["macro_f1"] == 1.0
    assert scores["accuracy"] == 1.0


def test_macro_f1_excludes_unknown():
    # Unknown is most of a real window; a model that only ever says unknown must not score well.
    only_unknown = np.zeros_like(TRUTH)
    assert per_sample(only_unknown, TRUTH)["macro_f1"] == 0.0


def test_the_mask_removes_samples_from_the_score():
    wrong = TRUTH.copy()
    wrong[:5] = UNKNOWN
    mask = np.zeros(len(TRUTH), dtype=bool)
    mask[5:] = True
    assert per_sample(wrong, TRUTH, mask)["accuracy"] == 1.0


def test_a_shredded_segment_loses_the_event_but_keeps_most_samples():
    shredded = TRUTH.copy()
    shredded[2] = UNKNOWN                       # one inhale becomes two
    assert per_sample(shredded, TRUTH)["f1_inhale"] > 0.7
    assert event_level(shredded, TRUTH)["event_recall_inhale"] == 0.0


def test_boundary_error_is_reported_in_samples():
    # Padded, so shifting does not push the last segment off the end and clip its boundary.
    truth = np.concatenate((TRUTH, [UNKNOWN, UNKNOWN]))
    shifted = np.concatenate(([UNKNOWN], truth[:-1]))
    assert event_level(shifted, truth, iou_threshold=0.3)["boundary_mae_samples"] == 1.0


def test_durations_convert_by_the_analysis_rate():
    durations = phase_durations(TRUTH, fps=10.0)
    assert durations["mean_inhale_sec"] == 0.3
    assert durations["mean_exhale_sec"] == 0.3
    assert durations["ie_ratio"] == 1.0
