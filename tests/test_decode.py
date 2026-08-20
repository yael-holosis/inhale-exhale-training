"""The decoder: impossible transitions cannot survive, and neither can one-sample phases."""

import numpy as np

from phase.decode import decode, enforce_min_duration, transition_matrix
from phase.labels import EXHALE, INHALE, PHASE_IDS, STOP, UNKNOWN

ALLOWED = {"unknown": ["inhale", "exhale", "stop"], "inhale": ["unknown"],
           "exhale": ["unknown", "stop"], "stop": ["unknown", "inhale"]}


def _logits(sequence, confidence=8.0):
    out = np.zeros((len(sequence), len(PHASE_IDS)))
    out[np.arange(len(sequence)), sequence] = confidence
    return out


def test_self_transitions_are_free_and_forbidden_ones_are_infinite():
    cost = transition_matrix(ALLOWED, 2.0)
    assert cost[INHALE, INHALE] == 0.0
    assert cost[INHALE, UNKNOWN] == -2.0
    assert cost[INHALE, EXHALE] == -np.inf         # production emits no phase for that turn


def test_viterbi_never_emits_a_forbidden_transition():
    cost = transition_matrix(ALLOWED, 2.0)
    path = decode(_logits([INHALE] * 5 + [EXHALE] * 5), cost)
    for before, after in zip(path, path[1:]):
        assert cost[before, after] > -np.inf


def test_a_single_stray_sample_is_absorbed():
    path = np.array([INHALE] * 6 + [EXHALE] + [INHALE] * 6)
    assert set(enforce_min_duration(path, {"inhale": 3, "exhale": 3}).tolist()) == {INHALE}


def test_a_run_at_exactly_the_floor_survives():
    path = np.array([INHALE] * 5 + [STOP] * 3 + [INHALE] * 5)
    assert enforce_min_duration(path, {"inhale": 3, "stop": 3}).tolist() == path.tolist()


def test_decoding_without_a_cost_matrix_is_argmax():
    sequence = [UNKNOWN, INHALE, INHALE, EXHALE]
    assert decode(_logits(sequence)).tolist() == sequence
