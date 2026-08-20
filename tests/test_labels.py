"""The label vocabulary and the span round trip. No AWS, no torch."""

import numpy as np
import pytest

from phase.labels import (EXHALE, INHALE, N_CLASSES, PHASES, STOP, SWAP_ON_FLIP, UNKNOWN,
                          class_counts, spans_to_targets, targets_to_spans)


def test_unknown_is_zero_so_an_unclaimed_sample_is_unknown():
    assert UNKNOWN == 0
    assert spans_to_targets([], 5).tolist() == [0] * 5


def test_gaps_between_spans_stay_unknown():
    target = spans_to_targets([{"phase": "inhale", "start": 0, "end": 2},
                               {"phase": "exhale", "start": 4, "end": 6}], 8)
    assert target.tolist() == [INHALE, INHALE, UNKNOWN, UNKNOWN, EXHALE, EXHALE, UNKNOWN, UNKNOWN]


def test_spans_are_clipped_not_dropped():
    target = spans_to_targets([{"phase": "stop", "start": -2, "end": 12}], 4)
    assert target.tolist() == [STOP] * 4


def test_an_unknown_phase_name_fails_loudly():
    with pytest.raises(KeyError):
        spans_to_targets([{"phase": "sigh", "start": 0, "end": 1}], 3)


def test_span_round_trip():
    spans = [{"phase": "inhale", "start": 1, "end": 4}, {"phase": "exhale", "start": 4, "end": 7}]
    target = spans_to_targets(spans, 9)
    back = [s for s in targets_to_spans(target) if s["phase"] != "unknown"]
    assert back == spans


def test_flip_exchanges_inhale_and_exhale_and_leaves_stop_alone():
    assert SWAP_ON_FLIP[INHALE] == EXHALE
    assert SWAP_ON_FLIP[EXHALE] == INHALE
    # The stop is defined as the stretch between an exhale ending and the next inhale starting,
    # which is the trough of whichever way up the window is held.
    assert SWAP_ON_FLIP[STOP] == STOP
    assert SWAP_ON_FLIP[UNKNOWN] == UNKNOWN


def test_flipping_twice_is_the_identity():
    target = np.array([UNKNOWN, INHALE, EXHALE, STOP])
    assert SWAP_ON_FLIP[SWAP_ON_FLIP[target]].tolist() == target.tolist()


def test_class_counts_covers_every_phase():
    counts = class_counts(np.array([INHALE, INHALE, UNKNOWN]))
    assert set(counts) == set(PHASES)
    assert sum(counts.values()) == 3
    assert len(PHASES) == N_CLASSES
