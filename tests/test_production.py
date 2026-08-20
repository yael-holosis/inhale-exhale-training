"""The production phase call, on a synthetic breath - `holosissystem` only, no AWS.

The assertion that matters is the last one: the spans this module pairs must average back to the
durations production itself returned. That is what says the pairing reads production's boundary
arrays correctly rather than plausibly.
"""

import numpy as np
import pytest

from phase import production
from phase.labels import PHASES, spans_to_targets

FPS = 10.0
RATE_BPM = 15.0


def synthetic_trace(cycles: int = 8, rate_bpm: float = RATE_BPM,
                    rise_fraction: float = 0.35) -> np.ndarray:
    """Asymmetric breaths - a short rise, a longer fall, a pause at the trough."""
    period = int(round(60.0 / rate_bpm * FPS))
    rise = max(2, int(round(period * rise_fraction)))
    pause = max(1, int(round(period * 0.2)))
    fall = max(2, period - rise - pause)
    one = np.concatenate([np.sin(np.linspace(-np.pi / 2, np.pi / 2, rise)),
                          np.sin(np.linspace(np.pi / 2, -np.pi / 2, fall)),
                          np.full(pause, -1.0)])
    return np.tile(one, cycles)


@pytest.fixture(scope="module")
def call():
    return production.run(synthetic_trace(), RATE_BPM, FPS)


def test_production_parameters_are_read_from_the_installed_package():
    params = production.respiration_params()
    assert isinstance(params, dict) and params
    # A fresh copy each time - the callee mutates window size.
    assert production.respiration_params() is not params


def test_the_call_reaches_its_boundary_arrays(call):
    assert call.reached_boundaries
    assert call.time_vec.size
    assert call.crest_idx.size and call.trough_idx.size


def test_the_durations_are_production_s_own(call):
    assert call.inhale_time_sec is not None
    assert call.exhale_time_sec is not None
    assert not call.rejected


def test_spans_are_named_from_the_shared_vocabulary(call):
    assert {row["phase"] for row in production.spans_of(call)} <= set(PHASES)


def test_spans_lie_inside_the_trace_and_run_forwards(call):
    length = call.given.size
    for row in production.spans_of(call):
        assert 0 <= row["start"] < row["end"] <= length


def test_the_stop_sits_between_an_exhale_and_the_next_inhale(call):
    rows = production.spans_of(call)
    for before, current, after in zip(rows, rows[1:], rows[2:]):
        if current["phase"] == "stop":
            assert before["phase"] == "exhale"
            assert after["phase"] == "inhale"


def test_the_crest_is_left_unclaimed():
    """Production emits no phase for the turn from inhale to exhale, so `unknown` is structural
    here - it is not only where the trace could not be called."""
    call = production.run(synthetic_trace(), RATE_BPM, FPS)
    target = spans_to_targets(production.spans_of(call), call.given.size)
    assert (target == 0).any()


def test_the_paired_spans_average_back_to_production_s_durations(call):
    """The one derived step in the whole module, checked against the function it came from.

    Over **all** pairs, not the valid ones: production's own mean includes the pairs `_valid`
    rejects, so filtering first would compare two different quantities. Exact, not approximate -
    if the pairing is right these are the same arithmetic on the same numbers.
    """
    ours = production.durations_sec(call, production.spans_of(call, valid_only=False))

    # Production's `inhale_time_sec` describes the trace it *returned*. Where that is the
    # negation of its input, our spans have been re-labelled onto the input - so its inhale is
    # our exhale. That swap is the whole point of the mapping; here it is asserted rather than
    # trusted.
    theirs = {"inhale": call.inhale_time_sec, "exhale": call.exhale_time_sec}
    if call.inverted:
        theirs = {"inhale": theirs["exhale"], "exhale": theirs["inhale"]}

    for phase, reported in theirs.items():
        assert ours[phase] == pytest.approx(reported, abs=1e-9), (
            f"{phase}: paired {ours[phase]:.6f} s vs production's {reported:.6f} s")


def test_the_valid_filter_only_ever_removes_spans(call):
    kept = production.spans_of(call, valid_only=True)
    everything = production.spans_of(call, valid_only=False)
    assert len(kept) <= len(everything)


def test_a_window_with_no_rate_is_reported_not_guessed():
    rows, note = production.phases_for(synthetic_trace(), None, FPS)
    assert rows == []
    assert "no stored rate" in note


def test_inversion_is_measured_rather_than_read_off_the_flag(call):
    """Three sign decisions compose inside the function; only the net result matters here."""
    assert call.inverted == bool(np.allclose(call.returned, -call.given))


def test_a_rise_on_the_labelled_trace_is_always_inhale(call):
    """The property the whole mapping exists to guarantee: a label describes the array beside it.

    Checked on the samples that were passed in, which are the samples that get stored.
    """
    trace = call.given
    for row in production.spans_of(call):
        if row["phase"] == "inhale":
            assert trace[row["end"]] > trace[row["start"]], f"inhale falls at {row}"
        elif row["phase"] == "exhale":
            assert trace[row["end"]] < trace[row["start"]], f"exhale rises at {row}"
