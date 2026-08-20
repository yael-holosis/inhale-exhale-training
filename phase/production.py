"""Production's inhale/exhale answer for one window, and nothing else.

`holosissystem.analysis.RespirationAnalysis.calculate_inhale_exhale_time` is the device's phase
calculation. It returns three mean durations and a trace; the per-breath boundary indices it
averaged over never leave the function. This module calls it **unmodified** and captures its
frame while it runs, then pairs the boundary arrays into spans.

No phase logic is written here. The boundary indices, the turning points, the polarity decision
and the durations are all production's own values. The one derived step is the pairing, because
production reduces the per-breath differences to a mean inline and never holds them - and
`tests/test_production.py` asserts the pairs average back to the durations production returned.

Ported from `inhale-exhale-detection` (`production_trace.py`, `phase_detection.py`) and the
labelling app's `suggestion.py`, so this repo stands on `holosissystem` alone. If the phase code
upstream changes, this is the file that has to follow it.

## What the answer is, and is not

- **The boundaries are the 10% and 90% amplitude crossings** - rise and fall times, a median 65%
  of the true trough-to-crest rise. They are what production reports, not what a breath is.
- **There is no phase for the turn from inhale to exhale.** Production emits none, so the crest
  of every breath comes back unclaimed and lands in `unknown`.
- **The stop is re-derived, not read.** See `_pause_between_breaths`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from phase.labels import PHASES, EXHALE, INHALE, STOP

PHASE_INHALE, PHASE_EXHALE, PHASE_STOP = PHASES[INHALE], PHASES[EXHALE], PHASES[STOP]
SWAP = {PHASE_INHALE: PHASE_EXHALE, PHASE_EXHALE: PHASE_INHALE}

# Names of the locals in `calculate_inhale_exhale_time` that the pairing reads.
GO_UP_PEAKS = "go_up_peaks"
GO_DOWN_PEAKS = "go_down_peaks"
GO_UP_90 = "go_up_90"
GO_DOWN_10 = "go_down_10"
FLIP_FLAG = "flip_data_flag"
TIME_VEC = "time_vec"

PARAMS_FILE = "parameters/parameters.yaml"
PARAMS_ROOT_KEY = "US"
RESPIRATION_KEY = "Respiration"


# ------------------------------------------------------------------ holosissystem, as installed

@lru_cache(maxsize=1)
def _all_params() -> dict[str, Any]:
    import holosissystem

    with open(Path(holosissystem.__file__).resolve().parent / PARAMS_FILE) as handle:
        return yaml.safe_load(handle)


def respiration_params() -> dict[str, Any]:
    """Production's `US.Respiration` block. A fresh copy - the callee mutates window size."""
    return dict(_all_params()[PARAMS_ROOT_KEY][RESPIRATION_KEY])


@lru_cache(maxsize=1)
def analysis():
    """A `RespirationAnalysis` on production's own parameters."""
    from holosissystem.analysis import RespirationAnalysis

    return RespirationAnalysis(respiration_params=respiration_params())


def version() -> str:
    import importlib.metadata as metadata

    try:
        return metadata.version("holosissystem")
    except Exception:                                                     # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------------- capturing the frame

class Locals:
    """Snapshots of one call's frame, in execution order.

    Taken whenever a name first appears and again on return. That is enough to read both sides
    of an in-place rewrite - `first` gives the array before it, the return snapshot after -
    without copying arrays on every line of a long loop.
    """

    def __init__(self, history: list[dict[str, Any]]):
        self.history = history

    def first(self, *keys: str) -> dict[str, Any] | None:
        for snapshot in self.history:
            if all(key in snapshot for key in keys):
                return snapshot
        return None

    def last(self, *keys: str) -> dict[str, Any] | None:
        for snapshot in reversed(self.history):
            if all(key in snapshot for key in keys):
                return snapshot
        return None

    def value(self, key: str, default: Any = None) -> Any:
        snapshot = self.last(key)
        return default if snapshot is None else snapshot[key]


def _snapshot(frame) -> dict[str, Any]:
    out = {}
    for key, value in frame.f_locals.items():
        out[key] = value.copy() if isinstance(value, np.ndarray) else (
            list(value) if isinstance(value, list) else value)
    return out


def call_with_locals(function: Callable, *args, **kwargs) -> tuple[Any, Locals]:
    """Call `function` unmodified and return `(result, Locals)`.

    Only its own frame is traced; nested calls run at full speed.
    """
    code = getattr(function, "__func__", function).__code__
    history: list[dict[str, Any]] = []
    seen: set[str] = set()

    def tracer(frame, event, _arg):
        if frame.f_code is not code:
            return None
        if event == "line":
            if not frame.f_locals.keys() <= seen:
                history.append(_snapshot(frame))
                seen.update(frame.f_locals.keys())
        elif event == "return":
            history.append(_snapshot(frame))
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = function(*args, **kwargs)
    finally:
        sys.settrace(previous)
    return result, Locals(history)


@dataclass
class PhaseCall:
    """One `calculate_inhale_exhale_time` call: what went in, what came out, and its internals.

    `go_up_90` and `go_down_10` are read **as production first computed them**, because it
    rewrites `go_down_10` in place on a flipped window. `pause_boundaries` is the rewritten
    array, which is what its stop time was measured on.
    """

    given: np.ndarray                # the trace handed in, before any flip of production's own
    resp_rate_hz: float
    fps: int
    inhale_time_sec: float | None    # exactly what production returned
    exhale_time_sec: float | None
    stop_time_sec: float | None
    returned: np.ndarray             # production's returned trace, which may be negated
    locals_: Locals

    def _at_boundaries(self) -> dict[str, Any] | None:
        return self.locals_.first(GO_UP_90, GO_DOWN_10)

    @property
    def reached_boundaries(self) -> bool:
        """False when production returned early for want of turning points."""
        return self._at_boundaries() is not None

    @property
    def flipped(self) -> bool:
        return bool(self.locals_.value(FLIP_FLAG, 0))

    @property
    def rejected(self) -> bool:
        return self.inhale_time_sec is None

    @property
    def time_vec(self) -> np.ndarray:
        vector = self.locals_.value(TIME_VEC)
        return np.array([]) if vector is None else np.asarray(vector, dtype=float)

    def _boundary(self, key: str) -> np.ndarray | None:
        snapshot = self._at_boundaries()
        return None if snapshot is None else np.asarray(snapshot[key])

    @property
    def crest_idx(self) -> np.ndarray:
        return np.array([], dtype=int) if not self.reached_boundaries else \
            np.asarray(self._at_boundaries()[GO_UP_PEAKS], int)

    @property
    def trough_idx(self) -> np.ndarray:
        return np.array([], dtype=int) if not self.reached_boundaries else \
            np.asarray(self._at_boundaries()[GO_DOWN_PEAKS], int)

    @property
    def rise_boundaries(self) -> np.ndarray | None:
        """`go_up_90` - the 90% points, as first computed."""
        return self._boundary(GO_UP_90)

    @property
    def fall_boundaries(self) -> np.ndarray | None:
        """`go_down_10` before production's flipped-window rewrite."""
        return self._boundary(GO_DOWN_10)

    @property
    def inverted(self) -> bool:
        """Whether the trace production returned is the negation of the one it was given.

        Measured, not read off `flipped`: three sign decisions compose inside the function and
        only the net result matters for putting the answer back on the caller's picture.
        """
        return (self.returned.shape == self.given.shape
                and bool(np.allclose(self.returned, -self.given)))


def run(values: np.ndarray, rate_bpm: float, fps: float) -> PhaseCall:
    """Production's phase calculation on one window's samples."""
    given = np.asarray(values, dtype=float)
    resp = analysis()
    result, captured = call_with_locals(resp.calculate_inhale_exhale_time, given,
                                        float(rate_bpm) / 60.0, respiration_params(),
                                        int(round(fps)))
    inhale, exhale, stop, returned = result
    return PhaseCall(given=given, resp_rate_hz=float(rate_bpm) / 60.0, fps=int(round(fps)),
                     inhale_time_sec=inhale, exhale_time_sec=exhale, stop_time_sec=stop,
                     returned=np.asarray(returned, dtype=float), locals_=captured)


# -------------------------------------------------------------------------- pairing into spans

def _pairs(call: PhaseCall) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Production's own index expressions - the pairs its means are taken over.

    From `analysis.py`, the slices `calculate_inhale_exhale_time` averages: the 90% point on the
    rising side of a crest, the point leaving the trough before it, and the mirror pair on the
    falling side.
    """
    crest, trough = call.crest_idx, call.trough_idx
    n_up, n_down = crest.size, trough.size
    rise, fall = call.rise_boundaries, call.fall_boundaries

    rise_end = rise[0:2 * n_up:2]                # 90% of crest, rising side
    rise_start = fall[1:2 * n_up:2]              # leaving the trough
    fall_start = rise[1:2 * n_down - 1:2]        # 90% of crest, falling side
    fall_end = fall[2:2 * n_down:2]              # arriving at the next trough

    # Inhale always rises on the trace production *returned*: when it flips a window, the rising
    # edges of `data` become exhale and the returned trace is negated.
    if call.flipped:
        return list(zip(fall_start, fall_end)), list(zip(rise_start, rise_end))
    return list(zip(rise_start, rise_end)), list(zip(fall_start, fall_end))


def _valid(start: int, end: int, time_vec: np.ndarray, cycle_sec: float) -> bool:
    """`find_closest_points` leaves an endpoint at 0 when no sample came close enough, which
    yields spans running backwards or spanning several breaths. Those poison the reported means
    and are dropped rather than trained on."""
    if end <= start or (start == 0 and end == 0):
        return False
    if start >= time_vec.size or end >= time_vec.size:
        return False
    return (float(time_vec[end]) - float(time_vec[start])) <= cycle_sec


def spans_of(call: PhaseCall, valid_only: bool = True) -> list[dict[str, Any]]:
    """Production's phases for one window, as `{phase, start, end}` rows on the trace it was
    **given** - so the spans and the stored samples describe the same picture.

    Where production returned the negation of its input, a rise there is a fall here, so inhale
    and exhale exchange places. That is undone rather than passed on: a label has to describe the
    array it sits beside.

    `valid_only=False` keeps the pairs `_valid` rejects. That is not a training setting - it is
    how production's own mean is taken, so it is the only way to check the pairing reproduces
    the durations the function returned. `tests/test_production.py` does exactly that.
    """
    if not call.reached_boundaries or call.rejected:
        return []
    time_vec = call.time_vec
    if not time_vec.size:
        return []

    inhale_pairs, exhale_pairs = _pairs(call)
    cycle_sec = 1.0 / call.resp_rate_hz if call.resp_rate_hz else float("inf")
    rows = []
    for phase, pairs in ((PHASE_INHALE, inhale_pairs), (PHASE_EXHALE, exhale_pairs)):
        name = SWAP[phase] if call.inverted else phase
        for start, end in pairs:
            start, end = int(start), int(end)
            if valid_only and not _valid(start, end, time_vec, cycle_sec):
                continue
            rows.append({"phase": name, "start": start, "end": end})

    rows.sort(key=lambda row: row["start"])
    return sorted(rows + _pause_between_breaths(rows), key=lambda row: row["start"])


def durations_sec(call: PhaseCall, rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean seconds per phase over `rows`, measured on production's own time vector.

    On `time_vec`, not on `end - start`: production's boundaries are sample indices into a time
    axis it built itself, and a sample count would only agree with it by coincidence.
    """
    time_vec = call.time_vec
    out = {}
    for phase in (PHASE_INHALE, PHASE_EXHALE):
        lengths = [float(time_vec[row["end"]]) - float(time_vec[row["start"]])
                   for row in rows if row["phase"] == phase]
        out[phase] = float(np.mean(lengths)) if lengths else float("nan")
    return out


def _pause_between_breaths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The stop, taken from the gap between an exhale ending and the next inhale starting.

    Production measures its own stop between a pair of boundary points bracketing the trough of
    the array *it* worked on, and which array that is depends on two independent sign decisions
    that land differently on different windows. Picking one by rule got it wrong: on one measured
    window the pause came out at 79% of the amplitude range - the top, not the bottom.

    So it is not picked. Exhale falls and inhale rises on the trace being labelled, which is what
    the swap above guarantees, so the stretch between an exhale ending and the next inhale
    starting **is** the trough whichever way production happened to hold the window. On both
    windows checked it reproduces production's own bracket exactly.

    The turn from inhale to exhale is left unclaimed, because production has no phase for it.
    """
    out = []
    for before, after in zip(rows, rows[1:]):
        if before["phase"] != PHASE_EXHALE or after["phase"] != PHASE_INHALE:
            continue
        start, end = int(before["end"]), int(after["start"])
        if end > start:
            out.append({"phase": PHASE_STOP, "start": start, "end": end})
    return out


def phases_for(values: np.ndarray, rate_bpm: float | None,
               fps: float) -> tuple[list[dict[str, Any]], str]:
    """The whole answer for one stored window: `(spans, note)`.

    A window with no stored rate has nothing for the detector to run at - it filters at roughly
    twice the rate - so it comes back with no spans and says so rather than being run at a guess.
    """
    values = np.asarray(values, dtype=float)
    if not rate_bpm:
        return [], "no stored rate, so the detector has nothing to run at"
    call = run(values, rate_bpm, fps)
    if call.rejected:
        return [], "the detector found no usable breaths in this window"
    rows = spans_of(call)
    note = (f"{len(rows)} spans; "
            + ("the algorithm read this window upside down and its answer was mapped back"
               if call.inverted else "the algorithm did not invert the trace"))
    return rows, note
