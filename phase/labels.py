"""The label vocabulary, and turning spans into a per-sample target.

One class per sample, four classes. `UNKNOWN` is class 0 deliberately: it is what an
unclaimed sample is, so a target array starts filled with it and every span written over it
is a positive statement.

**What `unknown` means in an algorithm-labelled window is not what it means in a human one.**
A labeller marks it where they could not call the trace. Production leaves it in two places:
the same uncallable stretches, *and* the turn from inhale to exhale at every crest - it has no
phase for that turn, so `suggestion.suggest` emits no span there. So in this dataset `unknown`
sits structurally at the top of most breaths. Anything that reads class balance, or treats
`unknown` as "no breathing", has to know that.
"""

from __future__ import annotations

import numpy as np

UNKNOWN = 0
INHALE = 1
EXHALE = 2
STOP = 3

PHASES = ("unknown", "inhale", "exhale", "stop")
"""Index -> name. The names are the labelling app's own (`utils/waveforms.py` in the labelling repo)."""

PHASE_IDS = {name: index for index, name in enumerate(PHASES)}
N_CLASSES = len(PHASES)

SWAP_ON_FLIP = np.array([UNKNOWN, EXHALE, INHALE, STOP], dtype=np.int64)
"""Turning a trace over exchanges inhale and exhale. Stop sits at a trough either way - it is
defined as the stretch between an exhale ending and the next inhale starting, which is the
trough of whichever picture you are holding, so it does not move."""


def spans_to_targets(spans, n_samples: int) -> np.ndarray:
    """Per-sample classes from `{phase, start, end}` rows. `end` exclusive, gaps stay unknown.

    Rows outside the window are clipped rather than dropped - a span the detector put one sample
    past the end is a rounding artefact, not a reason to lose the breath.
    """
    target = np.full(int(n_samples), UNKNOWN, dtype=np.int64)
    for span in spans:
        phase = str(span["phase"]).lower()
        if phase not in PHASE_IDS:
            raise KeyError(f"unknown phase {span['phase']!r}; vocabulary is {PHASES}")
        start = max(0, int(span["start"]))
        end = min(int(n_samples), int(span["end"]))
        if end > start:
            target[start:end] = PHASE_IDS[phase]
    return target


def targets_to_spans(target: np.ndarray) -> list[dict]:
    """The inverse, for reporting and for the boundary metrics. Runs of equal class."""
    target = np.asarray(target, dtype=np.int64)
    if target.size == 0:
        return []
    edges = np.flatnonzero(np.diff(target)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [target.size]))
    return [{"phase": PHASES[int(target[start])], "start": int(start), "end": int(end)}
            for start, end in zip(starts, ends)]


def class_counts(target: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(target).ravel(), minlength=N_CLASSES)
    return {name: int(counts[index]) for index, name in enumerate(PHASES)}
