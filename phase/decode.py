"""Turn per-sample logits into segments that a breath could actually have made.

Argmax is free to emit inhale, exhale, inhale over three samples. Nothing in a lung does that,
and a single stray sample splits one breath into three in every event-level metric. Two cheap
passes fix it, both on the device's budget:

1. **Viterbi** over the four classes with a transition matrix - impossible transitions cost
   infinity, staying costs nothing, an allowed change costs `switch_penalty`.
2. **Minimum duration** - a surviving run shorter than its class's floor is absorbed into
   whichever neighbour is longer. **Off by default**: measured after Viterbi it changes nothing,
   and a floor deletes a real short phase rather than smoothing it. See `training.decoding`.

The allowed transitions are configured, not written in here, because they are a claim about the
label set rather than about physiology: production emits no phase for the turn from inhale to
exhale, so `inhale -> unknown -> exhale` is the normal path in this dataset and
`inhale -> exhale` is not.
"""

from __future__ import annotations

import numpy as np

from phase.labels import PHASES, UNKNOWN, classes_of, targets_to_spans


def transition_matrix(allowed: dict[str, list[str]], switch_penalty: float,
                      classes: tuple[str, ...] = PHASES) -> np.ndarray:
    """Log-cost matrix from `{phase: [phases it may become]}`. Self-transitions are free.

    A class that is not a key is not in the label set: it can neither be entered nor held.
    """
    ids = {name: index for index, name in enumerate(classes)}
    cost = np.full((len(classes), len(classes)), -np.inf, dtype=np.float64)
    for name, index in ids.items():
        if name not in allowed:
            continue
        cost[index, index] = 0.0
        for target in allowed.get(name, []):
            if target not in ids:
                raise KeyError(f"transition {name} -> {target}: {target!r} is not in {classes}")
            cost[index, ids[target]] = -abs(switch_penalty)
    return cost


def viterbi(log_probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Best path through (length, n_classes) log-probabilities under `cost`."""
    length, classes = log_probs.shape
    # A class that cannot be held is outside the label set, so a path may not start in it either.
    score = np.where(np.isfinite(np.diag(cost)), log_probs[0], -np.inf)
    back = np.zeros((length, classes), dtype=np.int64)
    for step in range(1, length):
        candidates = score[:, None] + cost                 # (from, to)
        back[step] = np.argmax(candidates, axis=0)
        score = candidates[back[step], np.arange(classes)] + log_probs[step]
    path = np.zeros(length, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for step in range(length - 1, 0, -1):
        path[step - 1] = back[step, path[step]]
    return path


def enforce_min_duration(path: np.ndarray, minimum: dict[str, int],
                         classes: tuple[str, ...] = PHASES) -> np.ndarray:
    """Absorb runs shorter than their class's floor into the longer neighbour.

    Repeated until nothing moves: absorbing one short run can leave its neighbour short too.
    """
    path = np.asarray(path, dtype=np.int64).copy()
    floors = np.array([minimum.get(name, 1) for name in classes], dtype=np.int64)
    for _ in range(len(floors) * 4):
        edges = np.flatnonzero(np.diff(path)) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [path.size]))
        lengths = ends - starts
        short = np.flatnonzero(lengths < floors[path[starts]])
        if short.size == 0:
            break
        run = int(short[np.argmin(lengths[short])])
        before = lengths[run - 1] if run > 0 else -1
        after = lengths[run + 1] if run + 1 < len(starts) else -1
        if before < 0 and after < 0:
            break
        path[starts[run]:ends[run]] = path[starts[run - 1]] if before >= after \
            else path[starts[run + 1]]
    return path


def fill_beside_unknown(path: np.ndarray) -> np.ndarray:
    """A span beside `unknown` becomes `unknown` too - its end is where the model gave up.

    One pass over the input, so it never cascades: a span two away from `unknown` survives.
    """
    path = np.asarray(path, dtype=np.int64)
    out = path.copy()
    spans = targets_to_spans(path)
    for index, span in enumerate(spans):
        if span["phase"] == PHASES[UNKNOWN]:
            continue
        beside = (spans[i]["phase"] for i in (index - 1, index + 1) if 0 <= i < len(spans))
        if PHASES[UNKNOWN] in beside:
            out[span["start"]:span["end"]] = UNKNOWN
    return out


def reports(path: np.ndarray, max_unknown_fraction: float | None) -> bool:
    """Whether a window is reported at all: not when the model abstains on most of it."""
    if max_unknown_fraction is None or not len(path):
        return True
    return float(np.mean(np.asarray(path) == UNKNOWN)) <= max_unknown_fraction


def decode(logits: np.ndarray, cost: np.ndarray | None = None,
           minimum: dict[str, int] | None = None) -> np.ndarray:
    """(length, n_classes) logits -> per-sample classes. Both passes are optional."""
    logits = np.asarray(logits, dtype=np.float64)
    if cost is not None and cost.shape[0] != logits.shape[1]:
        raise ValueError(f"{logits.shape[1]} logits against a {cost.shape[0]}-class transition "
                         "table")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    path = viterbi(log_probs, cost) if cost is not None else np.argmax(log_probs, axis=1)
    return (enforce_min_duration(path, minimum, classes_of(logits.shape[1])) if minimum
            else path)
