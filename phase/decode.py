"""Turn per-sample logits into segments that a breath could actually have made.

Argmax is free to emit inhale, exhale, inhale over three samples. Nothing in a lung does that,
and a single stray sample splits one breath into three in every event-level metric. Two cheap
passes fix it, both on the device's budget:

1. **Viterbi** over the four classes with a transition matrix - impossible transitions cost
   infinity, staying costs nothing, an allowed change costs `switch_penalty`.
2. **Minimum duration** - a surviving run shorter than its class's floor is absorbed into
   whichever neighbour is longer.

The allowed transitions are configured, not written in here, because they are a claim about the
label set rather than about physiology: production emits no phase for the turn from inhale to
exhale, so `inhale -> unknown -> exhale` is the normal path in this dataset and
`inhale -> exhale` is not.
"""

from __future__ import annotations

import numpy as np

from phase.labels import N_CLASSES, PHASE_IDS


def transition_matrix(allowed: dict[str, list[str]], switch_penalty: float) -> np.ndarray:
    """Log-cost matrix from `{phase: [phases it may become]}`. Self-transitions are always free."""
    cost = np.full((N_CLASSES, N_CLASSES), -np.inf, dtype=np.float64)
    for name, index in PHASE_IDS.items():
        cost[index, index] = 0.0
        for target in allowed.get(name, []):
            cost[index, PHASE_IDS[target]] = -abs(switch_penalty)
    return cost


def viterbi(log_probs: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Best path through (length, n_classes) log-probabilities under `cost`."""
    length, classes = log_probs.shape
    score = log_probs[0].copy()
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


def enforce_min_duration(path: np.ndarray, minimum: dict[str, int]) -> np.ndarray:
    """Absorb runs shorter than their class's floor into the longer neighbour.

    Repeated until nothing moves: absorbing one short run can leave its neighbour short too.
    """
    path = np.asarray(path, dtype=np.int64).copy()
    floors = np.array([minimum.get(name, 1) for name in PHASE_IDS], dtype=np.int64)
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


def decode(logits: np.ndarray, cost: np.ndarray | None = None,
           minimum: dict[str, int] | None = None) -> np.ndarray:
    """(length, n_classes) logits -> per-sample classes. Both passes are optional."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    path = viterbi(log_probs, cost) if cost is not None else np.argmax(log_probs, axis=1)
    return enforce_min_duration(path, minimum) if minimum else path
