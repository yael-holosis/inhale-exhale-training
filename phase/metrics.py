"""What a run is judged on. Per-sample agreement is the loss's own view - not the answer.

Three levels, because they fail differently:

- **Per-sample** F1 per class. Catches nothing about segment structure: a model that emits the
  right classes in shredded pieces scores well here.
- **Event level**, IoU-matched. One predicted segment matches one reference segment of the same
  class when their overlap over union clears a threshold; unmatched reference segments are
  misses and unmatched predictions are false alarms. This is what says whether breaths came out
  as breaths.
- **Boundary error**, in samples, over the matched pairs. What a phase *duration* inherits.

Every number here is against the labels in the dataset, which are production's own answer. On
the algorithm-labelled set they measure imitation, not correctness - only the human-labelled
windows measure correctness.
"""

from __future__ import annotations

import numpy as np

from phase.labels import PHASES, UNKNOWN, targets_to_spans

LOA_Z = 1.96
"""Bland-Altman: the limits of agreement are the bias plus and minus this many SD."""

EPS_SEC = 1e-9
"""Floor under a labelled duration before dividing by it, for a relative error."""


def per_sample(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None,
               classes: tuple[str, ...] = PHASES) -> dict[str, float]:
    """Precision / recall / F1 per class plus macro F1 over **every** class.

    `unknown` counts. It is 29% of the samples and it is where the model's mistakes concentrate -
    calling a breath phase over the turn, or falling silent over a real one - so a score that
    leaves it out cannot see the failure it is most likely to make.

    Macro averages only the classes actually in play - present in the labels or predicted. A
    class that is neither is undefined, and averaging a zero for it punishes a window for the
    phases it does not contain.
    """
    pred, truth = np.asarray(pred).ravel(), np.asarray(truth).ravel()
    if mask is not None:
        keep = np.asarray(mask).ravel().astype(bool)
        pred, truth = pred[keep], truth[keep]

    out: dict[str, float] = {}
    scores = []
    for index, name in enumerate(classes):
        tp = float(np.sum((pred == index) & (truth == index)))
        fp = float(np.sum((pred == index) & (truth != index)))
        fn = float(np.sum((pred != index) & (truth == index)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"precision_{name}"] = precision
        out[f"recall_{name}"] = recall
        out[f"f1_{name}"] = f1
        # A class that neither occurs nor is predicted is undefined here, not zero. Scoring it
        # zero put an all-unknown window the model called correctly at 1/4 - it has one class
        # right and three that were never in play.
        if tp + fp + fn > 0:
            scores.append(f1)
    out["macro_f1"] = float(np.mean(scores)) if scores else 1.0
    out["accuracy"] = float(np.mean(pred == truth)) if pred.size else 0.0
    return out


def _segments(labels: np.ndarray, drop_unknown: bool = True) -> list[dict]:
    spans = targets_to_spans(np.asarray(labels))
    return [span for span in spans if not (drop_unknown and span["phase"] == PHASES[UNKNOWN])]


def event_level(pred: np.ndarray, truth: np.ndarray, iou_threshold: float = 0.5,
                classes: tuple[str, ...] = PHASES) -> dict[str, float]:
    """Greedy best-IoU matching within each class, `unknown` included.

    Boundary error stays over the called phases: adjacent spans share an edge, so an `unknown`
    boundary is the same edge as the phase next to it.
    """
    out: dict[str, float] = {}
    starts, ends = [], []
    reference_spans = _segments(truth, drop_unknown=False)
    proposed_spans = _segments(pred, drop_unknown=False)
    for name in classes:
        reference = [s for s in reference_spans if s["phase"] == name]
        proposed = [s for s in proposed_spans if s["phase"] == name]
        matched, start_errors, end_errors = _match(reference, proposed, iou_threshold)
        precision = matched / len(proposed) if proposed else 0.0
        recall = matched / len(reference) if reference else 0.0
        out[f"event_f1_{name}"] = (2 * precision * recall / (precision + recall)
                                   if precision + recall else 0.0)
        out[f"event_recall_{name}"] = recall
        out[f"event_precision_{name}"] = precision
        out[f"n_true_{name}"] = float(len(reference))
        if name != PHASES[UNKNOWN]:
            # Boundary error over the called phases only. Adjacent spans share a boundary, so
            # an `unknown` edge is the same physical edge as the phase beside it - counting both
            # would weigh one boundary twice.
            starts.extend(start_errors)
            ends.extend(end_errors)
    out["boundary_mae_samples"] = float(np.mean(np.abs(starts + ends))) if starts else float("nan")
    out["start_mae_samples"] = float(np.mean(np.abs(starts))) if starts else float("nan")
    out["end_mae_samples"] = float(np.mean(np.abs(ends))) if ends else float("nan")
    return out


def match_spans(reference: list[dict], proposed: list[dict],
                threshold: float) -> list[tuple[dict, dict, float]]:
    """Greedy best-IoU pairing inside one class: `(labelled, called, iou)` per match.

    One proposal answers at most one reference span, so a prediction covering two labelled
    breaths is one match and one miss rather than two matches.
    """
    taken, pairs = set(), []
    for target in reference:
        best, best_iou = None, 0.0
        for position, candidate in enumerate(proposed):
            if position in taken:
                continue
            overlap = min(target["end"], candidate["end"]) - max(target["start"],
                                                                 candidate["start"])
            if overlap <= 0:
                continue
            union = (max(target["end"], candidate["end"])
                     - min(target["start"], candidate["start"]))
            iou = overlap / union
            if iou > best_iou:
                best, best_iou = position, iou
        if best is not None and best_iou >= threshold:
            taken.add(best)
            pairs.append((target, proposed[best], best_iou))
    return pairs


def _match(reference: list[dict], proposed: list[dict], threshold: float):
    pairs = match_spans(reference, proposed, threshold)
    return (len(pairs),
            [called["start"] - labelled["start"] for labelled, called, _ in pairs],
            [called["end"] - labelled["end"] for labelled, called, _ in pairs])


def phase_durations(labels: np.ndarray, fps: float,
                    classes: tuple[str, ...] = PHASES) -> dict[str, float]:
    """Mean seconds per phase and the I:E ratio, for comparing against the device's own numbers."""
    out = {}
    for name in classes:
        if name == PHASES[UNKNOWN]:
            continue
        lengths = [s["end"] - s["start"] for s in _segments(labels) if s["phase"] == name]
        out[f"mean_{name}_sec"] = float(np.mean(lengths) / fps) if lengths else float("nan")
    inhale, exhale = out.get("mean_inhale_sec"), out.get("mean_exhale_sec")
    out["ie_ratio"] = float(exhale / inhale) if inhale and inhale > 0 else float("nan")
    return out


def duration_pairs(prediction: np.ndarray, truth: np.ndarray, fps: float,
                   classes: tuple[str, ...] = PHASES) -> dict[str, tuple]:
    """`{phase: (predicted_sec, labelled_sec)}` - the mean span length each side calls.

    The quantity the device actually reports, so it is the one to measure agreement on. `nan`
    where a side called no span of that phase at all: that is a coverage fact, not a zero, and
    averaging a zero into it would invent agreement where there was no measurement.
    """
    said, meant = phase_durations(prediction, fps, classes), phase_durations(truth, fps, classes)
    return {name: (said[f"mean_{name}_sec"], meant[f"mean_{name}_sec"])
            for name in classes if name != PHASES[UNKNOWN]}


def duration_agreement(pairs: list[dict[str, tuple]],
                       classes: tuple[str, ...] = PHASES) -> dict[str, dict[str, float]]:
    """Bias, limits of agreement and MAE per phase, over the windows where both sides called it.

    Bland-Altman: bias is the mean signed error and says whether the model systematically over-
    or under-calls the phase; the limits are bias +- 1.96 SD and say how far a single window can
    be. They answer different questions and a mean absolute error hides the first.
    """
    out: dict[str, dict[str, float]] = {}
    for name in (n for n in classes if n != PHASES[UNKNOWN]):
        both = [(said, meant) for window in pairs
                for said, meant in [window[name]]
                if not (np.isnan(said) or np.isnan(meant))]
        if not both:
            out[name] = {"n": 0}
            continue
        said = np.array([a for a, _ in both])
        meant = np.array([b for _, b in both])
        error = said - meant
        bias, spread = float(np.mean(error)), float(np.std(error, ddof=1)) if len(error) > 1 \
            else 0.0
        out[name] = {
            "n": len(both), "coverage": len(both) / max(len(pairs), 1),
            "bias_sec": bias, "mae_sec": float(np.mean(np.abs(error))),
            "relative_mae": float(np.mean(np.abs(error) / np.maximum(meant, EPS_SEC))),
            "loa_low": bias - LOA_Z * spread, "loa_high": bias + LOA_Z * spread,
            "labelled_mean_sec": float(np.mean(meant)),
        }
    return out
