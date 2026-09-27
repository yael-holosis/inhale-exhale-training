"""Corrections applied to a per-sample target before it is written to a shard.

These take the labelling at less than its word. Both are **label** changes, not filters: the
window stays in the set and the corrected target is what the network is trained and scored
against, so every figure that reads a shard or a saved logit file already shows what the net saw.
Nothing here reads the trace - a correction is a statement about the labelling only.

`blank_edge_spans` - a labelled span that runs to the window boundary becomes `unknown`. The
    boundary cut that breath and whether a labeller marks the fragment it leaves is inconsistent.
    Only a span actually touching the edge: where the window already opens or closes with
    `unknown`, the labeller has said what they saw and nothing needs correcting.

`merge_stop_into_exhale` - a pause that directly follows an exhale becomes `exhale`, so it is part of
    the breath out rather than a phase of its own. A pause with no exhale before it - after
    `unknown`, after an inhale, or at the window start - becomes `unknown`: there is no exhale for it
    to continue. Applied first: the other corrections read the merged label set. The decoder then
    needs the `stop_as_exhale` transition table.

`all_unknown_above` - a window whose corrected target is more than this fraction `unknown` becomes
    entirely `unknown`. The fraction is measured **after** the edge spans are blanked, because
    that is the target the network would otherwise be given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from phase.labels import EXHALE, PHASES, STOP, UNKNOWN, targets_to_spans, touches_edge

BLANK_EDGE_SPANS = "blank_edge_spans"
ALL_UNKNOWN_ABOVE = "all_unknown_above"
MERGE_STOP_INTO_EXHALE = "merge_stop_into_exhale"


@dataclass(frozen=True)
class Corrections:
    """What `data.labels.corrections` asks for. All off by default."""

    blank_edge_spans: bool = False
    all_unknown_above: float | None = None
    merge_stop_into_exhale: bool = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "Corrections":
        cfg = dict(cfg or {})
        above = cfg.get(ALL_UNKNOWN_ABOVE)
        above = None if above is None else float(above)
        if above is not None and not 0.0 <= above < 1.0:
            raise ValueError(
                f"labels.corrections.{ALL_UNKNOWN_ABOVE} must be in [0, 1) - 1.0 would blank "
                "every window")
        return cls(blank_edge_spans=bool(cfg.get(BLANK_EDGE_SPANS, False)),
                   all_unknown_above=above,
                   merge_stop_into_exhale=bool(cfg.get(MERGE_STOP_INTO_EXHALE, False)))

    @property
    def enabled(self) -> bool:
        return (self.blank_edge_spans or self.all_unknown_above is not None
                or self.merge_stop_into_exhale)

    def describe(self) -> dict[str, Any]:
        return {BLANK_EDGE_SPANS: self.blank_edge_spans,
                ALL_UNKNOWN_ABOVE: self.all_unknown_above,
                MERGE_STOP_INTO_EXHALE: self.merge_stop_into_exhale}

    def apply(self, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """`(corrected target, forced entirely unknown)`. The input is never modified."""
        target = np.asarray(target, dtype=np.int64).copy()
        if not target.size:
            return target, False
        if self.merge_stop_into_exhale:
            target = merge_stops(target)
        if self.blank_edge_spans:
            target = blank_edges(target)
        if self.all_unknown_above is not None:
            if float((target == UNKNOWN).mean()) > self.all_unknown_above:
                return np.full_like(target, UNKNOWN), True
        return target, False


def merge_stops(target: np.ndarray) -> np.ndarray:
    """Each pause joins the exhale before it, or becomes `unknown` where there is none."""
    out = np.asarray(target, dtype=np.int64).copy()
    for span in targets_to_spans(out):
        if span["phase"] == PHASES[STOP]:
            follows_exhale = span["start"] > 0 and out[span["start"] - 1] == EXHALE
            out[span["start"]:span["end"]] = EXHALE if follows_exhale else UNKNOWN
    return out


def blank_edges(target: np.ndarray) -> np.ndarray:
    """A labelled span that reaches the window boundary rewritten to `unknown`, whole.

    Touching the boundary is the whole test. A window opening with `unknown` has no truncated
    breath at that end - the labeller looked at it and declined it - so its first phase is left
    exactly as drawn. 78% of windows start with a phase at sample 0 and 77% end with one.
    """
    out = np.asarray(target, dtype=np.int64).copy()
    spans = [span for span in targets_to_spans(out) if span["phase"] != "unknown"]
    if not spans:
        return out
    for span in (spans[0], spans[-1]):
        if touches_edge(span, out.size):
            out[span["start"]:span["end"]] = UNKNOWN
    return out
