"""Corrections applied to a per-sample target before it is written to a shard.

These take the labelling at less than its word. Both are **label** changes, not filters: the
window stays in the set and the corrected target is what the network is trained and scored
against, so every figure that reads a shard or a saved logit file already shows what the net saw.
Nothing here reads the trace - a correction is a statement about the labelling only.

`blank_edge_spans` - a labelled span that runs to the window boundary becomes `unknown`. The
    boundary cut that breath and whether a labeller marks the fragment it leaves is inconsistent.
    Only a span actually touching the edge: where the window already opens or closes with
    `unknown`, the labeller has said what they saw and nothing needs correcting.

`all_unknown_above` - a window whose corrected target is more than this fraction `unknown` becomes
    entirely `unknown`. The fraction is measured **after** the edge spans are blanked, because
    that is the target the network would otherwise be given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from phase.labels import UNKNOWN, targets_to_spans, touches_edge

BLANK_EDGE_SPANS = "blank_edge_spans"
ALL_UNKNOWN_ABOVE = "all_unknown_above"


@dataclass(frozen=True)
class Corrections:
    """What `data.labels.corrections` asks for. Both off by default."""

    blank_edge_spans: bool = False
    all_unknown_above: float | None = None

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
                   all_unknown_above=above)

    @property
    def enabled(self) -> bool:
        return self.blank_edge_spans or self.all_unknown_above is not None

    def describe(self) -> dict[str, Any]:
        return {BLANK_EDGE_SPANS: self.blank_edge_spans,
                ALL_UNKNOWN_ABOVE: self.all_unknown_above}

    def apply(self, target: np.ndarray) -> tuple[np.ndarray, bool]:
        """`(corrected target, forced entirely unknown)`. The input is never modified."""
        target = np.asarray(target, dtype=np.int64).copy()
        if not target.size:
            return target, False
        if self.blank_edge_spans:
            target = blank_edges(target)
        if self.all_unknown_above is not None:
            if float((target == UNKNOWN).mean()) > self.all_unknown_above:
                return np.full_like(target, UNKNOWN), True
        return target, False


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
