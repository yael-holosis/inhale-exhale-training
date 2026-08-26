"""Where a window's per-sample target comes from. Two sources, one shape.

Both return `{phase, start, end}` rows with an **exclusive** end, which `phase.labels` turns into
a per-sample target. Everything downstream is identical; only this file knows the difference.

`ALGORITHM` - production's own `calculate_inhale_exhale_time`, re-run on the stored samples.
    Available for every window, so it is the only source there is enough of to train on today.
    It is distillation: the ceiling is the current algorithm, its boundaries are the 10% and 90%
    amplitude crossings rather than phase durations, and it emits no phase for the turn from
    inhale to exhale - so `unknown` sits structurally at the crest of most breaths.

`data.labels.corrections` then takes either source at less than its word - blanking a span the
window boundary cut, blanking a window that is mostly unknown. See `phase.corrections`.

`HUMAN` - `BreathPhaseTimeRecord`, the spans a person drew. The real target and the only thing
    that measures correctness, but only a few dozen windows carry one. A window nobody has
    labelled has no target at all and is skipped, not filled with `unknown` - "nobody looked at
    this" and "somebody looked and could not call it" are different facts, and the labelling
    rule already writes the second one down explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from phase import production, sources
from phase.corrections import Corrections

ALGORITHM = "algorithm"
HUMAN = "human"
SOURCES = (ALGORITHM, HUMAN)

LATEST = "latest"
LOWEST_ID = "lowest_id"
CONFLICT_RULES = (LATEST, LOWEST_ID)


class LabelSource:
    """Resolves one window to `{phase, start, end}` rows, per `data.labels`."""

    def __init__(self, env_key: str, cfg: dict[str, Any]):
        self.env_key = env_key
        self.source = str(cfg.get("source", ALGORITHM))
        if self.source not in SOURCES:
            raise ValueError(f"labels.source must be one of {SOURCES}, got {self.source!r}")
        self.labeler_id = cfg.get("labeler_id")
        self.exclude_labeler_ids = [int(v) for v in (cfg.get("exclude_labeler_ids") or [])]
        self.on_conflict = str(cfg.get("on_conflict", LATEST))
        if self.on_conflict not in CONFLICT_RULES:
            raise ValueError(f"labels.on_conflict must be one of {CONFLICT_RULES}")
        self.requires_rate = bool(cfg.get("requires_rate", True))
        self.orient_by_reviewer_flip = bool(cfg.get("orient_by_reviewer_flip", False))
        # Applied to the target, not to the spans: a correction is about what the net is given.
        self.corrections = Corrections.from_config(cfg.get("corrections"))
        self._vocabulary: dict[int, str] | None = None

    # ------------------------------------------------------------------ description

    def describe(self) -> dict[str, Any]:
        """What went into `build_params.yaml`, so a dataset says how it was labelled."""
        out = {"source": self.source,
               "orient_by_reviewer_flip": self.orient_by_reviewer_flip,
               "corrections": self.corrections.describe()}
        if self.source == ALGORITHM:
            out["requires_rate"] = self.requires_rate
            out["holosissystem"] = production.version()
        else:
            out["labeler_id"] = self.labeler_id
            out["exclude_labeler_ids"] = self.exclude_labeler_ids
            out["on_conflict"] = self.on_conflict
        return out

    def eligible(self, catalogue: pd.DataFrame) -> pd.DataFrame:
        """Windows this source can label at all.

        For `human` that is the windows carrying spans - a few dozen, against several thousand -
        so the selection has to happen before anything is downloaded.
        """
        if self.source == ALGORITHM:
            return catalogue
        return catalogue[catalogue["Spans"] > 0]

    # ----------------------------------------------------------------- orientation

    def orient(self, row: pd.Series, values: np.ndarray) -> np.ndarray:
        """The trace the labels describe: negated where the reviewer set `ReviewerFlipped`.

        Amplitude only, never a reversal in time - the human spans are index-based.
        """
        if not self.orient_by_reviewer_flip:
            return values
        return -values if bool(row.get(sources.REVIEWER_FLIPPED, False)) else values

    # ------------------------------------------------------------------------ rows

    def rows_for(self, row: pd.Series, values: np.ndarray) -> tuple[list[dict[str, Any]], str]:
        """`(spans, note)` for one window. The note is kept in the manifest, not thrown away."""
        if self.source == ALGORITHM:
            rate = None if pd.isna(row.get("RespirationRate")) else float(row["RespirationRate"])
            if self.requires_rate and not rate:
                return [], "no stored rate, so the detector has nothing to run at"
            return production.phases_for(values, rate, float(row["AnalysisFps"]))
        return self._human_rows(row)

    def _human_rows(self, row: pd.Series) -> tuple[list[dict[str, Any]], str]:
        spans = sources.human_spans(self.env_key, int(row["ID"]),
                                    self.exclude_labeler_ids)
        if spans.empty:
            return [], ("nobody has labelled this window" if not self.exclude_labeler_ids
                        else "no labelling left once the excluded labellers are dropped")
        if self.labeler_id is not None:
            spans = spans[spans["LabelerID"] == int(self.labeler_id)]
            if spans.empty:
                return [], f"labeller {self.labeler_id} has not labelled this window"

        labellers = spans["LabelerID"].unique()
        note = f"{len(spans)} spans by labeller {int(labellers[0])}"
        if len(labellers) > 1:
            chosen = self._resolve(spans)
            note = (f"{len(labellers)} labellers ({', '.join(str(int(v)) for v in labellers)}); "
                    f"took {int(chosen)} by {self.on_conflict}")
            spans = spans[spans["LabelerID"] == chosen]

        if self._vocabulary is None:
            self._vocabulary = sources.phase_vocabulary(self.env_key)
        # `EndIndex` on a span is **inclusive** - adjacent spans share a boundary sample - while
        # a window's is exclusive. Off by one here shifts every human boundary by a sample.
        rows = [{"phase": self._vocabulary[int(span.BreathPhaseTypeID)],
                 "start": int(span.StartIndex), "end": int(span.EndIndex) + 1}
                for span in spans.itertuples()]
        return sorted(rows, key=lambda item: item["start"]), note

    def _resolve(self, spans: pd.DataFrame) -> int:
        """Which labeller wins where two have read the same window.

        Never a merge. Two readings of one breath are two opinions, and averaging them invents a
        third that neither person would endorse.
        """
        if self.on_conflict == LOWEST_ID:
            return int(spans["LabelerID"].min())
        newest = spans.sort_values("RecordCreationTime").iloc[-1]
        return int(newest["LabelerID"])
