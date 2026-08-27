"""How long each phase lasts, and whether the network agrees with the labeller about it.

The network exists to measure the time spent in each phase, so the duration **is** the result.
Per-sample F1 is a proxy for it and a poor one: a boundary two samples late costs almost nothing
in F1 and costs 0.2 s of inhale time, and a phase split into three scores badly while its total
time is right. Two views, because they answer different questions:

- **Per span** (`span_errors`). Every labelled span matched to the prediction of the same phase,
  one signed error per pair. This is the error on a single breath, and it is what a reader of a
  breath-by-breath display sees.
- **Per signal** (`signal_medians`). The median span length each side calls over all the windows
  of one recording, compared side by side. This is the number a session report would carry, and
  it is not the mean of the first: a per-breath error that cancels disappears here, and a
  systematic one does not.

**A span the window boundary cut is dropped from both**, on both sides. Its stored length is the
length of the fragment left inside the window, not of the phase - comparing that against a whole
span measures the cut. Dropping it on one side only would be the same mistake with a sign.

`unknown` is not measured. Production emits no phase for the turn from inhale to exhale, so the
class sits structurally at the crest of most breaths - a "duration" for it describes the label
set, not time anybody spends.

Every number here is against the labels in the dataset. On an algorithm-labelled set that is
production's own answer, so the agreement is imitation; only the human-labelled set makes it
correctness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phase.labels import (EXHALE, INHALE, PHASES, UNKNOWN, targets_to_spans,
                          touches_edge)
from phase.metrics import EPS_SEC, LOA_Z, match_spans

MEASURED = tuple(name for name in PHASES if name != PHASES[UNKNOWN])
"""The phases a duration means something for."""

RATIO = "exhale/inhale"
"""The I:E ratio, reported beside the phases because it is what a clinician reads and because it
is the one quantity a pair of compensating duration errors cannot hide in."""

COLUMNS = (*MEASURED, RATIO)
UNIT = {**{name: "s" for name in MEASURED}, RATIO: ""}
"""A ratio is dimensionless; formatting it with a `s` suffix would be a lie."""

LABEL, MODEL = "labelled", "model"
ERROR = "error_sec"
MEDIAN = "median"
SPANS = "spans"
USABLE = "usable"
SIGNAL_KEYS = ("env", "signal", "patient")
"""Keyed on `(env, signal)` because the two database instances have separate id spaces - the
signal id alone collides across them, exactly as the window id does."""

MIN_SPANS_PER_SIGNAL = 3
"""A median over one or two breaths is noise. Signals under this on either side are marked
unusable rather than dropped silently."""


def interior_spans(labels: np.ndarray, phase: str | None = None) -> list[dict]:
    """The called spans of one window that the window boundary did not cut."""
    labels = np.asarray(labels)
    spans = [span for span in targets_to_spans(labels)
             if span["phase"] != PHASES[UNKNOWN] and not touches_edge(span, labels.size)]
    return spans if phase is None else [span for span in spans if span["phase"] == phase]


def _identity(item: dict) -> dict:
    return {"env": str(item["env"]), "signal": int(item["signal"]),
            "patient": str(item["patient"]), "window_id": int(item["window_id"])}


def _seconds(span: dict, fps: float) -> float:
    return (span["end"] - span["start"]) / fps


def span_errors(items, iou_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(pairs, coverage)` - one pair row per matched span, one coverage row per window/phase.

    Matching is the same greedy best-IoU rule the event metrics use, so a pair here is a pair
    there. An unmatched span carries no duration error at all - a miss and a wrong length are
    different failures and averaging them together hides both - so it appears only in
    `coverage`.
    """
    pairs, coverage = [], []
    for item in items:
        fps = float(item["fps"])
        partner = {}
        for name in MEASURED:
            labelled = interior_spans(item[LABEL], name)
            called = interior_spans(item[MODEL], name)
            matched = match_spans(labelled, called, iou_threshold)
            for truth_span, pred_span, iou in matched:
                labelled_sec, model_sec = _seconds(truth_span, fps), _seconds(pred_span, fps)
                partner[_span_key(truth_span)] = (labelled_sec, model_sec)
                pairs.append({**_identity(item), "phase": name,
                              f"{LABEL}_sec": labelled_sec, f"{MODEL}_sec": model_sec,
                              ERROR: model_sec - labelled_sec, "iou": iou,
                              f"{LABEL}_start": truth_span["start"],
                              f"{MODEL}_start": pred_span["start"]})
            coverage.append({**_identity(item), "phase": name, LABEL: len(labelled),
                             MODEL: len(called), "matched": len(matched)})
        breaths, ratios = _breath_ratios(item, partner)
        pairs.extend(ratios)
        coverage.append({**_identity(item), "phase": RATIO, LABEL: breaths, MODEL: breaths,
                         "matched": len(ratios)})
    return pd.DataFrame(pairs), pd.DataFrame(coverage)


def _span_key(span: dict) -> tuple:
    return span["phase"], span["start"], span["end"]


def _breath_ratios(item: dict, partner: dict) -> tuple[int, list[dict]]:
    """`(labelled breaths, ratio rows)` - exhale over inhale, per breath.

    A breath is an inhale followed by the next called phase being an exhale, which is what the
    ratio means. Both of its spans must have matched a prediction: a ratio built from one matched
    span and one guess is not a reading of the same breath. Boundary-cut spans are already gone,
    so the called sequence here is contiguous.
    """
    order = interior_spans(item[LABEL])
    rows, breaths = [], 0
    for first, second in zip(order, order[1:]):
        if (first["phase"], second["phase"]) != (PHASES[INHALE], PHASES[EXHALE]):
            continue
        breaths += 1
        inhale, exhale = partner.get(_span_key(first)), partner.get(_span_key(second))
        if inhale is None or exhale is None:
            continue
        labelled_ratio = exhale[0] / inhale[0]
        model_ratio = exhale[1] / inhale[1]
        rows.append({**_identity(item), "phase": RATIO,
                     f"{LABEL}_sec": labelled_ratio, f"{MODEL}_sec": model_ratio,
                     ERROR: model_ratio - labelled_ratio, "iou": float("nan"),
                     f"{LABEL}_start": first["start"], f"{MODEL}_start": first["start"]})
    return breaths, rows


def signal_medians(items, min_spans: int = MIN_SPANS_PER_SIGNAL) -> pd.DataFrame:
    """The median span length each side calls, per recording and phase, and their difference.

    Deliberately unmatched. The question is whether the two sides would report the same time in
    the phase for this recording, which does not require them to agree breath by breath.
    """
    rows = []
    for item in items:
        fps = float(item["fps"])
        for side in (LABEL, MODEL):
            for span in interior_spans(item[side]):
                rows.append({**{key: _identity(item)[key] for key in SIGNAL_KEYS},
                             "phase": span["phase"], "side": side,
                             "duration_sec": _seconds(span, fps)})
    if not rows:
        return pd.DataFrame()

    keys = [*SIGNAL_KEYS, "phase"]
    grouped = (pd.DataFrame(rows).groupby([*keys, "side"])["duration_sec"]
               .agg(**{MEDIAN: "median", SPANS: "size"}))
    wide = grouped.unstack("side")
    # A signal that called the phase on one side only leaves the other column absent, not zero.
    wide = wide.reindex(columns=pd.MultiIndex.from_product([[MEDIAN, SPANS], [LABEL, MODEL]]))
    wide.columns = [f"{stat}_{side}" for stat, side in wide.columns]
    wide = wide.reset_index()
    wide[f"{SPANS}_{LABEL}"] = wide[f"{SPANS}_{LABEL}"].fillna(0).astype(int)
    wide[f"{SPANS}_{MODEL}"] = wide[f"{SPANS}_{MODEL}"].fillna(0).astype(int)
    wide[ERROR] = wide[f"{MEDIAN}_{MODEL}"] - wide[f"{MEDIAN}_{LABEL}"]
    wide[USABLE] = ((wide[f"{SPANS}_{LABEL}"] >= min_spans)
                    & (wide[f"{SPANS}_{MODEL}"] >= min_spans))
    wide = pd.concat([wide, _signal_ratios(wide)], ignore_index=True)
    return wide.sort_values([*SIGNAL_KEYS, "phase"]).reset_index(drop=True)


def _signal_ratios(wide: pd.DataFrame) -> pd.DataFrame:
    """One `exhale/inhale` row per signal, from that signal's two medians.

    From the medians rather than from per-breath ratios: this is the recording-level number, and
    the median of the ratios is not the ratio of the medians. Usable only where both phases were,
    so a ratio never rests on a median the row above already called too thin.
    """
    keyed = wide.set_index([*SIGNAL_KEYS, "phase"])
    rows = []
    for key in wide[list(SIGNAL_KEYS)].drop_duplicates().itertuples(index=False):
        signal = tuple(key)
        try:
            inhale = keyed.loc[(*signal, PHASES[INHALE])]
            exhale = keyed.loc[(*signal, PHASES[EXHALE])]
        except KeyError:
            continue
        row = {name: value for name, value in zip(SIGNAL_KEYS, signal)}
        row["phase"] = RATIO
        for side in (LABEL, MODEL):
            below = inhale[f"{MEDIAN}_{side}"]
            row[f"{MEDIAN}_{side}"] = (exhale[f"{MEDIAN}_{side}"] / below
                                       if below and below > 0 else float("nan"))
            row[f"{SPANS}_{side}"] = int(min(inhale[f"{SPANS}_{side}"],
                                             exhale[f"{SPANS}_{side}"]))
        row[ERROR] = row[f"{MEDIAN}_{MODEL}"] - row[f"{MEDIAN}_{LABEL}"]
        row[USABLE] = bool(inhale[USABLE] and exhale[USABLE])
        rows.append(row)
    return pd.DataFrame(rows, columns=wide.columns) if rows else pd.DataFrame(columns=wide.columns)


def _stats(error: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Bias, spread and limits - they answer different questions and one hides the others.

    Bias says whether the phase is systematically over- or under-called, which is correctable.
    The limits of agreement say how far a single reading can be, which is not. A mean absolute
    error alone shows neither.
    """
    bias = float(np.mean(error))
    spread = float(np.std(error, ddof=1)) if error.size > 1 else 0.0
    ratio = error / np.maximum(reference, EPS_SEC)
    return {"n": int(error.size), "bias_sec": bias, "mae_sec": float(np.mean(np.abs(error))),
            # Signed, so it keeps the direction the absolute version throws away: a phase called
            # 8% short and one called 8% long have the same relative_mae.
            "relative_bias": float(np.mean(ratio)),
            "sd_sec": spread, "median_error_sec": float(np.median(error)),
            "p5_sec": float(np.percentile(error, 5)),
            "p95_sec": float(np.percentile(error, 95)),
            "loa_low": bias - LOA_Z * spread, "loa_high": bias + LOA_Z * spread,
            # The mean of the per-reading ratio, NOT the MAE over the mean duration. The two
            # differ most on the shortest phase: a 0.15 s error on a 0.3 s stop is 50% here and
            # is weighted like the same error on a 2 s one. That is the intended reading - how
            # wrong a typical span is, in proportion to itself.
            "relative_mae": float(np.mean(np.abs(ratio))),
            "relative_median": float(np.median(np.abs(ratio))),
            f"{LABEL}_mean_sec": float(np.mean(reference))}


def agreement(frame: pd.DataFrame, reference: str, error: str = ERROR) -> pd.DataFrame:
    """One stat row per phase, over the rows where both sides called it. Indexed by phase."""
    rows = {}
    for name in COLUMNS:
        picked = (frame[frame["phase"] == name].dropna(subset=[error, reference])
                  if len(frame) else frame)
        if not len(picked):
            rows[name] = {"n": 0}
            continue
        rows[name] = _stats(picked[error].to_numpy(float), picked[reference].to_numpy(float))
    return pd.DataFrame(rows).T.rename_axis("phase")


def span_agreement(pairs: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """`agreement` over the matched pairs, with what was left unmatched beside it.

    The coverage columns are not decoration: a phase can hold a tiny duration error over the
    breaths it matched and miss half of them, and the error column alone reads as success.
    """
    stats = agreement(pairs, f"{LABEL}_sec")
    if not len(coverage):
        return stats
    totals = coverage.groupby("phase")[[LABEL, MODEL, "matched"]].sum()
    stats[f"{LABEL}_spans"] = totals[LABEL].reindex(stats.index)
    stats[f"{MODEL}_spans"] = totals[MODEL].reindex(stats.index)
    stats["matched_spans"] = totals["matched"].reindex(stats.index)
    stats["matched_fraction"] = (totals["matched"] / totals[LABEL].replace(0, np.nan)
                                ).reindex(stats.index)
    stats[f"unmatched_{MODEL}"] = (totals[MODEL] - totals["matched"]).reindex(stats.index)
    return stats


def signal_agreement(medians: pd.DataFrame) -> pd.DataFrame:
    """`agreement` over the usable signals, with how many were too thin to carry a median."""
    if not len(medians):
        return agreement(medians, f"{MEDIAN}_{LABEL}")
    usable = medians[medians[USABLE]]
    stats = agreement(usable, f"{MEDIAN}_{LABEL}")
    counted = medians.groupby("phase").size().reindex(stats.index)
    stats["signals_seen"] = counted
    stats["signals_thin"] = counted - stats["n"]
    return stats


def describe(stats: pd.DataFrame, unit: str) -> list[str]:
    """One printable line per column. The ratio row carries no `s` - it is dimensionless."""
    lines = []
    for name, row in stats.iterrows():
        if not row.get("n"):
            lines.append(f"  {name:13s} never called by both sides")
            continue
        suffix = UNIT.get(str(name), "")
        lines.append(f"  {name:13s} bias {row['bias_sec']:+.2f}{suffix} "
                     f"({row['relative_bias']:+.0%})  "
                     f"MAE {row['mae_sec']:.2f}{suffix} ({row['relative_mae']:.0%})  "
                     f"labelled mean {row[f'{LABEL}_mean_sec']:.2f}{suffix}  "
                     f"limits {row['loa_low']:+.2f} to {row['loa_high']:+.2f}{suffix}  "
                     f"n={int(row['n'])} {unit if suffix else 'breaths' if unit == 'spans' else unit}")
    return lines
