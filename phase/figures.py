"""Look at the test set: the trace, what the model said, and what the reference says.

A metric says how much agreement there is; it does not say what disagreement *looks* like. These
panels do - and the failure modes they expose are different in kind. A boundary two samples late
reads as a good breath; a breath split into three reads as a bad one; and both can sit at the
same macro F1.

**The model's answer is the shading behind the trace, and the reference is a ribbon underneath.**
That is the labelling app's own layout, and reading it the same way in both places is worth more
than any refinement here: somebody who has spent a morning labelling windows should not have to
learn a second visual language to check what the model did with them. The reference goes below
rather than on top because two overlaid translucent shadings force a reader to decode a colour
mix before they can see whether the two agree.

Colours are `plot.phase_colors` in the config, defaulting to the app's own values - change them
there, not here.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")                       # no display on a build host or in a training run
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import numpy as np

from phase.durations import COLUMNS, LABEL, MEDIAN, MODEL, RATIO, USABLE
from phase.labels import PHASES, UNKNOWN, classes_of, targets_to_spans
from phase.metrics import per_sample
from phase.splits import FOLD_COLUMN, SPLIT_COLUMN, TEST, TRAIN, VAL

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#eeeeee"
ZERO_LINE = "#dddddd"     # the app's zeroline

DEFAULT_COLORS = {"inhale": "#4C9BE8", "exhale": "#E8834C",
                  "stop": "#9AA0A6", "unknown": "#C2A878"}
"""The labelling app's `plot.phase_colors`. Overridden by `plot.phase_colors` in the config."""

DERIVED_COLOR = "#6B6094"
"""For a quantity derived from the phases rather than one of them - the I:E ratio. Deliberately
outside the phase palette: a reader must never wonder whether a colour means `inhale`."""

DEFAULT_SPAN_ALPHA = 0.30
DEFAULT_TRACE = "#222222"

DEFAULT_SPLIT_COLORS = {TRAIN: "#3E7CB1", VAL: "#E8A34C", TEST: "#B5544A"}
"""Splits are not phases and must not borrow their palette - a reader should never have to
ask whether a blue bar means `inhale` or `train`. Overridden by `plot.split_colors`."""

RIBBON_HEIGHT = 0.16
RIBBON_TOP = -0.20
TICK_PAD = 34
"""The x tick labels are pushed below the ribbon rather than the ribbon being squeezed above
them - a ribbon over the axis labels hides the time base, which is the one thing a reader needs
to judge a boundary."""


def colors_of(cfg: Mapping[str, Any] | None) -> dict[str, str]:
    """The palette, config first. Every phase gets a colour or the figure fails here, loudly."""
    chosen = {**DEFAULT_COLORS, **dict((cfg or {}).get("phase_colors", {}) or {})}
    missing = [name for name in PHASES if name not in chosen]
    if missing:
        raise KeyError(f"plot.phase_colors has no colour for {missing}; needs all of {PHASES}")
    return chosen


def _shade(ax, target: np.ndarray, fps: float, colors: Mapping[str, str],
           alpha: float) -> None:
    """The model's answer, as the background the trace is drawn over."""
    for span in targets_to_spans(np.asarray(target)):
        ax.axvspan(span["start"] / fps, span["end"] / fps,
                   facecolor=colors[span["phase"]], alpha=alpha, linewidth=0, zorder=0)


def _ribbon(ax, target: np.ndarray, y: float, fps: float,
            colors: Mapping[str, str]) -> None:
    """The reference, as a solid band under the axes - solid, so it is not the same visual
    language as the shading and cannot be mistaken for it."""
    for span in targets_to_spans(np.asarray(target)):
        ax.add_patch(plt.Rectangle(
            (span["start"] / fps, y), (span["end"] - span["start"]) / fps, RIBBON_HEIGHT,
            facecolor=colors[span["phase"]], edgecolor=SURFACE, linewidth=0.7,
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=3))


def panel(ax, values: np.ndarray, reference: np.ndarray, prediction: np.ndarray, fps: float,
          title: str, colors: Mapping[str, str], span_alpha: float, trace_color: str,
          reference_label: str = "reference", prediction_label: str = "model") -> None:
    # Drawn the way the labelling app draws it: **sample index on x, raw amplitude on y**.
    # The app labels in sample index and shows seconds only in the hover, so plotting seconds
    # against a z-scored y made the same window look like a different signal in the two places.
    # The z-score belongs to the model input (`WindowDataset._normalise`), not to the picture.
    index = np.arange(values.size)

    _shade(ax, prediction, 1.0, colors, span_alpha)
    ax.plot(index, values, color=trace_color, linewidth=1.6, zorder=2)
    ax.set_xlim(0, max(values.size - 1, 1))
    reach = float(np.max(np.abs(values))) or 1.0
    ax.set_ylim(-reach * 1.15, reach * 1.15)
    ax.axhline(0.0, color=ZERO_LINE, linewidth=1.0, zorder=1)
    ax.set_ylabel("amplitude", fontsize=7.5, color=INK_SOFT)
    # Three ticks: the traces run at 1e-3 and a default locator stacks five overlapping labels.
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.grid(color=GRID, linewidth=0.6, zorder=1)
    ax.set_axisbelow(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=7.5, length=0)
    ax.tick_params(axis="x", pad=TICK_PAD)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=8)

    _ribbon(ax, reference, RIBBON_TOP, 1.0, colors)
    # Axes fractions in **both** directions - `get_yaxis_transform` is data-in-y, so a y meant as
    # a fraction lands at that data value instead.
    # On the right: the left side now carries a real amplitude axis, and the row labels
    # collided with it.
    ax.text(1.006, RIBBON_TOP + RIBBON_HEIGHT / 2, reference_label, transform=ax.transAxes,
            ha="left", va="center", fontsize=7.5, color=INK_SOFT)
    ax.text(1.006, 0.5, prediction_label, transform=ax.transAxes, ha="left", va="center",
            fontsize=7.5, color=INK_SOFT)


def legend_handles(colors: Mapping[str, str],
                   classes: tuple[str, ...] = PHASES) -> list[Patch]:
    return [Patch(facecolor=colors[name], edgecolor="none", label=name) for name in classes]


def plot_windows(items: Sequence[dict[str, Any]], out_path: str | Path, heading: str = "",
                 reference_name: str = "algorithm",
                 plot_cfg: Mapping[str, Any] | None = None,
                 prediction_name: str = "model", caption: str | None = None,
                 classes: tuple[str, ...] = PHASES) -> Path:
    """A page of panels. Each item is `{values, reference, prediction, fps, title}`."""
    if not items:
        raise ValueError("nothing to plot")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = dict(plot_cfg or {})
    colors = colors_of(cfg)
    span_alpha = float(cfg.get("span_alpha", DEFAULT_SPAN_ALPHA))
    trace_color = str(cfg.get("trace_color", DEFAULT_TRACE))

    fig, axes = plt.subplots(len(items), 1, figsize=(11, 2.7 * len(items) + 1.3),
                             squeeze=False, facecolor=SURFACE)
    for ax, item in zip(axes.ravel(), items):
        ax.set_facecolor(SURFACE)
        panel(ax, item["values"], item["reference"], item["prediction"], item["fps"],
              item["title"], colors, span_alpha, trace_color, reference_name, prediction_name)
    axes.ravel()[-1].set_xlabel("sample index", fontsize=8, color=INK_SOFT,
                                labelpad=6)

    if heading:
        fig.suptitle(heading, fontsize=10.5, color=INK, x=0.012, ha="left", y=0.997)
    fig.text(0.988, 0.997,
             caption or f"shading = {prediction_name} · ribbon = {reference_name}",
             ha="right", va="top", fontsize=8, color=INK_SOFT)
    fig.legend(handles=legend_handles(colors, classes), loc="lower center", ncol=len(classes),
               frameon=False,
               fontsize=8, labelcolor=INK_SOFT, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0.055, 0.032, 1, 0.978), h_pad=3.0)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


SPREAD, WORST, BEST, RANDOM = "spread", "worst", "best", "random"
PICKS = (SPREAD, WORST, BEST, RANDOM)


def choose(scored: list[dict[str, Any]], n: int, how: str = SPREAD,
           seed: int = 0) -> list[dict[str, Any]]:
    """Which windows to draw.

    `spread` is the default and takes the worst, the median and the best in that order, so a page
    shows the range rather than a flattering sample of it. Picking at random is honest but on a
    set this size mostly returns median windows and hides both tails.
    """
    if not scored:
        return []
    ranked = sorted(scored, key=lambda item: item["score"])
    n = min(n, len(ranked))
    if how == WORST:
        return ranked[:n]
    if how == BEST:
        return ranked[-n:][::-1]
    if how == RANDOM:
        rng = np.random.default_rng(seed)
        return [ranked[i] for i in sorted(rng.choice(len(ranked), n, replace=False))]
    positions = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    return [ranked[position] for position in dict.fromkeys(positions)]


SCORE_LABEL = "accuracy"
"""What a panel title and the window picks are scored by."""


def score(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Accuracy - the number the panel title carries and the picks rank by.

    Not macro F1: on one window a class present for a single sample enters the macro average at
    zero, so a window 96% right scored 0.64. Pooled scores keep macro F1.
    """
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    return float(np.mean(prediction == reference)) if reference.size else 0.0


def macro_f1(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Macro F1 over the classes in play, for a caller that records it under that name."""
    return float(per_sample(prediction, reference)["macro_f1"])


# ------------------------------------------------------------------------------------- splits

SPLIT_ORDER = (TRAIN, VAL, TEST)
SPLITS_DIR = "splits"


def split_colors_of(cfg: Mapping[str, Any] | None) -> dict[str, str]:
    chosen = {**DEFAULT_SPLIT_COLORS, **dict((cfg or {}).get("split_colors", {}) or {})}
    missing = [name for name in SPLIT_ORDER if name not in chosen]
    if missing:
        raise KeyError(f"plot.split_colors has no colour for {missing}")
    return chosen


def split_summary(manifest, fold: int | None, out_path: str | Path,
                  plot_cfg: Mapping[str, Any] | None = None) -> Path:
    """Where every window went, per patient.

    `fold=None` draws the held-out split alone - train against test, which is the same in every
    fold and is the division a result is reported against. An integer draws that fold, where
    validation is carved out of the training side.

    Per patient rather than per split total, because the split is *by patient* - a total says
    the proportions came out right while hiding that one person carries a third of the test set.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = split_colors_of(dict(plot_cfg or {}))
    if fold is None:
        column, order = SPLIT_COLUMN, (TRAIN, TEST)
    else:
        column, order = FOLD_COLUMN.format(fold=fold), SPLIT_ORDER
    if column not in manifest.columns:
        raise KeyError(f"{column} not in the manifest - run make_splits.py first")

    counts = (manifest.groupby(["env", "PatientID", column]).size().unstack(fill_value=0)
              .reindex(columns=list(order), fill_value=0))
    # Ordered by which split a patient belongs to, then by size: patients land wholly in one
    # split, so grouping them makes the by-patient rule visible instead of implied.
    counts["_where"] = [order.index(row.idxmax()) for _, row in counts.iterrows()]
    counts = counts.sort_values(["_where", "env"]).drop(columns="_where")

    labels = [f"{patient}  ({env.replace('ds_', '')})" for env, patient in counts.index]
    totals = manifest[column].value_counts()
    windows = len(manifest)

    height = 0.26 * len(counts) + 1.9
    fig, (ax, bar) = plt.subplots(
        2, 1, figsize=(10, height), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [0.26 * len(counts), 0.62],
                     "hspace": min(0.30, 6.0 / max(len(counts), 1))})

    left = np.zeros(len(counts))
    for split in order:
        values = counts[split].to_numpy()
        ax.barh(labels, values, left=left, color=colors[split], label=split,
                height=0.74, linewidth=0)
        left += values
    for y, total in enumerate(left):
        ax.text(total + windows * 0.004, y, f"{int(total)}", va="center", fontsize=7.5,
                color=INK_SOFT)
    ax.set_xlim(0, left.max() * 1.10)
    ax.invert_yaxis()
    ax.set_xlabel("windows", fontsize=8, color=INK_SOFT)
    ax.tick_params(labelsize=7.5, colors=INK_SOFT, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(f"{len(counts)} patients, one split each", fontsize=8.5, color=INK,
                 loc="left", pad=6)

    start = 0.0
    for split in order:
        n = int(totals.get(split, 0))
        if not n:
            continue
        bar.barh([0], [n], left=[start], color=colors[split], height=0.62, linewidth=0)
        share = 100 * n / windows
        bar.text(start + n / 2, 0, f"{split}\n{n}  ({share:.0f}%)", ha="center", va="center",
                 fontsize=8, color=SURFACE, fontweight="bold")
        start += n
    bar.set_xlim(0, windows)
    bar.set_ylim(-0.5, 0.5)
    bar.axis("off")
    heading = "held out once" if fold is None else f"fold {fold}"
    bar.set_title(f"{heading}: {windows} windows, "
                  f"{manifest['RadarSignalID'].nunique()} signals, "
                  f"{manifest.groupby(['env', 'PatientID']).ngroups} patients",
                  fontsize=8.5, color=INK, loc="left", pad=4)

    fig.suptitle("who went where", fontsize=10.5, color=INK, x=0.012, ha="left", y=0.998)
    fig.text(0.988, 0.998, "grouped by patient - no patient is split across two parts",
             ha="right", va="top", fontsize=8, color=INK_SOFT)
    # `bar.axis("off")` is not tight_layout-compatible, so the margins are set directly.
    fig.subplots_adjust(left=0.19, right=0.97, top=1 - 0.55 / height, bottom=0.42 / height)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------------ fold report

def fold_report(per_fold, ensemble: Mapping[str, float], interval: tuple[float, float],
                matrix: np.ndarray, out_path: str | Path,
                plot_cfg: Mapping[str, Any] | None = None,
                headline: str = "macro_f1") -> Path:
    """Per-fold scores, the ensemble against them, and where the ensemble confuses classes.

    The fold dots and the ensemble line are drawn together on purpose: the ensemble is the number
    to quote, and seeing it beside the five it came from is what stops it being read as five
    models agreeing when they did not.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = colors_of(dict(plot_cfg or {}))
    splits = split_colors_of(dict(plot_cfg or {}))

    fig, (dots, per_class, heat) = plt.subplots(
        1, 3, figsize=(13, 4.0), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.0, 1.25, 1.15], "wspace": 0.30})

    # ------------------------------------------------------------------ fold spread
    scores = per_fold[headline].to_numpy() if headline in per_fold else np.array([])
    if scores.size:
        x = np.arange(scores.size)
        dots.scatter(x, scores, s=52, color=splits[TRAIN], zorder=3, label="fold")
        low, high = interval
        dots.axhspan(low, high, color=splits[TEST], alpha=0.13, zorder=0,
                     label="ensemble 95% CI")
        dots.axhline(ensemble[headline], color=splits[TEST], linewidth=1.8, zorder=2,
                     label="ensemble")
        dots.set_xticks(x)
        dots.set_xticklabels([name.replace("fold_", "") for name in per_fold.index],
                            fontsize=7.5)
        dots.set_xlim(-0.6, scores.size - 0.4)
        dots.set_ylim(0, 1)
        dots.set_xlabel("fold", fontsize=8, color=INK_SOFT)
        dots.legend(frameon=False, fontsize=7, labelcolor=INK_SOFT, loc="lower right")
    dots.set_title(headline.replace("_", " "), fontsize=8.5, color=INK, loc="left", pad=6)
    dots.grid(axis="y", color=GRID, linewidth=0.6)
    dots.set_axisbelow(True)
    dots.tick_params(labelsize=7.5, colors=INK_SOFT, length=0)
    for side in ("top", "right", "left"):
        dots.spines[side].set_visible(False)
    dots.spines["bottom"].set_color(GRID)

    # ------------------------------------------------------------------ per class
    classes = classes_of(matrix.shape[0])
    y = np.arange(len(classes))
    per_class.barh(y, [ensemble.get(f"f1_{name}", np.nan) for name in classes],
                   height=0.62, linewidth=0, color=[colors[name] for name in classes])
    per_class.set_yticks(y)
    per_class.set_yticklabels(classes, fontsize=7.5)
    per_class.invert_yaxis()
    per_class.set_xlim(0, 1)
    per_class.set_xlabel("per-sample F1", fontsize=8, color=INK_SOFT)
    per_class.set_title("ensemble, by class", fontsize=8.5, color=INK, loc="left", pad=6)
    per_class.grid(axis="x", color=GRID, linewidth=0.6)
    per_class.set_axisbelow(True)
    per_class.tick_params(labelsize=7.5, colors=INK_SOFT, length=0)
    for side in ("top", "right", "left"):
        per_class.spines[side].set_visible(False)
    per_class.spines["bottom"].set_color(GRID)

    # ------------------------------------------------------------------ confusion
    heat.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    heat.set_xticks(range(len(classes)), classes, fontsize=7, rotation=35, ha="right")
    heat.set_yticks(range(len(classes)), classes, fontsize=7)
    heat.set_xlabel("predicted", fontsize=8, color=INK_SOFT)
    heat.set_ylabel("labelled", fontsize=8, color=INK_SOFT)
    heat.set_title("row-normalised confusion", fontsize=8.5, color=INK, loc="left", pad=6)
    for i in range(len(classes)):
        for j in range(len(classes)):
            heat.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                      color=SURFACE if matrix[i, j] > 0.55 else INK)
    heat.tick_params(colors=INK_SOFT, length=0)
    for spine in heat.spines.values():
        spine.set_visible(False)

    fig.suptitle("test set - five folds and their ensemble", fontsize=10.5, color=INK,
                 x=0.008, ha="left", y=0.995)
    fig.text(0.992, 0.995, "one held-out test set, shared by every fold", ha="right", va="top",
             fontsize=8, color=INK_SOFT)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.135)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------- per patient

def patient_report(by_patient, ensemble: Mapping[str, float], out_path: str | Path,
                   plot_cfg: Mapping[str, Any] | None = None) -> Path:
    """Weighted Dice per test patient, worst first, against the pooled score.

    Sorted rather than alphabetical: the question a per-patient chart answers is who it fails on,
    and that is the top of the list. The window count sits on each bar because a low score on
    four windows and a low score on forty are different problems.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = colors_of(dict(plot_cfg or {}))
    splits = split_colors_of(dict(plot_cfg or {}))

    labels = [f"{row.patient}  ({int(row.windows)}w)" for row in by_patient.itertuples()]
    y = np.arange(len(by_patient))
    height = 0.28 * len(by_patient) + 1.9

    fig, (bars, classes) = plt.subplots(
        1, 2, figsize=(12, height), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.26})

    bars.barh(y, by_patient["weighted_dice"], height=0.66, color=splits[TEST], linewidth=0,
              label="weighted Dice")
    bars.scatter(by_patient["macro_f1"], y, s=26, color=INK, zorder=3, label="macro F1")
    pooled = ensemble.get("macro_f1")
    if pooled is not None:
        bars.axvline(pooled, color=INK_SOFT, linewidth=1.2, linestyle="--", zorder=2,
                     label=f"pooled macro F1 {pooled:.2f}")
    for position, value in zip(y, by_patient["weighted_dice"]):
        bars.text(value + 0.012, position, f"{value:.2f}", va="center", fontsize=7.5,
                  color=INK_SOFT)
    bars.set_yticks(y, labels, fontsize=7.5)
    bars.invert_yaxis()
    bars.set_xlim(0, 1.06)
    bars.set_xlabel("weighted Dice, support-weighted over inhale / exhale / stop", fontsize=8,
                    color=INK_SOFT)
    # Below the axes: every bar runs past 0.7, so any in-axes corner collides with one.
    bars.legend(frameon=False, fontsize=7, labelcolor=INK_SOFT, ncol=3,
                loc="upper center", bbox_to_anchor=(0.5, -0.13))
    bars.grid(axis="x", color=GRID, linewidth=0.6)
    bars.set_axisbelow(True)
    bars.tick_params(colors=INK_SOFT, length=0)
    for side in ("top", "right", "left"):
        bars.spines[side].set_visible(False)
    bars.spines["bottom"].set_color(GRID)
    bars.set_title("per test patient, worst first", fontsize=8.5, color=INK, loc="left", pad=6)

    called = [name for name in PHASES if f"f1_{name}" in by_patient.columns]
    left = np.zeros(len(by_patient))
    width = 1.0 / max(len(called), 1)
    for offset, name in enumerate(called):
        classes.barh(y + (offset - (len(called) - 1) / 2) * width * 0.8,
                     by_patient[f"f1_{name}"], height=width * 0.74,
                     color=colors[name], linewidth=0, label=name)
    classes.set_yticks(y, ["" for _ in y])
    classes.invert_yaxis()
    classes.set_xlim(0, 1)
    classes.set_xlabel("F1 by class", fontsize=8, color=INK_SOFT)
    classes.legend(frameon=False, fontsize=7, labelcolor=INK_SOFT, ncol=3,
                   loc="upper center", bbox_to_anchor=(0.5, -0.13))
    classes.grid(axis="x", color=GRID, linewidth=0.6)
    classes.set_axisbelow(True)
    classes.tick_params(colors=INK_SOFT, length=0)
    for side in ("top", "right", "left"):
        classes.spines[side].set_visible(False)
    classes.spines["bottom"].set_color(GRID)
    classes.set_title("which class fails, per patient", fontsize=8.5, color=INK, loc="left",
                      pad=6)

    fig.suptitle("test set - the ensemble, patient by patient", fontsize=10.5, color=INK,
                 x=0.008, ha="left", y=0.997)
    fig.subplots_adjust(left=0.145, right=0.985, top=1 - 0.55 / height,
                        bottom=1.05 / height)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------- fold scaling

def scaling_report(scaling, out_path: str | Path, plot_cfg: Mapping[str, Any] | None = None,
                   headline: str = "macro_f1") -> Path:
    """What each additional fold buys. Every subset of size k is a point, the line is their mean.

    Points rather than a bare line: with five folds there are ten ways to pick two, and the
    scatter across them is the part that says whether a gain is real or an ordering artefact.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits = split_colors_of(dict(plot_cfg or {}))

    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    means = scaling.groupby("folds_used")[headline].mean()
    for k, group in scaling.groupby("folds_used"):
        jitter = np.linspace(-0.13, 0.13, len(group)) if len(group) > 1 else np.zeros(1)
        ax.scatter(k + jitter, group[headline], s=26, color=splits[TRAIN], alpha=0.75,
                   zorder=3, label="one subset" if k == 1 else None)
    ax.plot(means.index, means.to_numpy(), color=splits[TEST], linewidth=1.8, zorder=4,
            marker="o", markersize=5, label="mean over subsets")

    ax.set_xticks(list(means.index))
    ax.set_xlabel("folds ensembled", fontsize=8.5, color=INK_SOFT)
    ax.set_ylabel(headline.replace("_", " "), fontsize=8.5, color=INK_SOFT)
    ax.set_title(f"{means.iloc[0]:.3f} with one fold, {means.iloc[-1]:.3f} with "
                 f"{int(means.index[-1])}", fontsize=8.5, color=INK, loc="left", pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, colors=INK_SOFT, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_SOFT, loc="lower right")

    fig.suptitle("what each fold is worth on the test set", fontsize=10.5, color=INK,
                 x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------- raw against decoded

def decoding_report(table, out_path: str | Path,
                    plot_cfg: Mapping[str, Any] | None = None,
                    raw: str = "raw", decoded: str = "viterbi") -> Path:
    """What the decoder changed, metric by metric.

    Both bars come from the **same logits** - the decoder is post-processing, so the network is
    identical in each and the difference is the decoding alone. Drawn whether it helped or not:
    a post-processing step that costs accuracy should be as visible as one that earns it.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits = split_colors_of(dict(plot_cfg or {}))

    # Per-sample and duration only. Event F1 scores whole spans by IoU, which is a proxy for
    # fragmentation rather than for the durations the device reports - it moved the most and
    # meant the least.
    keep = [m for m in ("macro_f1", "accuracy", *[f"f1_{n}" for n in PHASES])
            if m in set(table["metric"])]
    part = table[table.metric.isin(keep)].set_index("metric").loc[keep]

    y = np.arange(len(part))
    height = 0.34 * len(part) + 1.8
    fig, (bars, delta) = plt.subplots(
        1, 2, figsize=(11, height), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.08}, sharey=True)

    bars.barh(y - 0.19, part[raw], height=0.36, color="#b6b4b0", linewidth=0, label="argmax")
    bars.barh(y + 0.19, part[decoded], height=0.36, color=splits[TEST], linewidth=0,
              label="after viterbi")
    bars.set_yticks(y, list(part.index), fontsize=7.5)
    bars.invert_yaxis()
    bars.set_xlim(0, 1)
    bars.set_xlabel("score", fontsize=8, color=INK_SOFT)
    bars.legend(frameon=False, fontsize=7.5, labelcolor=INK_SOFT, loc="lower right")
    bars.set_title("same logits, decoded and not", fontsize=8.5, color=INK, loc="left", pad=6)

    widest = float(np.abs(part["delta"]).max()) or 0.01
    delta.barh(y, part["delta"], height=0.5, linewidth=0,
               color=["#2e7d5b" if v >= 0 else "#c0392b" for v in part["delta"]])
    for position, value in zip(y, part["delta"]):
        delta.text(value + (0.04 * widest if value >= 0 else -0.04 * widest), position,
                   f"{value:+.3f}", va="center",
                   ha="left" if value >= 0 else "right", fontsize=7, color=INK_SOFT)
    delta.axvline(0, color=INK_SOFT, linewidth=1.0)
    delta.set_xlim(-widest * 1.6, widest * 1.6)
    delta.set_xlabel("viterbi - argmax", fontsize=8, color=INK_SOFT)
    delta.set_title("what the decoder was worth", fontsize=8.5, color=INK, loc="left", pad=6)

    for ax in (bars, delta):
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5, colors=INK_SOFT, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    fig.suptitle("post-processing: Viterbi over the network's own output", fontsize=10.5,
                 color=INK, x=0.008, ha="left", y=0.995)
    fig.subplots_adjust(left=0.16, right=0.985, top=1 - 0.5 / height, bottom=0.55 / height)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ------------------------------------------------------------------------ phase durations

def duration_report(pairs, agreement, out_path: str | Path,
                    plot_cfg: Mapping[str, Any] | None = None) -> Path:
    """Bland-Altman per phase: does the model call the same duration the labeller did?

    One panel per phase, each window a point - x is the mean of the two readings, y their
    difference. The solid line is the bias and the dashed pair the 95% limits of agreement.
    Bias and limits answer different questions: a constant offset can be corrected, a wide
    spread cannot, and a mean absolute error alone shows neither.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = colors_of(dict(plot_cfg or {}))
    named = list(agreement)

    fig, axes = plt.subplots(1, len(named), figsize=(4.3 * len(named), 3.9), facecolor=SURFACE,
                             squeeze=False)
    for ax, name in zip(axes.ravel(), named):
        ax.set_facecolor(SURFACE)
        stats = agreement.get(name, {})
        both = [(said, meant) for window in pairs
                for said, meant in [window[name]]
                if not (np.isnan(said) or np.isnan(meant))]
        if both:
            said = np.array([a for a, _ in both])
            meant = np.array([b for _, b in both])
            ax.scatter((said + meant) / 2, said - meant, s=22, alpha=0.65,
                       color=colors[name], zorder=3)
            for value, style, label in ((stats["bias_sec"], "-", "bias"),
                                        (stats["loa_low"], "--", None),
                                        (stats["loa_high"], "--", "95% limits")):
                ax.axhline(value, color=INK_SOFT, linewidth=1.1, linestyle=style, zorder=2,
                           label=label)
            ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=1)
            ax.set_title(
                f"{name}  ·  bias {stats['bias_sec']:+.2f}s  ·  MAE {stats['mae_sec']:.2f}s\n"
                f"limits {stats['loa_low']:+.2f} to {stats['loa_high']:+.2f}s  ·  "
                f"n={stats['n']} ({stats['coverage']:.0%} of windows)",
                fontsize=8, color=INK, loc="left", pad=6)
        else:
            ax.set_title(f"{name} - never called by both", fontsize=8, color=INK_SOFT,
                         loc="left", pad=6)
        ax.set_xlabel("mean of the two, seconds", fontsize=8, color=INK_SOFT)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5, colors=INK_SOFT, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes.ravel()[0].set_ylabel("model - labeller, seconds", fontsize=8, color=INK_SOFT)
    axes.ravel()[0].legend(frameon=False, fontsize=7, labelcolor=INK_SOFT, loc="lower right")

    fig.suptitle("phase duration agreement - the number the device reports", fontsize=10.5,
                 color=INK, x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _tidy(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, colors=INK_SOFT, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK_SOFT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_SOFT)


def _bland_altman(ax, labelled: np.ndarray, called: np.ndarray, row, color: str) -> None:
    """A Bland-Altman plot: x the mean of the two readings, y their difference.

    The standard way two measurements of the same thing are compared, and the reason the bias and
    the limits are drawn rather than a correlation: a regression of one on the other can be near
    perfect while every reading is half a second out.
    """
    ax.scatter((labelled + called) / 2, called - labelled, s=20, alpha=0.6, color=color,
               zorder=3, edgecolors="none")
    for value, style, label in ((row["bias_sec"], "-", "bias"),
                                (row["loa_low"], "--", None),
                                (row["loa_high"], "--", "95% limits")):
        ax.axhline(value, color=INK_SOFT, linewidth=1.1, linestyle=style, zorder=2, label=label)
    ax.axhline(0.0, color=ZERO_LINE, linewidth=1.0, zorder=1)


def _identity_line(ax, labelled: np.ndarray, called: np.ndarray) -> None:
    both = np.concatenate([labelled, called])
    low, high = float(np.min(both)), float(np.max(both))
    pad = 0.05 * max(high - low, 0.1)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=INK_SOFT, linewidth=1.0,
            linestyle="--", zorder=2, label="equal")


def _coverage(row, unit: str) -> tuple[str, str]:
    """`(what was counted, what was not)`. A small duration error over half the spans is not
    agreement, so what fell out belongs on the figure and not only in the csv."""
    if f"{LABEL}_spans" in row and not np.isnan(row.get(f"{LABEL}_spans", np.nan)):
        return (f"n={int(row['n'])} of {int(row[f'{LABEL}_spans'])} labelled {unit} matched "
                f"({row['matched_fraction']:.0%})",
                f"\n{int(row[f'unmatched_{MODEL}'])} predicted spans matched nothing")
    if "signals_seen" in row and not np.isnan(row.get("signals_seen", np.nan)):
        return (f"n={int(row['n'])} of {int(row['signals_seen'])} {unit}",
                f"\n{int(row['signals_thin'])} signals too thin for a median")
    return f"n={int(row['n'])} {unit}", ""


CAPTION_CHARS_PER_INCH = 17
"""Characters of an 8 pt caption per inch of figure width, for wrapping it to the grid."""

CAPTION_LINE_HEIGHT = 0.022
"""Figure fraction a wrapped caption line takes beyond the two the layout already leaves room for."""


def _duration_grid(frame, stats, out_path: str | Path, heading: str, subtitle: str,
                   labelled_col: str, model_col: str, unit: str,
                   plot_cfg: Mapping[str, Any] | None = None) -> Path:
    """Two rows per quantity, because they are read differently and neither replaces the other.

    **Bland-Altman** says whether the error depends on the reading itself - a model right on a 1 s
    inhale and short on a 2 s one is a different problem from one uniformly short - and puts the
    bias and the limits of agreement on the same picture.

    **Model against labeller**, with the line of equality, is the one a reader checks a single
    recording against: it shows the range each side actually calls, which a difference plot throws
    away, and a systematic slope shows as a fan away from the line.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = colors_of(dict(plot_cfg or {}))

    # The columns the stats carry, so a merged class has no empty panel.
    columns = [name for name in COLUMNS if name in stats.index]
    fig, axes = plt.subplots(2, len(columns), figsize=(4.15 * len(columns), 7.0),
                             facecolor=SURFACE, squeeze=False)
    for column, name in enumerate(columns):
        # The ratio is dimensionless and is not a phase, so it takes neither the phase palette
        # nor the seconds axis.
        is_ratio = name == RATIO
        tint = DERIVED_COLOR if is_ratio else colors[name]
        axis = "ratio" if is_ratio else "seconds"
        suffix = "" if is_ratio else "s"
        altman, against = axes[0][column], axes[1][column]
        row = stats.loc[name] if name in stats.index else {"n": 0}
        picked = (frame[(frame["phase"] == name)].dropna(subset=[labelled_col, model_col])
                  if len(frame) else frame)
        if not row.get("n") or not len(picked):
            for ax in (altman, against):
                _tidy(ax)
                ax.set_title(f"{name} - never called by both", fontsize=8, color=INK_SOFT,
                             loc="left", pad=6)
            continue
        labelled = picked[labelled_col].to_numpy(float)
        called = picked[model_col].to_numpy(float)

        counted, dropped = _coverage(row, "breaths" if is_ratio and unit == "spans" else unit)
        _bland_altman(altman, labelled, called, row, tint)
        _tidy(altman, f"mean of the two, {axis}")
        altman.set_title(
            f"{name}  ·  bias {row['bias_sec']:+.2f}{suffix} "
            f"({row['relative_bias']:+.0%})  ·  "
            f"MAE {row['mae_sec']:.2f}{suffix} ({row['relative_mae']:.0%})"
            f"\nlimits {row['loa_low']:+.2f} to "
            f"{row['loa_high']:+.2f}{suffix}  ·  {counted}",
            fontsize=8, color=INK, loc="left", pad=6)

        against.scatter(labelled, called, s=20, alpha=0.6, color=tint, zorder=3,
                        edgecolors="none")
        _identity_line(against, labelled, called)
        _tidy(against, f"labeller, {axis}")
        # The distribution used to be a third row. What it carried that the two rows above do
        # not is the median error and the 5-95% band, which are asymmetry - so those stay, as
        # text, rather than the panel.
        against.set_title(f"labelled mean {row[f'{LABEL}_mean_sec']:.2f}{suffix}  ·  "
                          f"median error {row['median_error_sec']:+.2f}{suffix}  ·  "
                          f"5-95% {row['p5_sec']:+.2f} to {row['p95_sec']:+.2f}{suffix}"
                          f"{dropped}",
                          fontsize=8, color=INK_SOFT, loc="left", pad=6)

    axes[0][0].set_ylabel("Bland-Altman: model - labeller", fontsize=8, color=INK_SOFT)
    axes[1][0].set_ylabel("model", fontsize=8, color=INK_SOFT)
    for ax in (axes[0][0], axes[1][0]):
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=False, fontsize=7, labelcolor=INK_SOFT, loc="best")

    fig.suptitle(heading, fontsize=10.5, color=INK, x=0.008, ha="left", y=1.0)
    # Wrapped to the grid, or one long caption line sets the saved width and leaves it half empty.
    width = int(CAPTION_CHARS_PER_INCH * fig.get_figwidth())
    wrapped = "\n".join(textwrap.fill(line, width) for line in subtitle.split("\n"))
    fig.text(0.008, 0.968, wrapped, fontsize=8, color=INK_SOFT, ha="left", va="top",
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.905 - CAPTION_LINE_HEIGHT * (wrapped.count("\n") - 1)))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def span_duration_report(pairs, stats, out_path: str | Path,
                         plot_cfg: Mapping[str, Any] | None = None) -> Path:
    """The duration error on a single breath - one point per matched span."""
    return _duration_grid(
        pairs, stats, out_path,
        "phase duration, breath by breath - the time the network says was spent in the phase",
        "ROW 1: Bland-Altman - x is the mean of the two readings, y their difference; the solid "
        "line is the bias and the dashed pair the 95% limits of agreement, bias \u00b1 1.96 SD.  "
        "ROW 2: model against labeller, with the line of equality.\nOne point per labelled span, matched to the prediction of the same phase by "
        "IoU; spans the window boundary cut are excluded from both sides. The last column is one "
        "breath's exhale over its own inhale, where both spans of that breath matched. Percentages "
        "are per reading against the labelled value itself - signed beside the bias, absolute "
        "beside the MAE.",
        f"{LABEL}_sec", f"{MODEL}_sec", "spans", plot_cfg)


def signal_duration_report(medians, stats, out_path: str | Path,
                           plot_cfg: Mapping[str, Any] | None = None) -> Path:
    """The number a session report would carry - one point per recording."""
    return _duration_grid(
        medians[medians[USABLE]] if len(medians) else medians, stats, out_path,
        "phase duration per recording - the median breath, model against labeller",
        "ROW 1: Bland-Altman - x is the mean of the two readings, y their difference; the solid "
        "line is the bias and the dashed pair the 95% limits of agreement, bias \u00b1 1.96 SD.  "
        "ROW 2: model against labeller, with the line of equality.\nOne point per radar signal: the median span length each side called over "
        "that signal's windows, unmatched. Boundary-cut spans excluded; a signal needs enough spans "
        "on both sides to carry a median. The last column is the ratio of that signal's two "
        "medians, usable only where both phases were. Percentages are per signal against the "
        "labelled median - signed beside the bias, absolute beside the MAE.",
        f"{MEDIAN}_{LABEL}", f"{MEDIAN}_{MODEL}", "signals", plot_cfg)
