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

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")                       # no display on a build host or in a training run
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np

from phase.labels import PHASES, UNKNOWN, targets_to_spans
from phase.metrics import per_sample

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#eeeeee"

DEFAULT_COLORS = {"inhale": "#4C9BE8", "exhale": "#E8834C",
                  "stop": "#9AA0A6", "unknown": "#C2A878"}
"""The labelling app's `plot.phase_colors`. Overridden by `plot.phase_colors` in the config."""

DEFAULT_SPAN_ALPHA = 0.30
DEFAULT_TRACE = "#222222"

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
    t = np.arange(values.size) / fps
    centred = values - values.mean()
    scale = centred.std() or 1.0

    _shade(ax, prediction, fps, colors, span_alpha)
    ax.plot(t, centred / scale, color=trace_color, linewidth=1.5, zorder=2)
    ax.set_xlim(0, max(t[-1], 1e-6))
    ax.set_ylim(-3.2, 3.2)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=1)
    ax.set_axisbelow(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=7.5, length=0, pad=TICK_PAD)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=8)

    _ribbon(ax, reference, RIBBON_TOP, fps, colors)
    # Axes fractions in **both** directions - `get_yaxis_transform` is data-in-y, so a y meant as
    # a fraction lands at that data value instead.
    ax.text(-0.008, RIBBON_TOP + RIBBON_HEIGHT / 2, reference_label, transform=ax.transAxes,
            ha="right", va="center", fontsize=7.5, color=INK_SOFT)
    ax.text(-0.008, 0.5, prediction_label, transform=ax.transAxes, ha="right", va="center",
            fontsize=7.5, color=INK_SOFT)


def legend_handles(colors: Mapping[str, str]) -> list[Patch]:
    return [Patch(facecolor=colors[name], edgecolor="none", label=name) for name in PHASES]


def plot_windows(items: Sequence[dict[str, Any]], out_path: str | Path, heading: str = "",
                 reference_name: str = "algorithm",
                 plot_cfg: Mapping[str, Any] | None = None,
                 prediction_name: str = "model", caption: str | None = None) -> Path:
    """A page of panels. Each item is `{values, reference, prediction, fps, title}`."""
    if not items:
        raise ValueError("nothing to plot")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = dict(plot_cfg or {})
    colors = colors_of(cfg)
    span_alpha = float(cfg.get("span_alpha", DEFAULT_SPAN_ALPHA))
    trace_color = str(cfg.get("trace_color", DEFAULT_TRACE))

    fig, axes = plt.subplots(len(items), 1, figsize=(11, 2.2 * len(items) + 1.3),
                             squeeze=False, facecolor=SURFACE)
    for ax, item in zip(axes.ravel(), items):
        ax.set_facecolor(SURFACE)
        panel(ax, item["values"], item["reference"], item["prediction"], item["fps"],
              item["title"], colors, span_alpha, trace_color, reference_name, prediction_name)
    axes.ravel()[-1].set_xlabel("seconds", fontsize=8, color=INK_SOFT, labelpad=6)

    if heading:
        fig.suptitle(heading, fontsize=10.5, color=INK, x=0.012, ha="left", y=0.997)
    fig.text(0.988, 0.997,
             caption or f"shading = {prediction_name} · ribbon = {reference_name}",
             ha="right", va="top", fontsize=8, color=INK_SOFT)
    fig.legend(handles=legend_handles(colors), loc="lower center", ncol=4, frameon=False,
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


def score(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Macro F1 over the called classes - the number the panel title carries."""
    return float(per_sample(prediction, reference)["macro_f1"])
