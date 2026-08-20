"""Look at the test set: the trace, what the reference says, and what the model said.

A metric says how much agreement there is; it does not say what disagreement *looks* like. These
panels do - and the failure modes they expose are different in kind. A boundary that is two
samples late reads as a good breath; a breath split into three reads as a bad one; and both can
sit at the same macro F1.

**Two ribbons, not overlapping shading.** The waveform is drawn once, and the two segmentations
sit under it as separate bands. Overlaid translucent spans force a reader to decode a colour mix
before they can see whether the two agree; stacked ribbons make a disagreement a vertical
mismatch, which needs no decoding at all.

The palette is `[#2a78d6, #eb6834, #4a3aa7]` for inhale / exhale / stop, and it is not the one the
review repos use. Theirs pairs a grey stop with a sand unknown, which measure ΔE 7.2 apart in
normal vision - below the readability floor, and those two classes are frequently adjacent here
(the stop sits between breaths, the unknown at each crest). This one passes the lightness,
chroma, CVD-separation, normal-vision and contrast checks. `unknown` is deliberately not a
categorical hue: it is the absence of a claim, so it is drawn as a pale hatched band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")                       # no display on a build host or in a training run
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np

from phase.labels import EXHALE, INHALE, PHASES, STOP, UNKNOWN, targets_to_spans
from phase.metrics import per_sample

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
TRACE = "#3d3d3a"
GRID = "#e4e3dd"

PHASE_COLOR = {
    PHASES[INHALE]: "#2a78d6",
    PHASES[EXHALE]: "#eb6834",
    PHASES[STOP]: "#4a3aa7",
    PHASES[UNKNOWN]: "#e8e6dd",
}
UNKNOWN_HATCH = "///"
"""Texture, so the absence of a claim is legible without colour - in print, under CVD, and
against the pale band's low contrast with the surface."""

RIBBON_HEIGHT = 0.17
RIBBON_GAP = 0.06
RIBBON_TOP = -0.22
"""Where the first ribbon sits, in axes fractions below the trace. The x tick labels are pushed
below both ribbons rather than the ribbons being squeezed above them - a ribbon drawn over the
axis labels hides the time base, which is the one thing a reader needs to judge a boundary."""
TICK_PAD = 52


def _ribbon(ax, target: np.ndarray, y: float, fps: float) -> None:
    for span in targets_to_spans(np.asarray(target)):
        colour = PHASE_COLOR[span["phase"]]
        unknown = span["phase"] == PHASES[UNKNOWN]
        ax.add_patch(plt.Rectangle(
            (span["start"] / fps, y), (span["end"] - span["start"]) / fps, RIBBON_HEIGHT,
            facecolor=colour, edgecolor=SURFACE if not unknown else "#c9c7bd",
            linewidth=0.8, hatch=UNKNOWN_HATCH if unknown else None,
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=3))


def panel(ax, values: np.ndarray, reference: np.ndarray, prediction: np.ndarray,
          fps: float, title: str) -> None:
    """One window: the trace, then the reference ribbon, then the model's."""
    t = np.arange(values.size) / fps
    centred = values - values.mean()
    scale = centred.std() or 1.0

    ax.plot(t, centred / scale, color=TRACE, linewidth=1.4, zorder=2)
    ax.set_xlim(0, max(t[-1], 1e-6))
    ax.set_ylim(-3.2, 3.2)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=7.5, length=0, pad=TICK_PAD)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=10)

    rows = ((reference, RIBBON_TOP), (prediction, RIBBON_TOP - RIBBON_HEIGHT - RIBBON_GAP))
    for target, y in rows:
        _ribbon(ax, target, y, fps)
    # Axes fractions in **both** directions - `get_yaxis_transform` is data-in-y, so a y meant as
    # a fraction lands at that data value and the two labels pile up on each other.
    for label, (_, y) in zip(("reference", "model"), rows):
        ax.text(-0.008, y + RIBBON_HEIGHT / 2, label, transform=ax.transAxes,
                ha="right", va="center", fontsize=7.5, color=INK_SOFT)


def legend_handles() -> list[Patch]:
    handles = []
    for name in PHASES:
        unknown = name == PHASES[UNKNOWN]
        handles.append(Patch(facecolor=PHASE_COLOR[name], edgecolor="#c9c7bd" if unknown else SURFACE,
                             hatch=UNKNOWN_HATCH if unknown else None, label=name))
    return handles


def plot_windows(items: Sequence[dict[str, Any]], out_path: str | Path,
                 heading: str = "", reference_name: str = "algorithm") -> Path:
    """A page of panels. Each item is `{values, reference, prediction, fps, title}`."""
    if not items:
        raise ValueError("nothing to plot")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(items), 1, figsize=(11, 2.3 * len(items) + 1.4),
                             squeeze=False, facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.95)
    for ax, item in zip(axes.ravel(), items):
        ax.set_facecolor(SURFACE)
        panel(ax, item["values"], item["reference"], item["prediction"], item["fps"],
              item["title"])
    axes.ravel()[-1].set_xlabel("seconds", fontsize=8, color=INK_SOFT, labelpad=8)

    if heading:
        fig.suptitle(heading, fontsize=10.5, color=INK, x=0.012, ha="left", y=0.995)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False,
               fontsize=8, labelcolor=INK_SOFT, bbox_to_anchor=(0.5, -0.004))
    fig.text(0.988, 0.995, f"reference: {reference_name}", ha="right", fontsize=8,
             color=INK_SOFT)
    fig.tight_layout(rect=(0.055, 0.035, 1, 0.975), h_pad=3.4)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


SPREAD, WORST, BEST, RANDOM = "spread", "worst", "best", "random"
PICKS = (SPREAD, WORST, BEST, RANDOM)


def choose(scored: list[dict[str, Any]], n: int, how: str = SPREAD,
           seed: int = 0) -> list[dict[str, Any]]:
    """Which windows to draw.

    `spread` is the default and takes the worst, the median and the best in that order, so a page
    shows the range rather than a flattering sample of it. Picking at random is available and is
    honest, but on a set this size it mostly returns median windows and hides both tails.
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
