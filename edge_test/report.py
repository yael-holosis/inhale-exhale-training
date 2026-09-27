"""Turn what a device run left behind into figures and a table. Runs on the host, not there.

    poetry run python -m edge_test.report

Reads the three artifacts `run_inference.py` writes into this folder - `results.json`,
`timings.csv` and `predictions.npz` - and produces the two things the run exists to show:

1. **The latency distribution.** A histogram over the bulk of the measurements, the tail as one
   counted bar past a break, and what a window costs against its own duration. Every window is
   the same duration by construction, so a spread here is the device's scheduler, its thermal
   state and its other tenants - not the data.
2. **The predictions against the reference**, drawn window by window with `phase.figures`, the
   same drawing the training run makes. This is the check that the inference did its job, as
   opposed to running fast and returning nonsense.

Nothing is recomputed: the labels drawn are the device's own output.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402

try:                                                                    # noqa: E402
    from edge_test import run_inference as runner       # a checkout of the training repo
except ImportError:                                     # pragma: no cover - on a device
    import run_inference as runner                      # beside this file
from phase import figures                                               # noqa: E402
from phase.figures import GRID, INK, INK_SOFT, SURFACE                   # noqa: E402

FOLDER = Path(__file__).absolute().parent
INCOMING = FOLDER / runner.OUT_DIR
"""A run copied off a device lands in `edge_test/out/`, the same place a run made here writes to -
one folder holds the payload, the run and the report. `pack.py` clears `out/` when it re-packs, so
nothing from a past run is ever shipped to a device."""

REPORT_DIR = "report"
"""Figures and tables go in a subdirectory of the run they describe, one directory per run."""
PANELS_PER_PAGE = 5
PERCENTILES = (50, 90, 95, 99)
TICKS = (5, 7, 10, 15, 20, 30, 50, 75, 100, 200, 500)
BINS = 40
"""A ceiling. The count itself follows the sample - 40 bins over 30 measurements is a comb of
bars one high, which says nothing about a shape."""
TAIL_FENCE = 3.0
"""Where the tail starts, in multiples of the bulk's own spread past p95. A fixed multiple of p95
does not work: on a device whose measurements sit inside 2 ms, a single straggler at 1.2 x p95 is
under the fence and stretches the bins across the empty ms between it and the bulk. Measuring the
fence in p95-minus-p50 makes it scale with the distribution instead of with its position."""
BAR, TAIL = "#4C9BE8", "#9aa0a6"

SUMMARY_NAME = "latency_summary.csv"
LATENCY_NAME = "latency.png"
PAGE_NAME = "predictions_page_{page}.png"
INDEX_NAME = "index.csv"
"""One row per run, in `report/` itself. A run lands in its own stamped directory, so nothing is
overwritten and nothing tells you what else is there - this does."""

INDEX_COLUMNS = ("finished", "machine", "torch", "threads", "runs", "windows", "seconds",
                 "net_p50_ms", "net_p95_ms", "net_max_ms", "macro_f1_mean",
                 "passed", "checkpoint")
UNSTAMPED = "unstamped"
"""A run from before `run_inference.py` stamped its results. One such directory can exist; a
second would overwrite it, which is the behaviour the stamp exists to end."""
NET, TOTAL = "forward_ms", "total_ms"
"""`forward_ms` is the network alone - normalisation and the forward pass. `total_ms` adds the
decoder. **Everything reported here is the net**; the decoder is a separate concern with its own
fix, and mixing them invites reading one number as the other. `timings.csv` keeps all three
columns - it is the raw data and loses nothing."""

SUMMARY_COLUMNS = ("window_id", "env", "patient", "n", "net_mean_ms", "net_sd_ms", "net_min_ms",
                   "net_p50_ms", "net_p95_ms", "net_max_ms")


def read_timings(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("forward_ms", "decode_ms", "total_ms", "seconds"):
            row[key] = float(row[key])
        for key in ("run", "samples"):
            row[key] = int(row[key])
        row["window_id"] = int(row["window_id"])
    return rows


def read_predictions(path: Path) -> list[dict]:
    """Ragged like a shard: one item per window, with the device's own decoded labels."""
    with np.load(path, allow_pickle=False) as stored:
        offsets = stored["offsets"]
        return [{"values": stored["values"][offsets[i]:offsets[i + 1]],
                 "reference": stored["targets"][offsets[i]:offsets[i + 1]].astype(np.int64),
                 "prediction": stored["prediction"][offsets[i]:offsets[i + 1]].astype(np.int64),
                 "window_id": int(stored["window_id"][i]),
                 "patient": str(stored["patient"][i]),
                 "env": str(stored["env"][i]),
                 "fps": float(stored["fps"][i])}
                for i in range(len(offsets) - 1)]


def stats_of(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    out = {"n": int(array.size), "mean": float(array.mean()),
           "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
           "min": float(array.min()), "max": float(array.max())}
    for percentile, value in zip(PERCENTILES, np.percentile(array, PERCENTILES)):
        out[f"p{percentile}"] = float(value)
    return out


def by_window(timings: list[dict], series: str = NET) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for row in timings:
        out.setdefault(row["window_id"], []).append(row[series])
    return out


def _row(window_id, env: str, patient: str, net: list[float]) -> dict:
    stats = stats_of(net)
    return {"window_id": window_id, "env": env, "patient": patient, "n": stats["n"],
            "net_mean_ms": round(stats["mean"], 3), "net_sd_ms": round(stats["sd"], 3),
            "net_min_ms": round(stats["min"], 3), "net_p50_ms": round(stats["p50"], 3),
            "net_p95_ms": round(stats["p95"], 3), "net_max_ms": round(stats["max"], 3)}


def write_summary(out_dir: Path, timings: list[dict]) -> Path:
    """Per window, plus an `all` row. The distribution itself stays in `timings.csv`."""
    where = {row["window_id"]: (row["env"], row["patient"]) for row in timings}
    net = by_window(timings, NET)
    path = out_dir / SUMMARY_NAME
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for window_id in sorted(net):
            env, patient = where[window_id]
            writer.writerow(_row(window_id, env, patient, net[window_id]))
        writer.writerow(_row("all", "", "", [row[NET] for row in timings]))
    return path


def _log_x(ax, low: float, high: float) -> None:
    """A log x-axis with plain numbers on it. Matplotlib's own minor formatter puts `6 x 10^0`
    between the ticks, which is unreadable next to `7` and `10`."""
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([tick for tick in TICKS if low <= tick <= high])


def _ms(value: float) -> str:
    """Milliseconds at a precision the value can carry. A device runs tens of ms and a host runs
    single digits; `.0f` turns 6.8 and 7.4 into the same `7`."""
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def _bare(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)


def latency_figure(out_dir: Path, timings: list[dict], results: dict, seconds: float,
                   series: str = NET, stride: float | None = None,
                   signal: float | None = None) -> Path:
    """One histogram: how long the network takes on one window, and what that buys.

    **The net, and only the net.** The decoder is a separate concern with a separate fix, and a
    figure that carries both invites reading one number as the other. Its timings are still in
    `timings.csv`, which is the raw data and loses nothing. Per-window medians are a table -
    `latency_summary.csv` carries them.

    **The bins cover the bulk, not the range.** A device's tail runs to several times its own
    median off a handful of measurements, and binning across it collapses everything else into
    one bar. Past `TAIL_FENCE` spreads above p95 the measurements become a single counted bar
    past an axis break, so the shape of the bulk survives and the tail is still on the page.
    """
    values = np.array([row[series] for row in timings], dtype=np.float64)
    stats = stats_of(values)
    env = results["environment"]
    budget = seconds * 1e3
    subject = "net forward" if series == NET else "net forward + decode"

    # A floor under the spread, or a distribution tight enough that p50 and p95 coincide fences
    # off its own bulk.
    spread = max(stats["p95"] - stats["p50"], stats["p50"] * 0.02)
    cut = stats["p95"] + TAIL_FENCE * spread
    over, inside = values[values > cut], values[values <= cut]
    # The edges span the bulk, not up to `cut`: reaching to the cut empties the axis wherever the
    # tail starts further out than the bulk ends, which is the thing being fixed.
    edges = np.linspace(inside.min(), inside.max(),
                        int(np.clip(round(2.0 * inside.size ** 0.5), 8, BINS)) + 1)
    width = edges[1] - edges[0]

    fig = plt.figure(figsize=(9.5, 5.2), facecolor=SURFACE)
    ax = fig.add_axes((0.075, 0.30, 0.905, 0.535))
    strip = fig.add_axes((0.075, 0.10, 0.905, 0.055))
    _bare(ax)
    _bare(strip)

    counts, _, _ = ax.hist(inside, bins=edges, color=BAR, edgecolor=SURFACE, linewidth=0.6)
    top = counts.max()

    gap, over_w = width * 3.0, width * 2.2
    over_x = edges[-1] + gap
    if over.size:
        ax.bar(over_x, over.size, width=over_w, align="edge", color=SURFACE,
               edgecolor=TAIL, linewidth=1.2, hatch="///")
        ax.annotate(f"{over.size} above\n{_ms(cut)} ms\nmax {_ms(over.max())}",
                    xy=(over_x + over_w / 2, over.size), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK_SOFT, linespacing=1.45)
        for x in (edges[-1] + gap * 0.42, edges[-1] + gap * 0.58):
            ax.plot([x, x], [-top * 0.05, top * 0.05], color=INK_SOFT, linewidth=1.0,
                    alpha=0.5, solid_capstyle="butt", clip_on=False, zorder=3)

    # Every measurement as a tick in a band of its own under the bars, so the sparse far points
    # read as the handful of measurements they are rather than as a second mode.
    rug_y = -top * 0.085
    ax.plot(inside, np.full(inside.size, rug_y), marker="|", markersize=6, linestyle="none",
            color=INK_SOFT, alpha=0.5, clip_on=False)
    if over.size:
        ax.plot(np.linspace(over_x + over_w * 0.25, over_x + over_w * 0.75, over.size),
                np.full(over.size, rug_y), marker="|", markersize=6, linestyle="none",
                color=TAIL, clip_on=False)

    # Markers on a rail above the bars, not two full-height rules: a solid and a dashed black
    # line across the whole panel read as chart furniture, and which one was which needed a
    # legend in the corner. The caret carries the position, the label beside it carries the name.
    ax.set_ylim(-top * 0.15, top * 1.30)
    ax.set_xlim(edges[0] - width,
                (over_x + over_w + width * 2.5) if over.size else edges[-1] + width)
    left, right = ax.get_xlim()
    rail = top * 1.14
    close = abs(stats["p95"] - stats["p50"]) < (edges[-1] - edges[0]) * 0.22
    for name, label, ha in (("p50", "median", "right" if close else "center"),
                            ("p95", "p95", "left" if close else "center")):
        # A centred label runs off the panel when its marker sits against an edge, which is where
        # the median lands whenever the bulk is one narrow mode.
        margin = (right - left) * 0.12
        if stats[name] - left < margin:
            ha = "left"
        elif right - stats[name] < margin:
            ha = "right"
        ax.plot([stats[name]] * 2, [0, rail], color=INK_SOFT, linewidth=0.8, alpha=0.3, zorder=1)
        ax.plot([stats[name]], [rail], marker="v", markersize=7, color=INK, clip_on=False,
                zorder=5)
        ax.annotate(f"{label} {_ms(stats[name])} ms", xy=(stats[name], rail),
                    xytext=({"right": -8, "left": 8, "center": 0}[ha], 7),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=8, color=INK)

    ax.set_xlabel(f"ms per {seconds:g} s window", fontsize=9, color=INK_SOFT)
    ax.set_ylabel("measurements", fontsize=9, color=INK_SOFT)
    # The panel keeps room under zero for the rug; a tick there would read as a negative count.
    ticks = matplotlib.ticker.MaxNLocator(integer=True).tick_values(0, top)
    ax.set_yticks([tick for tick in ticks if 0 <= tick <= top])
    ax.set_xticks([tick for tick in ax.get_xticks() if edges[0] - width <= tick <= edges[-1]])

    # What the number buys: one window against its own duration. Log, because the forward pass,
    # the forward pass with the decoder and the window itself are decades apart.
    strip.set_xscale("log")
    strip.set_xlim(stats["p50"] * 0.5, budget * 2.2)
    strip.set_ylim(-0.5, 0.5)
    strip.set_yticks([])
    strip.set_xticks([])
    strip.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    strip.spines["left"].set_visible(False)
    strip.spines["bottom"].set_visible(False)
    strip.barh(0, stats["p50"], height=0.55, color=BAR, edgecolor=SURFACE)
    strip.axvline(budget, color=INK, linewidth=1.0)
    for x, y, text, offset, ha in (
            (stats["p50"], 0.0, f"net {_ms(stats['p50'])} ms", (6, 0), "left"),
            (budget, 0.0, f"one {seconds:g} s window", (-6, 0), "right")):
        strip.annotate(text, xy=(x, y), xytext=offset, textcoords="offset points", ha=ha,
                       va="bottom" if ha == "center" else "center", fontsize=8, color=INK_SOFT)
    median = stats["p50"]
    bottom = f"median {_ms(median)} ms per {seconds:g} s window"
    if stride and signal:
        windows = int((signal - seconds) // stride) + 1
        bottom += (f" · a window every {stride:g} s, so {windows} windows per {signal:g} s "
                   f"signal · median {windows * median / 1e3:.2f} s per signal")
    elif stride:
        bottom += f" · a window every {stride:g} s, so {stride * 1e3 / median:.0f}x real time"
    else:
        bottom += f" · {budget / median:.0f}x real time on non-overlapping windows"
    strip.set_xlabel(bottom, fontsize=8.5, color=INK_SOFT)

    fig.suptitle(f"{subject} · {results['runs']} runs x "
                 f"{len(by_window(timings, series))} windows of {seconds:g} s · "
                 f"{env['machine']}, torch {env['torch']}, {env['threads']} threads",
                 fontsize=10.5, color=INK, x=0.014, ha="left", y=0.975)
    fig.text(0.014, 0.905,
             f"{stats['n']} measurements · sd {_ms(stats['sd'])} · max {_ms(stats['max'])} ms",
             fontsize=8.5, color=INK_SOFT, ha="left")
    path = out_dir / LATENCY_NAME
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def prediction_pages(out_dir: Path, items: list[dict], results: dict, plot_cfg,
                     reference_name: str) -> list[Path]:
    """The device's own labels against the reference, `phase.figures` doing the drawing."""
    panels = []
    for item in items:
        score = figures.score(item["prediction"], item["reference"])
        panels.append({**item, "score": score,
                       "title": (f"{item['patient']} · window {item['window_id']} · "
                                 f"{item['env']} · {figures.SCORE_LABEL} {score:.2f}")})
    panels.sort(key=lambda panel: panel["score"])

    pages = []
    for index in range(0, len(panels), PANELS_PER_PAGE):
        page = index // PANELS_PER_PAGE + 1
        pages.append(figures.plot_windows(
            panels[index:index + PANELS_PER_PAGE], out_dir / PAGE_NAME.format(page=page),
            f"device predictions · page {page} · worst first · "
            f"{results['environment']['machine']}, torch {results['environment']['torch']}",
            reference_name, plot_cfg, prediction_name="device"))
    return pages


def write_index(report_dir: Path, results: dict, manifest: dict, timings: list[dict],
                seconds: float, run: str) -> Path:
    """Append this run to `report/index.csv`, replacing a row with the same stamp.

    Rewritten whole rather than appended to, so re-running the report on the same device output
    updates its row instead of adding a duplicate.
    """
    path = report_dir / INDEX_NAME
    rows = {}
    if path.exists():
        with path.open() as handle:
            # Narrowed to the current columns: an index written before a column was added or
            # dropped still reads, instead of failing the run that was going to rewrite it.
            rows = {row["finished"]: {key: row.get(key, "") for key in INDEX_COLUMNS}
                    for row in csv.DictReader(handle)}
    env = results["environment"]
    net = stats_of([row[NET] for row in timings])
    rows[run] = {"finished": run, "machine": env["machine"], "torch": env["torch"],
                 "threads": env["threads"], "runs": results["runs"],
                 "windows": len(results["windows"]), "seconds": f"{seconds:g}",
                 "net_p50_ms": round(net["p50"], 3), "net_p95_ms": round(net["p95"], 3),
                 "net_max_ms": round(net["max"], 3),
                 "macro_f1_mean": round(results["macro_f1"]["mean"], 3),
                 "passed": results["passed"], "checkpoint": manifest["checkpoint"]}
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    return path


def source(chosen: Path | None) -> Path | None:
    """Where the run to report on is. `--from` if given, else whichever of the staging directory
    and the payload folder actually holds a complete run - the payload one being a dry run of the
    whole thing on this machine."""
    candidates = [chosen] if chosen else [INCOMING, FOLDER]
    for candidate in candidates:
        if all((candidate / name).exists() for name in runner.RETURN_FILES):
            return candidate
    return None


def build(folder: Path, out: Path | None = None, series: str = NET,
          stride: float | None = None,
          signal: float | None = None) -> tuple[dict, list[Path]]:
    """Everything a report is, for one run: the figures, the table, and a copy of the run itself.

    Called by `run_inference.py` the moment a run finishes - so the plot exists on the machine that
    produced the numbers - and by `main` when a run is drawn somewhere else.
    """
    results = json.loads((folder / runner.RESULTS).read_text())
    manifest = json.loads((folder / runner.MANIFEST).read_text())
    timings = read_timings(folder / runner.TIMINGS)
    items = read_predictions(folder / runner.PREDICTIONS)

    # One directory per run, named by when the run finished, so a second run cannot land on the
    # first one's figures.
    run = str(results.get("finished") or UNSTAMPED)
    # A report directory already holds a copy of the run it describes, so drawing one again from
    # inside it must land there, not in a report inside a report.
    inside = folder.parent.name == REPORT_DIR
    report_dir = folder.parent if inside else folder / REPORT_DIR
    out_dir = out or (folder if inside else report_dir / run)
    out_dir.mkdir(parents=True, exist_ok=True)

    # From the manifest, not from `parameter/`: the figures are drawn where the run happened and
    # Hydra is not on a device.
    plot_cfg = manifest.get("plot") or {}
    seconds = float(results.get("window_seconds") or timings[0]["seconds"])
    stride = stride if stride is not None else manifest.get("window_stride_seconds")
    signal = signal if signal is not None else manifest.get("signal_seconds")
    if stride:
        print(f"  a window every {stride:g} s"
              + (f" over a {signal:g} s signal" if signal else ""))

    kept = [shutil.copy2(folder / name, out_dir / name) for name in runner.RETURN_FILES
            if (folder / name).absolute() != (out_dir / name).absolute()]
    written = [write_summary(out_dir, timings),
               latency_figure(out_dir, timings, results, seconds, series, stride, signal),
               *prediction_pages(out_dir, items, results, plot_cfg,
                                 "algorithm" if manifest["label_source"] == "algorithm"
                                 else "labeller"),
               *kept,
               write_index(report_dir, results, manifest, timings, seconds, run)]

    print(f"\nfigures for run {run} -> {out_dir}")
    for path in written:
        print(f"  {path.name}")
    return results, written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", type=Path, default=None,
                        help=f"the run to draw: a directory holding "
                             f"{', '.join(runner.RETURN_FILES)}. Default is "
                             f"{runner.OUT_DIR}/ beside this file")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"where the figures go (default: the run's own "
                             f"{REPORT_DIR}/<its stamp>/)")
    parser.add_argument("--stride", type=float, default=None,
                        help="seconds between consecutive windows in production, for the cost of "
                             "a whole recording. Default is what pack.py measured into the "
                             "manifest")
    parser.add_argument("--signal", type=float, default=None,
                        help="how long one radar signal runs. Default is what pack.py measured "
                             "into the manifest")
    args = parser.parse_args()

    folder = source(args.source.absolute() if args.source else None)
    if folder is None:
        print(f"no complete run found in {INCOMING}. Fetch the device's {runner.OUT_DIR}/ "
              f"directory, from the repo root:\n"
              f"  {runner.fetch_command('<the folder on the device>')}")
        return 2
    print(f"reading the run from {folder}")

    results, _ = build(folder, args.out, NET, args.stride,
                       args.signal)
    timings = read_timings(folder / runner.TIMINGS)
    manifest = json.loads((folder / runner.MANIFEST).read_text())
    env = results["environment"]
    net = stats_of([row[NET] for row in timings])
    seconds = float(results.get("window_seconds") or timings[0]["seconds"])
    stride = args.stride if args.stride is not None else manifest.get("window_stride_seconds")
    signal = args.signal if args.signal is not None else manifest.get("signal_seconds")

    print(f"\n  {manifest['checkpoint']}\n  {manifest['dataset']}, fold {manifest['fold']}, "
          f"{manifest['split']} split")
    print(f"  device: {env['machine']}, python {env['python']}, torch {env['torch']}, "
          f"{env['threads']} threads")
    print(f"  {results['runs']} runs x {len(results['windows'])} windows of {seconds:g} s = "
          f"{net['n']} measurements\n")
    print(f"    net forward  mean {net['mean']:6.2f}  sd {net['sd']:5.2f}  "
          f"min {net['min']:6.2f}  p50 {net['p50']:6.2f}  p95 {net['p95']:6.2f}  "
          f"p99 {net['p99']:6.2f}  max {net['max']:6.2f}")
    print(f"  {seconds:g} s of signal per {net['p50']:.2f} ms - "
          f"{seconds * 1e3 / net['p50']:.0f}x real time")
    print(f"  macro F1 mean {results['macro_f1']['mean']:.2f}, "
          f"range {results['macro_f1']['min']:.2f}-{results['macro_f1']['max']:.2f}")
    print(f"  largest logit difference from the host {results['max_logit_drift']:.2e} "
          f"({'PASS' if results['passed'] else 'FAIL'})")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
