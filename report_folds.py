"""Aggregate the folds on the one test set they share.

    poetry run python report_folds.py --runs outputs/human_folds

Every fold holds out the same windows, so the five models are directly comparable and their
logits can be combined. Three things come out, and they answer different questions:

`per_fold.csv` - each fold's own test metrics, with mean and spread. The spread measures
    **sensitivity to the train/val partition**, not generalisation: the folds share one test set
    and their training sets overlap heavily, so it is not a confidence interval and is not
    labelled as one.

`ensemble` - the five models' **logits** averaged per sample, then decoded once. This is the
    single number to quote. Averaging logits rather than decoded labels matters: a majority vote
    over labels throws away the transition matrix and the minimum durations that the decoder is
    there to enforce.

`bootstrap` - the ensemble's score recomputed over test **patients** resampled with replacement.
    With a test set this small the dominant uncertainty is which people are in it, which the
    fold spread cannot see. This is the interval worth reading.

A mean of five macro F1s is deliberately not reported as the model's F1 - macro F1 is not linear
in samples, so the mean of five is the F1 of nothing.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from models.lightning_module import allowed_for
from phase import durations, figures
from phase.building import load_windows, shard_path
from phase.decode import decode, transition_matrix
from phase.labels import N_CLASSES, PHASES, UNKNOWN
from phase.labelsources import ALGORITHM
from phase.metrics import (duration_agreement, duration_pairs, event_level,
                           per_sample)
from phase.splits import STUDY_COLUMN, study_of

LOGITS_NAME = "test_logits.npz"
METRICS_NAME = "test_metrics.csv"
HEADLINE = "macro_f1"


def load_fold(run_dir: Path) -> dict:
    with np.load(run_dir / LOGITS_NAME, allow_pickle=False) as stored:
        return {"logits": stored["logits"], "targets": stored["targets"].astype(np.int64),
                "offsets": stored["offsets"], "window_id": stored["window_id"],
                "patient": stored["patient"], "env": stored["env"], "fps": stored["fps"],
                "fold": run_dir.name}


def windows_of(fold: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    offsets = fold["offsets"]
    return [(fold["logits"][offsets[i]:offsets[i + 1]],
             fold["targets"][offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)]


def score_windows(windows, cost, min_duration, fps: float, event_iou: float) -> dict:
    """Per-sample metrics over the pooled samples, event metrics averaged per window."""
    decoded = [(decode(logits, cost, min_duration), truth) for logits, truth in windows]
    metrics = per_sample(np.concatenate([p for p, _ in decoded]),
                         np.concatenate([t for _, t in decoded]))
    events: dict[str, list[float]] = {}
    for prediction, truth in decoded:
        for key, value in event_level(prediction, truth, event_iou).items():
            events.setdefault(key, []).append(value)
    for key, values in events.items():
        clean = [v for v in values if not np.isnan(v)]
        if clean:
            metrics[key] = float(np.mean(clean))
    if "boundary_mae_samples" in metrics:
        metrics["boundary_mae_sec"] = metrics["boundary_mae_samples"] / fps
    return metrics


CALLED = tuple(PHASES)
"""Every class, `unknown` included - it is 29% of the samples and where the mistakes are."""


def weighted_dice(metrics: dict, truth: np.ndarray) -> float:
    """Per-class Dice weighted by that class's support in this patient's own labels.

    Dice and F1 are the same quantity per class, so these are the `f1_*` values already computed
    - weighted by support instead of averaged flat. `unknown` is in the weighting.
    """
    support = {name: float(np.sum(truth == PHASES.index(name))) for name in CALLED}
    total = sum(support.values())
    if not total:
        return float("nan")
    return sum(metrics[f"f1_{name}"] * support[name] for name in CALLED) / total


def per_patient(windows, patients: np.ndarray, cost, min_duration, fps: float,
                event_iou: float) -> pd.DataFrame:
    """One row per test patient. Which people it fails on is not visible in a pooled score."""
    rows = []
    for name in sorted(set(patients.tolist())):
        picked = [windows[i] for i in np.flatnonzero(patients == name)]
        metrics = score_windows(picked, cost, min_duration, fps, event_iou)
        truth = np.concatenate([truth for _, truth in picked])
        rows.append({"patient": name, STUDY_COLUMN: study_of(name),
                     "windows": len(picked), "samples": int(truth.size),
                     "weighted_dice": weighted_dice(metrics, truth),
                     "macro_f1": metrics["macro_f1"],
                     **{f"f1_{n}": metrics[f"f1_{n}"] for n in CALLED}})
    return pd.DataFrame(rows).sort_values("weighted_dice").reset_index(drop=True)


def fold_scaling(folds, cost, min_duration, fps: float, event_iou: float) -> pd.DataFrame:
    """What ensembling k of the folds is worth, for every k.

    Every subset of size k is scored, not the first k: with five folds an "ensemble of two" that
    happened to pick the two best would read as a gain that ordering alone produced. The spread
    across subsets is the honest error bar on "use k folds".
    """
    rows = []
    for k in range(1, len(folds) + 1):
        for subset in combinations(range(len(folds)), k):
            logits = np.stack([folds[i]["logits"] for i in subset]).mean(axis=0)
            pooled = {**folds[0], "logits": logits}
            scored = score_windows(windows_of(pooled), cost, min_duration, fps, event_iou)
            rows.append({"folds_used": k,
                         "which": "+".join(str(i) for i in subset),
                         HEADLINE: scored[HEADLINE],
                         "accuracy": scored["accuracy"]})
    return pd.DataFrame(rows)


DATASET_NAME = "dataset.txt"
PAGES_DIR = "test_windows"


WINDOW_KEYS = ["env", "RespirationWindowID"]


def dataset_of(runs: Path) -> Path | None:
    """The dataset a run was trained on, off the `dataset.txt` written beside every fold."""
    named = next((d / DATASET_NAME for d in sorted(runs.iterdir())
                  if (d / DATASET_NAME).exists()), None)
    if named is None:
        return None
    dataset = Path(named.read_text().strip())
    if not dataset.is_absolute() and not dataset.exists():
        # `dataset.txt` records the path as the run saw it, which is relative to the repo root.
        # A caller running from anywhere else - a notebook, say - resolves it against its own
        # directory and finds nothing, which reads as a deleted dataset rather than a bad path.
        dataset = Path(__file__).resolve().parent / dataset
    return dataset if dataset.exists() else None


def provenance_for(runs: Path, window_ids: np.ndarray,
                   envs: np.ndarray) -> dict[tuple[str, int], dict]:
    """Where each test window came from, keyed by `(env, window id)`.

    Keyed by the pair because the id alone is not unique - the two instances have separate id
    spaces and 13 of 360 windows collide on the current set. The same holds for the signal id,
    which is why anything grouped by signal has to carry `env` with it.
    """
    dataset = dataset_of(runs)
    if dataset is None:
        print(f"no reachable dataset named under {runs} - no provenance for the test windows")
        return {}
    manifest = load_windows(dataset).set_index(WINDOW_KEYS)
    out = {}
    for env, window_id in zip(envs.tolist(), window_ids.tolist()):
        key = (str(env), int(window_id))
        if key not in manifest.index:
            continue
        row = manifest.loc[key]
        out[key] = {"shard": str(row["shard"]), "position": int(row["position"]),
                    "signal": int(row["RadarSignalID"]),
                    "session": int(row["SessionID"]),
                    "window_index": int(row["WindowIndex"])}
    return out


def traces_for(runs: Path, window_ids: np.ndarray,
               envs: np.ndarray) -> dict[tuple[str, int], dict]:
    """`provenance_for` with the stored waveform attached, for the panels."""
    dataset = dataset_of(runs)
    found = provenance_for(runs, window_ids, envs)
    if dataset is None or not found:
        return {}
    out = {}
    for key, row in found.items():
        with np.load(shard_path(dataset, row["shard"]), allow_pickle=False) as stored:
            start, end = (stored["offsets"][row["position"]],
                          stored["offsets"][row["position"] + 1])
            out[key] = {**row, "values": stored["values"][start:end].astype(np.float32)}
    return out


def draw_every_window(runs: Path, out_dir: Path, ensemble: dict, windows, cost, min_duration,
                      plot_cfg, per_page: int = 8, shading: str = "ensemble") -> list[Path]:
    """Every test window with the ensemble's answer, paginated.

    All of them, not a spread: a page of six says what the tails look like, and this says how
    often each tail happens. Paginated because sixty-odd panels on one canvas is unreadable at
    any scale that shows a breath.
    """
    traces = traces_for(runs, ensemble["window_id"], ensemble["env"])
    if not traces:
        return []

    items = []
    for index, window_id in enumerate(ensemble["window_id"].tolist()):
        found = traces.get((str(ensemble["env"][index]), int(window_id)))
        if found is None:
            continue
        logits, truth = windows[index]
        prediction = decode(logits, cost, min_duration)
        items.append({"values": found["values"], "reference": truth, "prediction": prediction,
                      "fps": float(ensemble["fps"][index]),
                      "title": (f"{ensemble['patient'][index]} · {ensemble['env'][index]} · "
                                f"signal {found['signal']} · session {found['session']} · "
                                f"window {int(window_id)} (index {found['window_index']}) · "
                                f"macro F1 {figures.score(prediction, truth):.2f}")})

    pages_dir = Path(out_dir) / PAGES_DIR
    written = []
    total = (len(items) + per_page - 1) // per_page
    for page in range(total):
        chunk = items[page * per_page:(page + 1) * per_page]
        written.append(figures.plot_windows(
            chunk, pages_dir / f"page_{page + 1:02d}.png",
            f"test set, ensemble {shading} · page {page + 1} of {total} · "
            f"windows {page * per_page + 1}-{page * per_page + len(chunk)} of {len(items)}",
            "labeller", plot_cfg, prediction_name=shading,
            caption=f"shading = {shading} · ribbon = labeller"))
    return written


def confusion(windows, cost, min_duration) -> np.ndarray:
    """Row-normalised, so a class that is 29% of the samples cannot dominate every row."""
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=float)
    for logits, truth in windows:
        prediction = decode(logits, cost, min_duration)
        for true_class, predicted in zip(truth, prediction):
            matrix[int(true_class), int(predicted)] += 1
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def bootstrap(windows, patients: np.ndarray, cost, min_duration, fps: float, event_iou: float,
              draws: int, seed: int) -> tuple[float, float]:
    """Resample test *patients*, not windows - windows from one person are not independent."""
    rng = np.random.default_rng(seed)
    unique = np.unique(patients)
    scores = []
    for _ in range(draws):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        picked = [w for name in drawn
                  for w in [windows[i] for i in np.flatnonzero(patients == name)]]
        if picked:
            scores.append(score_windows(picked, cost, min_duration, fps, event_iou)[HEADLINE])
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))) if scores \
        else (float("nan"), float("nan"))


RAW = "raw"
DECODED = "viterbi"


HYDRA_CONFIG = Path(".hydra") / "config.yaml"


def config_of(runs: Path):
    """The config a run was executed under, off its own Hydra output. None if it has none.

    **Never read `parameter/` to describe a finished run.** That tree is what the next run will
    use, and it moves - a decoding penalty, a transition table or a correction changed since the
    run would silently rescore it under settings it never saw. `OmegaConf.load` does not resolve
    Hydra's `defaults:` either, so the merged tree has no `data` at all.
    """
    for candidate in (runs / HYDRA_CONFIG, *(d / HYDRA_CONFIG for d in sorted(runs.iterdir())
                                             if d.is_dir())):
        if candidate.is_file():
            return OmegaConf.load(candidate)
    return None


def decoding_of(runs: Path) -> dict:
    """`training.decoding` as the run had it - penalty, floors and the transition tables."""
    cfg = config_of(runs)
    found = None if cfg is None else OmegaConf.select(cfg, "training.decoding")
    if found is None:
        raise FileNotFoundError(
            f"no .hydra config under {runs} - its decoding settings are not recoverable, and "
            f"reading them from parameter/ would score it under settings it never saw")
    return OmegaConf.to_container(found, resolve=True)


def label_source_of(runs: Path) -> str:
    """The label source the run actually used, off its own Hydra config."""
    cfg = config_of(runs)
    found = None if cfg is None else OmegaConf.select(cfg, "data.labels.source")
    if found:
        return str(found)
    print(f"no .hydra config under {runs} - assuming labels.source={ALGORITHM}")
    return ALGORITHM


def load_run(runs: Path) -> list[dict] | None:
    """Every fold under `runs`, checked for the shared test set that makes them comparable."""
    folds = [load_fold(d) for d in sorted(runs.iterdir())
             if d.is_dir() and (d / LOGITS_NAME).exists()]
    if not folds:
        print(f"no fold under {runs} carries {LOGITS_NAME}")
        return None
    reference = folds[0]
    for fold in folds[1:]:
        if not np.array_equal(fold["window_id"], reference["window_id"]):
            raise SystemExit(
                f"{fold['fold']} was tested on different windows than {reference['fold']} - "
                f"these folds are not comparable and must not be pooled")
    print(f"{len(folds)} folds, {len(reference['window_id'])} test windows each, "
          f"{len(np.unique(reference['patient']))} patients - window ids identical")
    return folds


def settings_of(runs: Path, folds: list[dict], label_source: str | None,
                enforce_min: bool | None) -> dict:
    """How to decode and score this run - **its own** settings, not `parameter/`.

    That tree is what the next run will use and it moves. Reading it here re-scores a finished
    run under a penalty, a transition table or a duration floor it never saw, and nothing in the
    report says the change happened.
    """
    cfg = OmegaConf.load("parameter/config.yaml")
    run_cfg = config_of(runs)
    if run_cfg is not None and OmegaConf.select(run_cfg, "training.decoding") is not None:
        decoding = OmegaConf.to_container(run_cfg.training.decoding, resolve=True)
        event_iou = float(run_cfg.training.event_iou)
    else:
        print(f"no .hydra config under {runs} - falling back to parameter/, which may have moved "
              f"since the run")
        training = OmegaConf.load("parameter/training/default.yaml")
        decoding = OmegaConf.to_container(training.decoding, resolve=True)
        event_iou = float(training.event_iou)
    if enforce_min is not None and bool(enforce_min) != bool(decoding["enforce_min"]):
        print(f"OVERRIDE: enforce_min {decoding['enforce_min']} -> {bool(enforce_min)} "
              f"- this is an ablation, not what the run decoded with")
        decoding["enforce_min"] = bool(enforce_min)
    source = label_source or label_source_of(runs)
    print(f"labels.source = {source}, switch_penalty = {decoding['switch_penalty']}, "
          f"enforce_min = {decoding['enforce_min']}, "
          f"transition table = {sorted(allowed_for(decoding, source))}")
    cost = (transition_matrix(allowed_for(decoding, source), decoding["switch_penalty"])
            if decoding["viterbi"] else None)
    return {"cost": cost,
            "min_duration": decoding["min_duration"] if decoding["enforce_min"] else None,
            "event_iou": event_iou, "fps": float(np.median(folds[0]["fps"])),
            "plot_cfg": OmegaConf.to_container(cfg.plot, resolve=True), "seed": int(cfg.seed)}


def ensemble_of(folds: list[dict]) -> tuple[dict, list]:
    """The folds' **logits** averaged per sample, and the per-window view of them.

    Logits, not decoded labels: a majority vote over labels throws away the transition matrix
    and the duration floors that the decoder exists to enforce.
    """
    stacked = np.stack([fold["logits"] for fold in folds])
    ensemble = {**folds[0], "logits": stacked.mean(axis=0), "fold": "ensemble"}
    return ensemble, windows_of(ensemble)


def passes_of(settings: dict) -> tuple[tuple[str, tuple], ...]:
    """The two scorings every report runs: the network alone, and the network plus the decoder."""
    return ((RAW, (None, None)),
            (DECODED, (settings["cost"], settings["min_duration"])))


def aggregate(runs: Path, out_dir: Path | None = None, draws: int = 2000,
              per_page: int = 8, label_source: str | None = None,
              enforce_min: bool | None = None) -> dict | None:
    """Pool every fold under `runs`, scored **twice** - once on the network's own argmax and once
    through the decoder - so what the post-processing is worth is visible rather than assumed.

    The decoder is post-processing, nothing more: the same saved logits are scored both ways, and
    the network is identical in each. Returns the decoded metrics, which are the ones to quote.
    """
    runs = Path(runs)
    out_dir = Path(out_dir) if out_dir else runs
    folds = load_run(runs)
    if not folds:
        return None
    settings = settings_of(runs, folds, label_source, enforce_min)

    both = {}
    for label, (this_cost, this_min) in passes_of(settings):
        both[label] = _report_one(label, out_dir / label, folds, this_cost, this_min,
                                  settings["fps"], settings["event_iou"], draws,
                                  settings["seed"], settings["plot_cfg"], runs, per_page)

    _compare(both, out_dir, settings["plot_cfg"])
    return both[DECODED]


def durations_only(runs: Path, out_dir: Path | None = None, label_source: str | None = None,
                   enforce_min: bool | None = None) -> bool:
    """Just the duration reports, for both passes - no bootstrap, no subsets, no panels.

    The point is re-reporting a finished run after the duration metric changes, on a machine
    that is probably training something else: the expensive parts of `aggregate` say nothing
    about durations and there is no reason to pay for them again.
    """
    runs = Path(runs)
    out_dir = Path(out_dir) if out_dir else runs
    folds = load_run(runs)
    if not folds:
        return False
    settings = settings_of(runs, folds, label_source, enforce_min)
    ensemble, windows = ensemble_of(folds)
    for label, (this_cost, this_min) in passes_of(settings):
        target = out_dir / label
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 20} {label} {'=' * 20}")
        duration_reports(runs, target, ensemble, windows, this_cost, this_min,
                         settings["event_iou"], settings["plot_cfg"])
        print(f"  -> {target}/")
    return True


SPAN_ERRORS = "span_duration_errors"
SPAN_AGREEMENT = "span_duration_agreement"
SIGNAL_MEDIANS = "signal_duration_medians"
SIGNAL_AGREEMENT = "signal_duration_agreement"
SPAN_FIGURE = "span_durations"
SIGNAL_FIGURE = "signal_durations"
DURATION_TABLES = (SPAN_ERRORS, SPAN_AGREEMENT, SIGNAL_MEDIANS, SIGNAL_AGREEMENT)
DURATION_FIGURES = (SPAN_FIGURE, SIGNAL_FIGURE)

PASSES = (RAW, DECODED)
AGGREGATE_FIGURES = ("fold_report", "per_patient", "fold_scaling", *DURATION_FIGURES)
AGGREGATE_TABLES = ("per_fold", "ensemble_metrics", "per_patient", "fold_scaling",
                    *DURATION_TABLES)
"""What gets uploaded, named here rather than in `train.py` - a report that grows an output and an
uploader that lists them by hand drift apart silently, and the missing figure is only noticed when
somebody looks for it in ClearML.

`durations` and `duration_agreement` are written to disk and deliberately **not** uploaded. That
figure is the mean span length of a whole window, taken over the spans the window boundary cut, so
it carries the length of every fragment. `span_durations` and `signal_durations` measure the same
thing without them and split it into the per-breath and per-recording questions."""


def duration_items(runs: Path, ensemble: dict, windows, cost, min_duration) -> list[dict]:
    """One entry per test window: the two label arrays, and which recording it came from.

    The signal id is not in the saved logits - it comes off the dataset manifest - and a window
    without it is dropped from the per-signal view rather than being pooled under a made-up key.
    """
    found = provenance_for(runs, ensemble["window_id"], ensemble["env"])
    items = []
    for index, window_id in enumerate(ensemble["window_id"].tolist()):
        key = (str(ensemble["env"][index]), int(window_id))
        if key not in found:
            continue
        logits, truth = windows[index]
        items.append({durations.LABEL: truth,
                      durations.MODEL: decode(logits, cost, min_duration),
                      "fps": float(ensemble["fps"][index]),
                      "env": key[0], "window_id": key[1],
                      "signal": found[key]["signal"],
                      "patient": str(ensemble["patient"][index])})
    return items


def duration_reports(runs: Path, out_dir: Path, ensemble: dict, windows, cost, min_duration,
                     event_iou: float, plot_cfg) -> None:
    """Time spent in each phase, per breath and per recording - what the network is *for*.

    Separate from `duration_agreement` above, which compares the mean span length of a whole
    window. That mean is taken over spans the window boundary cut, so it carries the length of
    every fragment; these two exclude them and are the numbers to read.
    """
    items = duration_items(runs, ensemble, windows, cost, min_duration)
    if not items:
        print("no test window could be traced to a signal - skipping the duration reports")
        return

    pairs, coverage = durations.span_errors(items, event_iou)
    span_stats = durations.span_agreement(pairs, coverage)
    pairs.to_csv(out_dir / f"{SPAN_ERRORS}.csv", index=False)
    span_stats.to_csv(out_dir / f"{SPAN_AGREEMENT}.csv")
    figures.span_duration_report(pairs, span_stats, out_dir / f"{SPAN_FIGURE}.png", plot_cfg)
    print("\nper-span duration error, boundary-cut spans excluded:")
    print("\n".join(durations.describe(span_stats, "spans")))

    medians = durations.signal_medians(items)
    signal_stats = durations.signal_agreement(medians)
    medians.to_csv(out_dir / f"{SIGNAL_MEDIANS}.csv", index=False)
    signal_stats.to_csv(out_dir / f"{SIGNAL_AGREEMENT}.csv")
    figures.signal_duration_report(medians, signal_stats, out_dir / f"{SIGNAL_FIGURE}.png",
                                   plot_cfg)
    usable = int(medians[durations.USABLE].sum()) if len(medians) else 0
    print(f"per-signal median duration, {usable} of {len(medians)} signal-phase pairs carry "
          f"at least {durations.MIN_SPANS_PER_SIGNAL} spans on both sides:")
    print("\n".join(durations.describe(signal_stats, "signals")))


def _report_one(label: str, out_dir: Path, folds, cost, min_duration, fps: float,
                event_iou: float, draws: int, seed: int, plot_cfg, runs: Path,
                per_page: int) -> dict:
    """One complete report - `cost=None` is the network's raw argmax, no transitions, no floors."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- per fold
    # Recomputed from each fold's own logits rather than read out of its test_metrics.csv: that
    # csv was written by whatever the metric code said when the run happened, and comparing it
    # against an ensemble scored now would put two different definitions in one table.
    rows = []
    for fold in folds:
        scored = score_windows(windows_of(fold), cost, min_duration, fps, event_iou)
        rows.append({"fold": fold["fold"], **scored})
    per_fold = pd.DataFrame(rows).set_index("fold")
    per_fold.to_csv(out_dir / "per_fold.csv")

    # ---------------------------------------------------------------- ensemble
    ensemble, windows = ensemble_of(folds)
    metrics = score_windows(windows, cost, min_duration, fps, event_iou)

    patients = ensemble["patient"]
    low, high = bootstrap(windows, patients, cost, min_duration, fps, event_iou,
                          draws, seed)

    headline = per_fold[HEADLINE] if HEADLINE in per_fold else pd.Series(dtype=float)
    print(f"\n[{label}] per fold {HEADLINE}: "
          + "  ".join(f"{name.replace('fold_', '')}={value:.2f}"
                      for name, value in headline.items()))
    if len(headline) > 1:
        print(f"  mean {headline.mean():.2f}, spread {headline.std():.2f} "
              f"(partition sensitivity, NOT a generalisation interval)")
    print(f"\nensemble {HEADLINE}: {metrics[HEADLINE]:.2f}   "
          f"95% CI over test patients [{low:.2f}, {high:.2f}]")
    for key in sorted(metrics):
        print(f"  {key:28s} {metrics[key]:.2f}")

    summary = pd.DataFrame([{"metric": k, "ensemble": v} for k, v in sorted(metrics.items())])
    summary.to_csv(out_dir / "ensemble_metrics.csv", index=False)

    scaling = fold_scaling(folds, cost, min_duration, fps, event_iou)
    scaling.to_csv(out_dir / "fold_scaling.csv", index=False)
    summary = scaling.groupby("folds_used")[HEADLINE].agg(["mean", "min", "max", "count"])
    print(f"\nensembling k folds ({HEADLINE}):")
    for k, row in summary.iterrows():
        print(f"  k={k}  mean {row['mean']:.3f}  range {row['min']:.3f}-{row['max']:.3f}  "
              f"({int(row['count'])} subset{'s' if row['count'] > 1 else ''})")
    gain = summary.loc[len(folds), "mean"] - summary.loc[1, "mean"]
    print(f"  all {len(folds)} vs one fold: {gain:+.3f}")

    by_patient = per_patient(windows, patients, cost, min_duration, fps, event_iou)
    by_patient.to_csv(out_dir / "per_patient.csv", index=False)
    # A patient whose windows carry no called phase has nothing to weight. Kept in the csv and
    # left off the chart, rather than sorted to the end and read as the best score in the set.
    scored = by_patient.dropna(subset=["weighted_dice"])
    skipped = len(by_patient) - len(scored)
    if len(scored):
        print(f"\nweighted dice per patient: worst {scored.iloc[0]['patient']} "
              f"{scored.iloc[0]['weighted_dice']:.2f}, best {scored.iloc[-1]['patient']} "
              f"{scored.iloc[-1]['weighted_dice']:.2f}"
              + (f" ({skipped} patient(s) carry no called phase)" if skipped else ""))

    # ---------------------------------------------------------------- figures
    figures.fold_report(per_fold, metrics, (low, high),
                        confusion(windows, cost, min_duration),
                        out_dir / "fold_report.png", plot_cfg, headline=HEADLINE)
    figures.patient_report(scored, metrics, out_dir / "per_patient.png", plot_cfg)
    figures.scaling_report(scaling, out_dir / "fold_scaling.png", plot_cfg, headline=HEADLINE)

    # The durations are what the device reports, so agreement on them is the result rather than
    # a diagnostic. Per window, over the windows where both sides called the phase at all.
    pairs = [duration_pairs(decode(logits, cost, min_duration), truth, fps)
             for logits, truth in windows]
    agreement = duration_agreement(pairs)
    pd.DataFrame(agreement).T.to_csv(out_dir / "duration_agreement.csv")
    figures.duration_report(pairs, agreement, out_dir / "durations.png", plot_cfg)
    for name, stats in agreement.items():
        if stats.get("n"):
            print(f"  {name:8s} bias {stats['bias_sec']:+.2f}s  MAE {stats['mae_sec']:.2f}s  "
                  f"({stats['relative_mae']:.0%})  limits {stats['loa_low']:+.2f} to "
                  f"{stats['loa_high']:+.2f}s  n={stats['n']}")

    duration_reports(runs, out_dir, ensemble, windows, cost, min_duration, event_iou, plot_cfg)
    # Both passes get their own pages: seeing the same window decoded and undecoded is how the
    # post-processing is judged by eye rather than only by a metric.
    pages = draw_every_window(runs, out_dir, ensemble, windows, cost, min_duration,
                              plot_cfg, per_page, shading=label)
    if pages:
        print(f"every test window ({label}): {len(pages)} pages in {out_dir / PAGES_DIR}")
    print(f"  -> {out_dir}/")
    return metrics


def _compare(both: dict, out_dir: Path, plot_cfg) -> None:
    """What the decoder was worth, side by side. Written whether it helped or not."""
    rows = []
    for metric in sorted(set(both[RAW]) & set(both[DECODED])):
        raw, decoded = both[RAW][metric], both[DECODED][metric]
        rows.append({"metric": metric, RAW: raw, DECODED: decoded,
                     "delta": decoded - raw})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "raw_vs_viterbi.csv", index=False)

    headline = table[table.metric == HEADLINE]
    if len(headline):
        row = headline.iloc[0]
        print(f"\n{HEADLINE}: argmax {row[RAW]:.3f} -> viterbi {row[DECODED]:.3f} "
              f"({row['delta']:+.3f})")
    figures.decoding_report(table, out_dir / "raw_vs_viterbi.png", plot_cfg)
    print(f"{out_dir}/  ({RAW}/, {DECODED}/, raw_vs_viterbi.csv, raw_vs_viterbi.png)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="outputs/human_folds",
                        help="directory holding one subdirectory per fold")
    parser.add_argument("--out", default=None, help="where to write (default: --runs)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--per-page", type=int, default=8,
                        help="test windows per page in the full set of panels")
    parser.add_argument("--labels", default=None,
                        help="label source, for the transition table (default: read the run's "
                             "own .hydra config)")
    parser.add_argument("--enforce-min", choices=("auto", "on", "off"), default="auto",
                        help="minimum-duration pass. `auto` uses what the run decoded with; "
                             "anything else is an ablation and is printed as one")
    parser.add_argument("--durations-only", action="store_true",
                        help="only the phase-duration reports - skips the bootstrap, the fold "
                             "subsets and the window panels")
    args = parser.parse_args()
    override = {"auto": None, "on": True, "off": False}[args.enforce_min]
    out = args.out and Path(args.out)
    if args.durations_only:
        return 0 if durations_only(Path(args.runs), out, args.labels, override) else 1
    return 0 if aggregate(Path(args.runs), out, args.bootstrap, args.per_page,
                          args.labels, override) else 1


if __name__ == "__main__":
    raise SystemExit(main())
