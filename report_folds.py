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
from phase import figures
from phase.building import load_windows, shard_path
from phase.decode import decode, transition_matrix
from phase.labels import N_CLASSES, PHASES, UNKNOWN
from phase.labelsources import ALGORITHM
from phase.metrics import event_level, per_sample
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


def traces_for(runs: Path, window_ids: np.ndarray,
               envs: np.ndarray) -> dict[tuple[str, int], dict]:
    """The stored waveform and provenance for each test window, keyed by `(env, window id)`.

    The signal and the index within it come along because they are what identifies a window in
    the labelling app - the window id alone does not, it collides across the two instances.

    Keyed by the pair because the id alone is not unique - the two instances have separate id
    spaces and 13 of 360 windows collide on the current set. The traces come from the dataset the
    run names in `dataset.txt`, which is why that file is written beside every run.
    """
    named = next((d / DATASET_NAME for d in sorted(runs.iterdir())
                  if (d / DATASET_NAME).exists()), None)
    if named is None:
        return {}
    dataset = Path(named.read_text().strip())
    if not dataset.exists():
        print(f"{dataset} is gone - cannot draw the windows")
        return {}

    manifest = load_windows(dataset).set_index(["env", "RespirationWindowID"])
    out = {}
    for env, window_id in zip(envs.tolist(), window_ids.tolist()):
        key = (str(env), int(window_id))
        if key not in manifest.index:
            continue
        row = manifest.loc[key]
        with np.load(shard_path(dataset, str(row["shard"])), allow_pickle=False) as stored:
            position = int(row["position"])
            start, end = stored["offsets"][position], stored["offsets"][position + 1]
            out[key] = {"values": stored["values"][start:end].astype(np.float32),
                        "signal": int(row["RadarSignalID"]),
                        "session": int(row["SessionID"]),
                        "window_index": int(row["WindowIndex"])}
    return out


def draw_every_window(runs: Path, out_dir: Path, ensemble: dict, windows, cost, min_duration,
                      plot_cfg, per_page: int = 8) -> list[Path]:
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
            f"test set, ensemble · page {page + 1} of {total} · "
            f"windows {page * per_page + 1}-{page * per_page + len(chunk)} of {len(items)}",
            "labeller", plot_cfg, prediction_name="ensemble",
            caption="shading = ensemble · ribbon = labeller"))
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


def aggregate(runs: Path, out_dir: Path | None = None, draws: int = 2000,
              per_page: int = 8) -> dict | None:
    """Pool every fold under `runs`. Returns the ensemble metrics, or None if there are none."""
    runs = Path(runs)
    out_dir = Path(out_dir) if out_dir else runs
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

    cfg = OmegaConf.merge(
        OmegaConf.load("parameter/config.yaml"),
        {"training": OmegaConf.load("parameter/training/default.yaml"),
         "plot": OmegaConf.load("parameter/config.yaml").plot})
    decoding = OmegaConf.to_container(cfg.training.decoding, resolve=True)
    # The same table the run decoded with - scoring under a different one would measure a
    # model nobody trained.
    source = str(cfg.data.labels.source) if "data" in cfg else ALGORITHM
    cost = (transition_matrix(allowed_for(decoding, source), decoding["switch_penalty"])
            if decoding["viterbi"] else None)
    min_duration = decoding["min_duration"] if decoding["enforce_min"] else None
    fps = float(np.median(reference["fps"]))
    event_iou = float(cfg.training.event_iou)

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
    stacked = np.stack([fold["logits"] for fold in folds])
    ensemble = {**reference, "logits": stacked.mean(axis=0), "fold": "ensemble"}
    windows = windows_of(ensemble)
    metrics = score_windows(windows, cost, min_duration, fps, event_iou)

    patients = ensemble["patient"]
    low, high = bootstrap(windows, patients, cost, min_duration, fps, event_iou,
                          draws, int(cfg.seed))

    headline = per_fold[HEADLINE] if HEADLINE in per_fold else pd.Series(dtype=float)
    print(f"\nper fold {HEADLINE}: "
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
    plot_cfg = OmegaConf.to_container(cfg.plot, resolve=True)
    figures.fold_report(per_fold, metrics, (low, high),
                        confusion(windows, cost, min_duration),
                        out_dir / "fold_report.png", plot_cfg, headline=HEADLINE)
    figures.patient_report(scored, metrics, out_dir / "per_patient.png", plot_cfg)
    figures.scaling_report(scaling, out_dir / "fold_scaling.png", plot_cfg, headline=HEADLINE)
    pages = draw_every_window(runs, out_dir, ensemble, windows, cost, min_duration, plot_cfg,
                              per_page)
    if pages:
        print(f"every test window: {len(pages)} pages in {out_dir / PAGES_DIR}")
    print(f"\n{out_dir}/  (per_fold.csv, ensemble_metrics.csv, per_patient.csv, "
          f"fold_scaling.csv, fold_report.png, per_patient.png, fold_scaling.png)")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="outputs/human_folds",
                        help="directory holding one subdirectory per fold")
    parser.add_argument("--out", default=None, help="where to write (default: --runs)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--per-page", type=int, default=8,
                        help="test windows per page in the full set of panels")
    args = parser.parse_args()
    return 0 if aggregate(Path(args.runs), args.out and Path(args.out),
                          args.bootstrap, args.per_page) else 1


if __name__ == "__main__":
    raise SystemExit(main())
