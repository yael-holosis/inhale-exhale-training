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
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from phase import figures
from phase.decode import decode, transition_matrix
from phase.labels import N_CLASSES, PHASES
from phase.metrics import event_level, per_sample

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


def aggregate(runs: Path, out_dir: Path | None = None, draws: int = 2000) -> dict | None:
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
    decoding = cfg.training.decoding
    cost = (transition_matrix(OmegaConf.to_container(decoding.allowed, resolve=True),
                              decoding.switch_penalty) if decoding.viterbi else None)
    min_duration = (OmegaConf.to_container(decoding.min_duration, resolve=True)
                    if decoding.enforce_min else None)
    fps = float(np.median(reference["fps"]))
    event_iou = float(cfg.training.event_iou)

    # ---------------------------------------------------------------- per fold
    rows = []
    for fold in folds:
        path = Path(runs / fold["fold"] / METRICS_NAME)
        got = pd.read_csv(path).iloc[0].to_dict() if path.exists() else {}
        rows.append({"fold": fold["fold"],
                     **{k.replace("test/", ""): v for k, v in got.items()}})
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

    # ---------------------------------------------------------------- figures
    plot_cfg = OmegaConf.to_container(cfg.plot, resolve=True)
    figures.fold_report(per_fold, metrics, (low, high),
                        confusion(windows, cost, min_duration),
                        out_dir / "fold_report.png", plot_cfg, headline=HEADLINE)
    print(f"\n{out_dir}/  (per_fold.csv, ensemble_metrics.csv, fold_report.png)")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="outputs/human_folds",
                        help="directory holding one subdirectory per fold")
    parser.add_argument("--out", default=None, help="where to write (default: --runs)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    return 0 if aggregate(Path(args.runs), args.out and Path(args.out),
                          args.bootstrap) else 1


if __name__ == "__main__":
    raise SystemExit(main())
