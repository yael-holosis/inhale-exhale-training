"""Rank every window by its own loss, out of fold, and draw the worst ones.

Same audit as `notebooks/high_loss.ipynb`, with one filter added: a window is only drawn if the
labeller called every sample in it. `unknown` is where the two label sets disagree by
construction - a labeller marks it where they could not call the trace - and a gap in the ribbon
produces a large loss on its own, which buries the windows where the label is complete and the
model still disagrees. Those are the ones that say something about the labels.

No window is scored by a model that trained on it: a train-pool window is validation in exactly
one fold and that fold judges it, test windows are outside every fold and get the ensemble.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.lightning_module import PhaseSegmenter
from phase import figures
from phase.building import load_windows, shard_path
from phase.decode import decode
from phase.labels import EXHALE, INHALE, UNKNOWN
from phase.splits import FOLD_COLUMN, SPLIT_COLUMN, TEST, TRAIN, VAL

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "out" / "high_loss"
ROLES = (VAL, TEST, TRAIN)
"""Order for the report. `val` and `test` first: those are the honest scores. A `train` row is
the same window judged by a fold that trained on it, and it is here only as the contrast."""

ROLE_NOTE = {
    VAL: "held out of the one fold that judged it",
    TEST: "outside every fold, judged by the five averaged",
    TRAIN: "judged by a fold that trained on it - memorised, shown for contrast",
}
WORST = 8
"""Panels per figure. A page taller than this stops being readable in a browser."""


def complete(run: Path) -> bool:
    """A run is auditable only once every fold it started has written its dataset."""
    folds = sorted(run.glob("fold_*"))
    return bool(folds) and all((f / "dataset.txt").is_file() for f in folds)


def newest_run(outputs: Path) -> Path:
    """Hydra writes `outputs/<date>/<time>/`; older runs are `outputs/<name>/`."""
    runs = sorted((r for r in [*outputs.glob("*"), *outputs.glob("*/*")]
                   if r.is_dir() and complete(r)),
                  key=lambda r: (r / "fold_0" / "dataset.txt").stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no finished run under {outputs}")
    return runs[-1]


def load_folds(run: Path) -> dict[int, PhaseSegmenter]:
    models = {}
    for path in sorted(run.glob("fold_*/checkpoints/*.ckpt")):
        model = PhaseSegmenter.load_from_checkpoint(path, map_location="cpu")
        model.eval()
        models[int(path.parent.parent.name.split("_")[1])] = model
    if not models:
        raise FileNotFoundError(f"{run} has no checkpoints")
    return models


def roles_of(row, folds) -> list[tuple[str, list[int]]]:
    """Every (role, judging folds) pair for one window."""
    if row[SPLIT_COLUMN] == TEST:
        return [(TEST, sorted(folds))]
    held_out = [k for k in folds if row.get(FOLD_COLUMN.format(fold=k)) == VAL]
    trained_on = [k for k in folds if row.get(FOLD_COLUMN.format(fold=k)) == TRAIN]
    pairs = []
    if held_out:
        pairs.append((VAL, held_out))
    if trained_on:
        pairs.append((TRAIN, trained_on[:1]))
    return pairs


def trace_of(dataset: Path, row) -> tuple[np.ndarray, np.ndarray]:
    with np.load(shard_path(dataset, str(row["shard"])), allow_pickle=False) as stored:
        position = int(row["position"])
        start, end = stored["offsets"][position], stored["offsets"][position + 1]
        return (stored["values"][start:end].astype(np.float32),
                stored["targets"][start:end].astype(np.int64))


def normalise(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean()
    scale = centred.std()
    return centred / scale if scale > 1e-8 else centred


def polarity(values: np.ndarray, target: np.ndarray) -> bool | None:
    """Does inhale rise here? A falling inhale is the signature of an inverted label set."""
    step = np.diff(values.astype(float))
    rising = (target[:-1] == INHALE) & (target[1:] == INHALE)
    falling = (target[:-1] == EXHALE) & (target[1:] == EXHALE)
    if rising.sum() < 3 or falling.sum() < 3:
        return None
    return bool(step[rising].mean() > step[falling].mean())


def score_run(run: Path, dataset: Path, manifest: pd.DataFrame,
              folds: dict[int, PhaseSegmenter]) -> pd.DataFrame:
    """One row per (window, role): its loss, its score, and how much of it was left uncalled."""
    rows = []
    for _, row in manifest.iterrows():
        values, target = trace_of(dataset, row)
        x = torch.from_numpy(normalise(values)[None, None, :])
        for role, judges in roles_of(row, folds):
            with torch.no_grad():
                logits = torch.stack([folds[k](x)[0] for k in judges]).mean(0)
            truth = torch.from_numpy(target)[None, :]
            # Unweighted mean cross-entropy: comparable between windows, which the
            # class-weighted training loss is not.
            loss = float(F.cross_entropy(logits[None, ...], truth).item())
            judge = folds[judges[0]]
            prediction = decode(logits.permute(1, 0).numpy(), judge.cost, judge.min_duration)
            rows.append({
                "role": role, "loss": loss,
                "macro_f1": figures.score(prediction, target),
                "label_unknown": float((target == UNKNOWN).mean()),
                "model_unknown": float((prediction == UNKNOWN).mean()),
                "inhale_rises": polarity(values, target),
                "patient": row["PatientID"], "study": row.get("study"), "env": row["env"],
                "window": int(row["RespirationWindowID"]),
                "signal": int(row["RadarSignalID"]), "session": int(row["SessionID"]),
                "window_index": int(row["WindowIndex"]),
                "flipped": bool(row["reviewer_flipped"]), "samples": int(row["samples"]),
                "judged_by": "+".join(str(k) for k in judges),
                "values": values, "target": target, "prediction": prediction,
                "fps": float(row["analysis_fps"]),
            })
    return pd.DataFrame(rows).sort_values("loss", ascending=False).reset_index(drop=True)


def title_of(row) -> str:
    """Enough to open the window in the labelling app: signal, session, index within signal."""
    return (f"{row['patient']} · {row['env']} · signal {row['signal']} · "
            f"session {row['session']} · window {row['window']} (index {row['window_index']})"
            f" · loss {row['loss']:.2f} · F1 {row['macro_f1']:.2f}"
            + ("  ⚠ inverted" if row["inhale_rises"] is False else "")
            + ("  (flipped)" if row["flipped"] else ""))


def draw(part: pd.DataFrame, out_path: Path, heading: str, plot_cfg) -> Path | None:
    if part.empty:
        return None
    panels = [{"values": r["values"], "reference": r["target"], "prediction": r["prediction"],
               "fps": r["fps"], "title": title_of(r)} for _, r in part.iterrows()]
    return figures.plot_windows(panels, out_path, heading, "labeller", plot_cfg,
                                prediction_name="model",
                                caption="shading = model · ribbon = labeller")


def summarise(table: pd.DataFrame, called: pd.DataFrame) -> pd.DataFrame:
    """Loss per role, over every window and over the fully-called ones only."""
    rows = []
    for role in ROLES:
        every = table[table.role == role]["loss"]
        complete_only = called[called.role == role]["loss"]
        if every.empty:
            continue
        rows.append({"role": role, "windows": len(every),
                     "mean loss": every.mean(), "median loss": every.median(),
                     "fully called": len(complete_only),
                     "mean loss, fully called": complete_only.mean(),
                     "median loss, fully called": complete_only.median(),
                     "max loss, fully called": complete_only.max()})
    return pd.DataFrame(rows).round(3)


def by_loss_decile(called: pd.DataFrame) -> pd.DataFrame:
    """How much of the model's answer is `unknown`, per loss decile, over the called windows.

    The labeller committed on every sample of these, so any `unknown` the model emits is a
    refusal to call a stretch that was callable - worth reporting next to the loss it earns.
    """
    honest = called[called.role.isin((VAL, TEST))].copy()
    if honest.empty:
        return pd.DataFrame()
    honest["decile"] = pd.qcut(honest.loss, 10, labels=False)
    return (honest.groupby("decile")[["loss", "model_unknown", "macro_f1"]]
            .agg(["count", "mean"]).round(3))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="run directory under outputs/ (default: newest finished)")
    parser.add_argument("--out", default=str(OUT_DIR), help="where the figures go")
    parser.add_argument("--worst", type=int, default=WORST, help="panels per figure")
    parser.add_argument("--max-label-unknown", type=float, default=0.0,
                        help="keep windows whose labels are at most this fraction unknown")
    parser.add_argument("--max-model-unknown", type=float, default=0.30,
                        help="drop windows where the model declined on more than this "
                             "fraction - a window the model mostly refused says nothing about "
                             "the labelling")
    parser.add_argument("--tag", default="", help="suffix for the figure filenames")
    args = parser.parse_args()

    run = Path(args.run) if args.run else newest_run(REPO / "outputs")
    if not run.is_absolute():
        run = REPO / run
    dataset = Path((run / "fold_0" / "dataset.txt").read_text().strip())
    if not dataset.is_absolute():
        dataset = REPO / dataset

    plot_cfg = OmegaConf.to_container(
        OmegaConf.load(REPO / "parameter" / "config.yaml").plot, resolve=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_windows(dataset)
    folds = load_folds(run)
    print(f"run     {run.relative_to(REPO)}  -  {len(folds)} folds")
    print(f"dataset {dataset.name}  -  {len(manifest)} windows")

    scored = score_run(run, dataset, manifest, folds)
    table = scored.drop(columns=["values", "target", "prediction"])
    table.to_csv(out / "window_loss.csv", index=False)

    called = scored[(scored.label_unknown <= args.max_label_unknown)
                    & (scored.model_unknown <= args.max_model_unknown)]
    # val and test partition the manifest exactly once each; a window id repeats across
    # instances, so counting distinct ids would undercount.
    once = scored.role.isin((VAL, TEST))
    print(f"\nfully-called windows (labels at most {args.max_label_unknown:.0%} unknown): "
          f"{int((called.role.isin((VAL, TEST))).sum())} of {int(once.sum())}\n")
    flat = called.drop(columns=["values", "target", "prediction"])
    stats = summarise(table, flat)
    stats.to_csv(out / "loss_by_role.csv", index=False)
    print(stats.to_string(index=False))

    deciles = by_loss_decile(flat)
    deciles.to_csv(out / "called_loss_deciles.csv")
    print(f"\n{deciles.to_string()}")
    inverted = flat[flat.inhale_rises == False]                          # noqa: E712
    print(f"\ninverted polarity among the called windows: {len(inverted)} rows")

    tag = f"_{args.tag}" if args.tag else ""
    for role in ROLES:
        part = called[called.role == role].head(args.worst)
        path = draw(part, out / f"called_worst_{role}{tag}.png",
                    f"{role}: the {len(part)} highest-loss windows the labeller called in full "
                    f"- {ROLE_NOTE[role]}", plot_cfg)
        if path:
            print(f"wrote {path.relative_to(REPO)}")
        part.drop(columns=["values", "target", "prediction"]).to_csv(
            out / f"called_worst_{role}{tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
