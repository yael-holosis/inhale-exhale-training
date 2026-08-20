"""Score a checkpoint - on the algorithm-labelled held-out fold, or against human labels.

    poetry run python evaluate.py --checkpoint outputs/.../best-epoch=41.ckpt
    poetry run python evaluate.py --checkpoint ... --human

**The two runs answer different questions.**

Without `--human` the reference is production's own answer, so the score says how faithfully the
network reproduces the current algorithm. That is what the training set is, and a high number
there means the clone is good - including wherever the algorithm is wrong.

With `--human` the reference is `BreathPhaseTimeRecord`. It prints **three** rows, and the third
is the point of the whole exercise:

    model   vs human      what the network gets right
    teacher vs human      what production gets right on the same windows
    model   vs teacher    how much of the gap is the network's own

A student cannot beat its teacher by imitating it. If row 1 is not at least row 2, the network
has not yet earned its place. The human set is small - 21 windows on `ds_algo`, 2 on `ds_prod` at
the time of writing - so read it as a sanity check, not a verdict.

Nothing is re-derived for the comparison. The samples come from the shard the network trained
against, the teacher's labels are the ones stored in it, and the human spans are joined on
`RespirationWindowID`. All three therefore describe the same array, which is the only way the
three rows are comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from models.lightning_module import PhaseSegmenter
from phase import bridge
from phase.decode import decode
from phase.labels import PHASES, spans_to_targets
from phase.metrics import event_level, per_sample, phase_durations
from phase.splits import patient_folds, split_for

CONFIG_DIR = Path(__file__).parent / "parameter"
REPORT_KEYS = ("macro_f1", "f1_inhale", "f1_exhale", "f1_stop", "f1_unknown",
               "event_f1_inhale", "event_f1_exhale", "event_f1_stop", "boundary_mae_samples")


def load_model(checkpoint: str) -> PhaseSegmenter:
    model = PhaseSegmenter.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    return model


@torch.no_grad()
def predict(model: PhaseSegmenter, values: np.ndarray) -> np.ndarray:
    """One window or one whole signal. Fully convolutional, so length does not matter."""
    centred = np.asarray(values, dtype=np.float64) - np.mean(values)
    scale = centred.std()
    normalised = centred / scale if scale > 1e-8 else centred
    logits = model(torch.from_numpy(normalised[None, None, :].astype(np.float32)))
    return decode(logits[0].permute(1, 0).numpy(), model.cost, model.min_duration)


def window_samples(root: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The stored trace and the teacher's labels for one manifest row, from the local shard."""
    with np.load(root / str(row["shard"]), allow_pickle=False) as stored:
        position = int(row["position"])
        start, end = stored["offsets"][position], stored["offsets"][position + 1]
        return (stored["values"][start:end].astype(np.float32),
                stored["targets"][start:end].astype(np.int64))


def report(name: str, pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    scores = {**per_sample(pred, truth), **event_level(pred, truth)}
    line = "  ".join(f"{key.replace('boundary_mae_samples', 'bnd')}={scores[key]:.2f}"
                     for key in REPORT_KEYS if key in scores)
    print(f"  {name:22s} {line}")
    return scores


def load_manifest(cfg) -> pd.DataFrame:
    path = Path(cfg.data.dir) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"no dataset at {path} - run build_dataset.py first")
    return pd.read_csv(path)


def on_fold(args, cfg) -> int:
    manifest = patient_folds(load_manifest(cfg), cfg.data.folds, seed=cfg.seed)
    test = split_for(manifest, args.fold, cfg.data.val_fraction, seed=cfg.seed)["test"]
    print(f"fold {args.fold}: {len(test)} windows, "
          f"{test['PatientID'].nunique()} held-out patients")

    model = load_model(args.checkpoint)
    root = Path(cfg.data.dir)
    preds, truths = [], []
    for _, row in test.iterrows():
        values, target = window_samples(root, row)
        preds.append(predict(model, values))
        truths.append(target)

    fps = float(test["analysis_fps"].median())
    print("\nagainst production's own labels - this measures imitation, not correctness:")
    report("model vs teacher", np.concatenate(preds), np.concatenate(truths))
    print("\nmean phase durations:")
    for name, labels in (("model", np.concatenate(preds)), ("teacher", np.concatenate(truths))):
        durations = phase_durations(labels, fps)
        print(f"  {name:8s} " + "  ".join(f"{key}={value:.2f}"
                                          for key, value in durations.items()))
    return 0


def on_human(args, cfg) -> int:
    """Every window in the dataset that also carries human spans, scored three ways."""
    bundle = bridge.require_app(str(cfg.repos.labeling_app))
    manifest = load_manifest(cfg)
    root = Path(cfg.data.dir)

    rows = []
    for env, group in manifest.groupby("env"):
        # `Spans` is the count of phase records on the window. The app's `browse` frame has no
        # boolean "labelled" column - filtering on the wrong one silently scores every window.
        catalogue = bundle["db"].browse(str(env))
        labelled = catalogue.loc[catalogue["Spans"].fillna(0) > 0, "ID"].astype(int)
        wanted = group[group["RespirationWindowID"].isin(set(labelled))]
        if wanted.empty:
            continue
        phase_ids = {str(name).lower(): int(ident) for ident, name in
                     bundle["db"].frame(str(env), bundle["db"].LABELS,
                                        "SELECT ID, Name FROM BreathPhaseType").itertuples(
                                            index=False)}
        names = {ident: name for name, ident in phase_ids.items()}
        for _, row in wanted.iterrows():
            spans = bundle["db"].spans_for_window(str(env), int(row["RespirationWindowID"]))
            if spans.empty:
                continue
            values, teacher = window_samples(root, row)
            # `EndIndex` on a span is **inclusive** - adjacent spans share a boundary sample -
            # while the window's is exclusive. Off by one here shifts every human boundary.
            human = spans_to_targets(
                [{"phase": names[int(span["BreathPhaseTypeID"])],
                  "start": int(span["StartIndex"]), "end": int(span["EndIndex"]) + 1}
                 for span in spans.to_dict("records")], values.size)
            rows.append({"model": None, "values": values, "human": human, "teacher": teacher,
                         "fps": float(row["analysis_fps"]), "patient": row["PatientID"],
                         "window": f"{int(row['RadarSignalID'])}_w{int(row['WindowIndex'])}"})

    if not rows:
        print("no window in this dataset carries human spans - nothing to score against")
        return 1

    model = load_model(args.checkpoint)
    for row in rows:
        row["model"] = predict(model, row["values"])

    print(f"\n{len(rows)} human-labelled windows, "
          f"{len({row['patient'] for row in rows})} patients\n")
    stacked = {key: np.concatenate([row[key] for row in rows])
               for key in ("model", "human", "teacher")}
    report("model vs human", stacked["model"], stacked["human"])
    report("teacher vs human", stacked["teacher"], stacked["human"])
    report("model vs teacher", stacked["model"], stacked["teacher"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--human", action="store_true",
                        help="score against BreathPhaseTimeRecord instead of the built labels")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--data-dir", default=None,
                        help="override data.dir - a checkpoint outlives a config edit")
    args = parser.parse_args()

    root = OmegaConf.load(CONFIG_DIR / "config.yaml")
    cfg = OmegaConf.merge(root, {"data": OmegaConf.load(CONFIG_DIR / "data" /
                                                        f"{root.defaults[0]['data']}.yaml")})
    if args.data_dir:
        cfg.data.dir = args.data_dir
    args.fold = cfg.data.fold if args.fold is None else args.fold
    return on_human(args, cfg) if args.human else on_fold(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
