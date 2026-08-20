"""Score a checkpoint - on the algorithm-labelled held-out fold, or against human labels.

    poetry run python evaluate.py --checkpoint outputs/.../best-epoch=41.ckpt
    poetry run python evaluate.py --checkpoint ... --human --env ds_prod

**The two runs answer different questions.**

Without `--human` the reference is production's own answer, so the score says how faithfully the
network reproduces the current algorithm. That is what the training set is, and a high number
there means the clone is good - including wherever the algorithm is wrong.

With `--human` the reference is `BreathPhaseTimeRecord`, read through the labelling repo. It
prints **three** rows, and the third is the point of the whole exercise:

    model   vs human      what the network gets right
    teacher vs human      what production gets right on the same windows
    model   vs teacher    how much of the gap is the network's own

A student cannot beat its teacher by imitating it. If row 1 is not at least row 2, the network
has not yet earned its place; if it is above row 2, the smoothing and the whole-signal context
have bought something the per-window algorithm does not have. Either way the human set is small
- 15 approved windows as of 2026-08-18 - so read it as a sanity check, not as a verdict.
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
from phase.dataset import WindowDataset
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
    centred = values - values.mean()
    scale = centred.std()
    normalised = centred / scale if scale > 1e-8 else centred
    logits = model(torch.from_numpy(normalised[None, None, :].astype(np.float32)))
    return decode(logits[0].permute(1, 0).numpy(), model.cost, model.min_duration)


def report(name: str, pred: np.ndarray, truth: np.ndarray, fps: float) -> dict[str, float]:
    scores = {**per_sample(pred, truth), **event_level(pred, truth)}
    line = "  ".join(f"{key.replace('boundary_mae_samples', 'bnd')}={scores[key]:.2f}"
                     for key in REPORT_KEYS if key in scores)
    print(f"  {name:22s} {line}")
    return scores


def on_fold(args, cfg) -> int:
    manifest = pd.read_csv(Path(cfg.data.dir) / "manifest.csv")
    manifest = patient_folds(manifest, cfg.data.folds, seed=cfg.seed)
    test = split_for(manifest, args.fold, cfg.data.val_fraction, seed=cfg.seed)["test"]
    print(f"fold {args.fold}: {len(test)} windows, "
          f"{test['PatientID'].nunique()} held-out patients")

    model = load_model(args.checkpoint)
    dataset = WindowDataset(test, Path(cfg.data.dir), crop=cfg.data.crop_samples, train=False,
                            normalise=cfg.data.normalise, seed=cfg.seed)
    preds, truths = [], []
    for index in range(len(dataset)):
        values, target = dataset._window(index)
        preds.append(predict(model, values))
        truths.append(target)

    fps = float(test["analysis_fps"].median())
    print("\nagainst production's own labels - this measures imitation, not correctness:")
    report("model vs teacher", np.concatenate(preds), np.concatenate(truths), fps)
    print("\nmean phase durations, model against production:")
    for name, labels in (("model", np.concatenate(preds)), ("teacher", np.concatenate(truths))):
        durations = phase_durations(labels, fps)
        print(f"  {name:8s} " + "  ".join(f"{key}={value:.2f}"
                                          for key, value in durations.items()))
    return 0


def on_human(args, cfg) -> int:
    """Every window in `BreathPhaseTimeRecord`, scored three ways."""
    bundle = bridge.require_labeling(str(cfg.repos.labeling))
    sys.path.append(str(bundle["root"]))
    from utils import db as labeling_db                                   # noqa: PLC0415
    from utils import waveforms as labeling_waveforms                     # noqa: PLC0415

    windows = labeling_db.browse(args.env)
    labelled = windows[windows["Labelled"].astype(str).str.strip().astype(bool)] \
        if "Labelled" in windows.columns else windows
    if labelled.empty:
        print(f"no labelled windows on {args.env}")
        return 1

    phase_ids = {row["Name"].lower(): int(row["ID"])
                 for _, row in labeling_db.frame(args.env, labeling_db.LABELS,
                                                 "SELECT ID, Name FROM BreathPhaseType").iterrows()}
    model = load_model(args.checkpoint)

    rows = []
    for _, window in labelled.iterrows():
        spans = labeling_db.spans_for_window(args.env, int(window["ID"]))
        if spans.empty:
            continue
        values, _ = labeling_waveforms.window_samples(str(window["WaveformS3Path"]))
        human = spans_to_targets(
            [{"phase": _name_of(int(row["BreathPhaseTypeID"]), phase_ids),
              "start": int(row["StartIndex"]), "end": int(row["EndIndex"]) + 1}
             for _, row in spans.iterrows()], values.size)
        teacher_rows, _ = bundle["suggestion"].suggest(values, window.get("RespirationRate"),
                                                       float(window["AnalysisFps"]))
        teacher = spans_to_targets(teacher_rows, values.size)
        rows.append({"model": predict(model, values), "human": human, "teacher": teacher,
                     "fps": float(window["AnalysisFps"]),
                     "window": f"{int(window['RadarSignalID'])}_w{int(window['WindowIndex'])}"})

    if not rows:
        print("no window carried both samples and spans")
        return 1

    fps = float(np.median([row["fps"] for row in rows]))
    print(f"\n{len(rows)} human-labelled windows on {args.env}, {fps:g} fps\n")
    model_all = np.concatenate([row["model"] for row in rows])
    human_all = np.concatenate([row["human"] for row in rows])
    teacher_all = np.concatenate([row["teacher"] for row in rows])
    report("model vs human", model_all, human_all, fps)
    report("teacher vs human", teacher_all, human_all, fps)
    report("model vs teacher", model_all, teacher_all, fps)
    return 0


def _name_of(type_id: int, phase_ids: dict[str, int]) -> str:
    for name, value in phase_ids.items():
        if value == type_id:
            return name
    raise KeyError(f"phase type {type_id} is not in BreathPhaseType")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--human", action="store_true",
                        help="score against BreathPhaseTimeRecord instead of the built dataset")
    parser.add_argument("--env", default=None, help="which instance, for --human")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--data-dir", default=None,
                        help="override data.dir - a checkpoint outlives a config edit")
    args = parser.parse_args()

    root = OmegaConf.load(CONFIG_DIR / "config.yaml")
    cfg = OmegaConf.merge(root, {"data": OmegaConf.load(CONFIG_DIR / "data" /
                                                        f"{root.defaults[0]['data']}.yaml")})
    if args.data_dir:
        cfg.data.dir = args.data_dir
    args.env = args.env or cfg.data.env
    args.fold = cfg.data.fold if args.fold is None else args.fold
    return on_human(args, cfg) if args.human else on_fold(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
