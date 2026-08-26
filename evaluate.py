"""Score a checkpoint - on the algorithm-labelled held-out fold, or against human labels.

    poetry run python evaluate.py --checkpoint outputs/.../best-epoch=41.ckpt
    poetry run python evaluate.py --checkpoint ... --plot 6
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

`--plot N` draws N test windows to `out/test_windows/`. **Sampled as a spread by default** -
worst, median, best - because a page of randomly drawn windows on a set this size is mostly
median ones, and hides both tails. `--plot-pick worst` when a number needs explaining.
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
from phase import figures, sources
from phase.decode import decode
from phase.labels import PHASES, UNKNOWN, spans_to_targets
from phase.building import existing_params, shard_path, load_windows, resolve
from phase.corrections import Corrections
from phase.metrics import event_level, per_sample, phase_durations
from phase.splits import split_for

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
    with np.load(shard_path(root, str(row["shard"])), allow_pickle=False) as stored:
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


def dataset_of(cfg, override: str | None = None) -> Path:
    return resolve(cfg.data.root, override or cfg.data.dir, cfg.data.labels.source)


def collect(model, root: Path, frame: pd.DataFrame) -> list[dict]:
    """Every window of a split, with its samples, its reference and the model's answer."""
    out = []
    for _, row in frame.iterrows():
        values, reference = window_samples(root, row)
        prediction = predict(model, values)
        out.append({"values": values, "reference": reference, "prediction": prediction,
                    "fps": float(row["analysis_fps"]), "row": row,
                    "score": figures.score(prediction, reference)})
    return out


def draw(items: list[dict], out_dir: Path, name: str, heading: str, how: str, n: int,
         seed: int, reference_name: str, plot_cfg=None) -> Path | None:
    chosen = figures.choose(items, n, how, seed)
    if not chosen:
        return None
    panels = [{**item,
               "title": (f"{item['row']['PatientID']} · window "
                         f"{int(item['row']['RespirationWindowID'])} · "
                         f"{item['row']['env']} · macro F1 {item['score']:.2f}")}
              for item in chosen]
    return figures.plot_windows(panels, Path(out_dir) / name, heading, reference_name,
                                plot_cfg)


def on_fold(args, cfg) -> int:
    root = dataset_of(cfg, args.dataset)
    manifest = load_windows(root)
    test = split_for(manifest, args.fold)["test"]
    print(f"{root}\nfold {args.fold}: {len(test)} windows, "
          f"{test['PatientID'].nunique()} held-out patients")

    model = load_model(args.checkpoint)
    items = collect(model, root, test)
    preds = [item["prediction"] for item in items]
    truths = [item["reference"] for item in items]

    fps = float(test["analysis_fps"].median())
    print("\nagainst production's own labels - this measures imitation, not correctness:")
    report("model vs teacher", np.concatenate(preds), np.concatenate(truths))
    print("\nmean phase durations:")
    for name, labels in (("model", np.concatenate(preds)), ("teacher", np.concatenate(truths))):
        durations = phase_durations(labels, fps)
        print(f"  {name:8s} " + "  ".join(f"{key}={value:.2f}"
                                          for key, value in durations.items()))

    scores = np.array([item["score"] for item in items])
    # Split on whether the reference says anything at all. A window the detector found nothing in
    # is all `unknown`, so any phase the model calls there scores zero by construction - pooling
    # those with the rest reports a disagreement with an absent opinion as a modelling error.
    labelled = np.array([bool(item["row"]["n_spans"] > 0) for item in items])
    print(f"\nper-window macro F1 over {len(scores)} test windows: "
          f"median {np.median(scores):.2f}, worst {scores.min():.2f}, best {scores.max():.2f}")
    if labelled.any():
        print(f"  reference has phases ({labelled.sum():4d} windows): "
              f"median {np.median(scores[labelled]):.2f}, "
              f"{100 * (scores[labelled] < 0.5).mean():.0f}% below 0.50")
        report("  pooled over those", np.concatenate([i["prediction"] for i, keep
                                                      in zip(items, labelled) if keep]),
               np.concatenate([i["reference"] for i, keep in zip(items, labelled) if keep]))
    if (~labelled).any():
        called = np.array([(item["prediction"] != UNKNOWN).mean()
                           for item, keep in zip(items, labelled) if not keep])
        print(f"  reference all unknown ({(~labelled).sum():4d} windows): the detector found "
              f"nothing in them, so every score here is 0.00 by construction.")
        print(f"      the model calls a phase on {100 * called.mean():.0f}% of their samples. "
              f"Whether it is right is not something the algorithm can answer - it is the case "
              f"for human labels.")

    if args.plot:
        pool = items
        if args.plot_filter == "labelled":
            pool = [item for item, keep in zip(items, labelled) if keep]
        elif args.plot_filter == "unlabelled":
            pool = [item for item, keep in zip(items, labelled) if not keep]
        if not pool:
            print(f"\nno {args.plot_filter} windows to draw")
            return 0
        suffix = "" if args.plot_filter == "all" else f"_{args.plot_filter}"
        path = draw(pool, Path(cfg.out_dir) / "test_windows",
                    f"{Path(args.checkpoint).stem}_fold{args.fold}_{args.plot_pick}{suffix}.png",
                    f"fold {args.fold} test set · {args.plot_pick} of {len(pool)} "
                    + {"all": "windows", "labelled": "windows the algorithm labelled",
                       "unlabelled": "windows the algorithm left entirely unknown"}[args.plot_filter],
                    args.plot_pick, args.plot, cfg.seed, "algorithm",
                    OmegaConf.to_container(cfg.plot, resolve=True))
        print(f"\n{path}")
    return 0


def on_human(args, cfg) -> int:
    """Every window in the dataset that also carries human spans, scored three ways."""
    # Reads two accounts, like a build does - sign both in before the first query rather than
    # discovering the second is dead halfway through.
    if not sources.ensure_session():
        return 1
    root = dataset_of(cfg, args.dataset)
    manifest = load_windows(root)
    corrections = Corrections.from_config(
        existing_params(root).get("labels", {}).get("corrections"))

    rows = []
    for env, group in manifest.groupby("env"):
        # `Spans` is the count of phase records on the window. There is no boolean "labelled"
        # column - filtering on one that is not there silently scores every window.
        catalogue = sources.catalogue(str(env))
        labelled = catalogue.loc[catalogue["Spans"] > 0, "ID"].astype(int)
        wanted = group[group["RespirationWindowID"].isin(set(labelled))]
        if wanted.empty:
            continue
        names = sources.phase_vocabulary(str(env))
        for _, row in wanted.iterrows():
            spans = sources.human_spans(str(env), int(row["RespirationWindowID"]))
            if spans.empty:
                continue
            values, teacher = window_samples(root, row)
            # `EndIndex` on a span is **inclusive** - adjacent spans share a boundary sample -
            # while the window's is exclusive. Off by one here shifts every human boundary.
            human = spans_to_targets(
                [{"phase": names[int(span["BreathPhaseTypeID"])],
                  "start": int(span["StartIndex"]), "end": int(span["EndIndex"]) + 1}
                 for span in spans.to_dict("records")], values.size)
            # The same corrections the dataset was built under, or this scores the model on a
            # target it was never trained against.
            human, _ = corrections.apply(human)
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
    parser.add_argument("--dataset", default=None,
                        help="which dataset directory - a checkpoint outlives a config edit")
    parser.add_argument("--plot", type=int, default=0, metavar="N",
                        help="draw N test windows to out/test_windows/")
    parser.add_argument("--plot-pick", choices=list(figures.PICKS), default=figures.SPREAD,
                        help="which N: spread (worst..best), worst, best, or random")
    parser.add_argument("--plot-filter", choices=("all", "labelled", "unlabelled"),
                        default="all",
                        help="which windows to draw from: those the reference labelled, those "
                             "it left entirely unknown, or all of them")
    args = parser.parse_args()

    root = OmegaConf.load(CONFIG_DIR / "config.yaml")
    cfg = OmegaConf.merge(root, {"data": OmegaConf.load(CONFIG_DIR / "data" /
                                                        f"{root.defaults[0]['data']}.yaml")})
    args.fold = 0 if args.fold is None else args.fold
    return on_human(args, cfg) if args.human else on_fold(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
