"""Train the phase segmenter on one fold, logging to ClearML.

    poetry run python train.py                          # fold 0, config defaults
    poetry run python train.py data.fold=3 training.max_epochs=100
    DISABLE_CLEARML=true poetry run python train.py     # local only

Hydra owns the config (`parameter/`) and the run directory; Lightning owns the loop; ClearML
picks up the scalars through the TensorBoard logger and the config through `connect_configuration`.

**What the numbers mean.** The labels are production's own answer, so every metric here measures
how closely the network reproduces the current algorithm - not how closely it reproduces a
breath. A macro F1 of 1.00 would mean the algorithm has been cloned, including where it is
wrong. `evaluate.py --human` is the run that measures the other thing.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from clearml_utils import initialize_clearml_task, run_title
from models.lightning_module import PhaseSegmenter
from phase import figures
from phase.building import shard_path, load_windows, resolve
from phase.decode import decode
from phase.dataset import LengthBucketSampler, WindowDataset, class_weights, collate
from phase.labels import PHASES
from phase.splits import SPLIT_COLUMN, describe, split_for
from report_folds import (AGGREGATE_FIGURES, AGGREGATE_TABLES, PASSES, PAGES_DIR,
                          aggregate)

TEST_LOGITS_NAME = "test_logits.npz"
FOLD_DIR = "fold_{fold}"


def loaders(cfg: DictConfig, splits: dict[str, pd.DataFrame],
            root: Path) -> dict[str, DataLoader]:
    out = {}
    for name, frame in splits.items():
        train = name == "train"
        dataset = WindowDataset(frame, root, crop=cfg.data.crop_samples, train=train,
                                augment=OmegaConf.to_container(cfg.training.augmentation,
                                                               resolve=True),
                                normalise=cfg.data.normalise, seed=cfg.seed)
        # Batches of one length, so nothing is padded and BatchNorm never sees a padded
        # position. `collate` still pads, because the last batch of a length group can be short
        # and because a caller may want plain batching.
        # `drop_last` stays off: bucketed by length, the short batch is the whole of a rare
        # length rather than a remainder - dropping it would discard every window of 300 samples
        # and longer, which is the part variable-length training exists to keep.
        sampler = LengthBucketSampler(frame["samples"].to_numpy(), cfg.training.batch_size,
                                      shuffle=train, drop_last=False, seed=cfg.seed)
        out[name] = DataLoader(dataset, batch_sampler=sampler,
                               num_workers=cfg.training.num_workers, collate_fn=collate,
                               persistent_workers=cfg.training.num_workers > 0)
    return out


def test_predictions(model, dataset: Path, test: pd.DataFrame) -> list[dict]:
    """One window at a time, so no padding reaches the net and the logits are the window's own.

    The logits are kept, not just the decoded labels: aggregating folds means averaging logits
    before the decoder runs, and a decoded label cannot be averaged back into one.
    """
    model.eval()
    items = []
    for _, row in test.iterrows():
        with np.load(shard_path(dataset, str(row["shard"])), allow_pickle=False) as stored:
            position = int(row["position"])
            start, end = stored["offsets"][position], stored["offsets"][position + 1]
            values = stored["values"][start:end].astype(np.float32)
            reference = stored["targets"][start:end].astype(np.int64)
        centred = values - values.mean()
        scale = centred.std()
        normalised = centred / scale if scale > 1e-8 else centred
        with torch.no_grad():
            logits = model(torch.from_numpy(normalised[None, None, :]))
        logits = logits[0].permute(1, 0).cpu().numpy()
        prediction = decode(logits, model.cost, model.min_duration)
        items.append({"values": values, "reference": reference, "prediction": prediction,
                      "logits": logits, "fps": float(row["analysis_fps"]), "row": row,
                      "score": figures.score(prediction, reference)})
    return items


def save_test_logits(items: list[dict], run_dir: Path) -> Path:
    """Per-sample logits on the test set, ragged like a shard - `report_folds.py` reads these."""
    lengths = np.array([item["logits"].shape[0] for item in items], dtype=np.int64)
    out = run_dir / TEST_LOGITS_NAME
    np.savez_compressed(
        out,
        logits=np.concatenate([item["logits"] for item in items]).astype(np.float32),
        targets=np.concatenate([item["reference"] for item in items]).astype(np.int8),
        offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        window_id=np.array([int(item["row"]["RespirationWindowID"]) for item in items],
                           dtype=np.int64),
        patient=np.array([str(item["row"]["PatientID"]) for item in items]),
        env=np.array([str(item["row"]["env"]) for item in items]),
        fps=np.array([item["fps"] for item in items], dtype=np.float32))
    return out


def test_figure(cfg: DictConfig, items: list[dict], run_dir: Path,
                fold: int) -> Path | None:
    """Draw a spread of test windows - worst, median, best - beside the metrics."""
    if not cfg.plot.windows or not items:
        return None
    chosen = figures.choose(items, cfg.plot.windows, cfg.plot.pick, cfg.seed)
    if not chosen:
        return None
    panels = [{**item, "title": (f"{item['row']['PatientID']} · window "
                                 f"{int(item['row']['RespirationWindowID'])} · "
                                 f"{item['row']['env']} · macro F1 {item['score']:.2f}")}
              for item in chosen]
    return figures.plot_windows(
        panels, run_dir / "test_windows.png",
        f"fold {fold} test set · {cfg.plot.pick} of {len(items)} windows",
        "algorithm" if cfg.data.labels.source == "algorithm" else "labeller",
        OmegaConf.to_container(cfg.plot, resolve=True))


def run_fold(cfg: DictConfig, dataset: Path, manifest: pd.DataFrame, fold: int,
             run_dir: Path, task=None) -> dict:
    """Train one fold and test it. Returns that fold's test metrics."""
    pl.seed_everything(cfg.seed + fold, workers=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    cut = cfg.data.split

    # Read off the columns, never recomputed here: the split that trained a model has to be the
    # one recorded beside the data, not whatever the config happens to say at run time.
    splits = split_for(manifest, fold)
    print(f"dataset {dataset}\nfold {fold} of {cut.folds}, test held out of every fold\n"
          + describe(splits, total=len(manifest), stratify_cols=cut.stratify_cols))

    weights = class_weights(splits["train"], PHASES, power=cfg.training.class_weight_power,
                            cap=cfg.training.class_weight_cap)
    print("class weights: " + "  ".join(f"{name}={value:.2f}"
                                        for name, value in zip(PHASES, weights)))

    fps = float(splits["train"]["analysis_fps"].median())
    model = PhaseSegmenter(model=OmegaConf.to_container(cfg.model, resolve=True),
                           training=OmegaConf.to_container(cfg.training, resolve=True),
                           class_weights=weights.tolist(), fps=fps, fold=fold,
                           label_source=str(cfg.data.labels.source))
    print(f"{cfg.model.name}: {model.net.n_parameters():,} parameters, "
          f"receptive field {model.net.receptive_field()} samples "
          f"({model.net.receptive_field() / fps:.0f} s at {fps:g} fps)")


    data = loaders(cfg, splits, dataset)
    checkpoint = ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor=cfg.training.monitor,
                                 mode=cfg.training.monitor_mode, save_top_k=1,
                                 filename="best-{epoch:02d}")
    callbacks = [checkpoint]
    if cfg.training.early_stopping_patience:
        callbacks.append(EarlyStopping(monitor=cfg.training.monitor,
                                       mode=cfg.training.monitor_mode,
                                       patience=cfg.training.early_stopping_patience))

    trainer = pl.Trainer(max_epochs=cfg.training.max_epochs,
                         accelerator=cfg.training.accelerator,
                         precision=cfg.training.precision, callbacks=callbacks,
                         logger=pl.loggers.TensorBoardLogger(save_dir=str(run_dir), name=None,
                                                             version=""),
                         default_root_dir=str(run_dir), log_every_n_steps=10)
    trainer.fit(model, data["train"], data["val"])
    results = trainer.test(model, data["test"], ckpt_path=checkpoint.best_model_path or None)

    report = pd.DataFrame(results)
    report.to_csv(run_dir / "test_metrics.csv", index=False)
    # The dataset that produced these numbers, named beside them - a run directory that only says
    # "fold 0" cannot be traced back to what it was fold 0 of.
    (run_dir / "dataset.txt").write_text(f"{dataset}\n")
    print("\ntest, fold %d" % fold)
    for key, value in sorted(results[0].items()):
        print(f"  {key:34s} {value:.2f}")

    # A page of test windows with every run, not on request: a macro F1 does not say whether the
    # breaths came out as breaths, and nobody goes back to draw one for a run that looked fine.
    items = test_predictions(model, dataset, splits["test"])
    print(f"test logits: {save_test_logits(items, run_dir)}")
    figure = test_figure(cfg, items, run_dir, fold)
    if figure:
        print(f"test windows: {figure}")

    # The checkpoint is the artifact, so it travels with the task rather than sitting in a run
    # directory somebody has to find.
    if task is not None and checkpoint.best_model_path:
        task.upload_artifact(f"fold_{fold}_checkpoint",
                             artifact_object=checkpoint.best_model_path)
        task.upload_artifact(f"fold_{fold}_test_metrics",
                             artifact_object=str(run_dir / "test_metrics.csv"))
        # The per-fold page is not uploaded: the aggregate writes every test window for both
        # passes, so a third undifferentiated "test windows" group is only noise.
    return results[0]


@hydra.main(version_base=None, config_path="parameter", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    dataset = resolve(cfg.data.root, cfg.data.dir, cfg.data.labels.source)
    manifest = load_windows(dataset)
    cut = cfg.data.split
    if SPLIT_COLUMN not in manifest.columns:
        raise KeyError(f"{dataset} carries no splits - run: "
                       f"poetry run python make_splits.py --dataset {dataset}")

    # One task for the whole run, so every fold reports into the same charts as its own
    # series. A task per fold would put each on a chart of its own and compare nothing.
    # The division a result is reported against: train against test, the same in every fold.
    # Validation rotates inside the training side and is a property of a fold, not of the run.
    splits_figure = figures.split_summary(
        manifest, None, run_dir / "splits.png",
        OmegaConf.to_container(cfg.plot, resolve=True))

    task = None
    if cfg.clearml.enabled:
        task = initialize_clearml_task(project_name=cfg.clearml.project_name,
                                       task_name=run_title(cfg),
                                       timeout=cfg.clearml.timeout_s, cfg=cfg)
    if task is not None and splits_figure.exists():
        task.get_logger().report_image("splits", "train and test, by patient", iteration=0,
                                       local_path=str(splits_figure), max_image_history=1)

    wanted = int(cut.folds)
    available = sum(1 for column in manifest.columns
                    if column.endswith("_split") and column != SPLIT_COLUMN)
    if not 1 <= wanted <= available:
        raise ValueError(f"data.split.folds={wanted}, but {dataset.name} carries {available} - "
                         f"re-split it, or override to at most {available}")

    for fold in range(wanted):
        print(f"\n{'=' * 30} fold {fold} of {wanted} {'=' * 30}")
        run_fold(cfg, dataset, manifest, fold, run_dir / FOLD_DIR.format(fold=fold),
                 task)

    if wanted > 1:
        # Aggregated here rather than left to be remembered: the folds are only comparable
        # because they share a test set, and that is exactly what gets forgotten.
        print(f"\n{'=' * 30} aggregating {wanted} folds {'=' * 30}")
        aggregate(run_dir, run_dir, label_source=str(cfg.data.labels.source))
        if task is not None:
            # The aggregate is the result; it belongs on the task rather than only on disk.
            # Both passes go up: `raw` is the network alone, `viterbi` is after post-processing,
            # and the comparison is what says whether the post-processing earned its place.
            logger = task.get_logger()
            uploaded = [name for name in cfg.clearml.passes if name in PASSES]
            missing = [name for name in cfg.clearml.passes if name not in PASSES]
            if missing:
                raise ValueError(f"clearml.passes has {missing}, which no pass produces; "
                                 f"the passes are {list(PASSES)}")
            print(f"uploading the {', '.join(uploaded)} pass to ClearML; "
                  f"{', '.join(n for n in PASSES if n not in uploaded) or 'nothing'} stays on "
                  f"disk only")
            for pass_name in uploaded:
                for name in AGGREGATE_FIGURES:
                    figure = run_dir / pass_name / f"{name}.png"
                    if figure.exists():
                        logger.report_image(f"aggregate - {pass_name}", name, iteration=0,
                                            local_path=str(figure), max_image_history=1)
                for name in AGGREGATE_TABLES:
                    table = run_dir / pass_name / f"{name}.csv"
                    if table.exists():
                        task.upload_artifact(f"{pass_name}_{name}", artifact_object=str(table))
            for pass_name in uploaded:
                pages = sorted((run_dir / pass_name / PAGES_DIR).glob("page_*.png"))
                for page in pages[:6]:
                    logger.report_image(f"test windows - {pass_name}", page.stem,
                                        iteration=0, local_path=str(page),
                                        max_image_history=6)
            comparison = run_dir / "raw_vs_viterbi.png"
            if comparison.exists():
                logger.report_image("raw vs viterbi", "what the decoder changed", iteration=0,
                                    local_path=str(comparison), max_image_history=1)
            if (run_dir / "raw_vs_viterbi.csv").exists():
                task.upload_artifact("raw_vs_viterbi",
                                     artifact_object=str(run_dir / "raw_vs_viterbi.csv"))


if __name__ == "__main__":
    main()
