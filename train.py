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
from phase.building import load_windows, resolve
from phase.decode import decode
from phase.dataset import WindowDataset, class_weights
from phase.labels import PHASES
from phase.splits import SPLIT_COLUMN, describe, split_for


def loaders(cfg: DictConfig, splits: dict[str, pd.DataFrame],
            root: Path) -> dict[str, DataLoader]:
    out = {}
    for name, frame in splits.items():
        train = name == "train"
        dataset = WindowDataset(frame, root, crop=cfg.data.crop_samples, train=train,
                                augment=OmegaConf.to_container(cfg.training.augmentation,
                                                               resolve=True),
                                normalise=cfg.data.normalise, seed=cfg.seed)
        out[name] = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=train,
                               num_workers=cfg.data.num_workers, drop_last=train,
                               persistent_workers=cfg.data.num_workers > 0)
    return out


def test_figure(cfg: DictConfig, model, dataset: Path, test: pd.DataFrame,
                run_dir: Path) -> Path | None:
    """Draw a spread of test windows - worst, median, best - beside the metrics."""
    if not cfg.plot.windows:
        return None
    model.eval()
    items = []
    for _, row in test.iterrows():
        with np.load(dataset / str(row["shard"]), allow_pickle=False) as stored:
            position = int(row["position"])
            start, end = stored["offsets"][position], stored["offsets"][position + 1]
            values = stored["values"][start:end].astype(np.float32)
            reference = stored["targets"][start:end].astype(np.int64)
        centred = values - values.mean()
        scale = centred.std()
        normalised = centred / scale if scale > 1e-8 else centred
        with torch.no_grad():
            logits = model(torch.from_numpy(normalised[None, None, :]))
        prediction = decode(logits[0].permute(1, 0).cpu().numpy(), model.cost,
                            model.min_duration)
        items.append({"values": values, "reference": reference, "prediction": prediction,
                      "fps": float(row["analysis_fps"]), "row": row,
                      "score": figures.score(prediction, reference)})

    chosen = figures.choose(items, cfg.plot.windows, cfg.plot.pick, cfg.seed)
    if not chosen:
        return None
    panels = [{**item, "title": (f"{item['row']['PatientID']} · window "
                                 f"{int(item['row']['RespirationWindowID'])} · "
                                 f"{item['row']['env']} · macro F1 {item['score']:.2f}")}
              for item in chosen]
    return figures.plot_windows(
        panels, run_dir / "test_windows.png",
        f"fold {cfg.data.split.fold} test set · {cfg.plot.pick} of {len(items)} windows",
        "the production algorithm" if cfg.data.labels.source == "algorithm" else "a labeller")


@hydra.main(version_base=None, config_path="parameter", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed + cfg.data.split.fold, workers=True)
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    dataset = resolve(cfg.data.root, cfg.data.dir, cfg.data.labels.source)
    manifest = load_windows(dataset)
    cut = cfg.data.split
    if SPLIT_COLUMN not in manifest.columns:
        raise KeyError(f"{dataset} carries no splits - run: "
                       f"poetry run python make_splits.py --dataset {dataset}")

    # Read off the columns, never recomputed here: the split that trained a model has to be the
    # one recorded beside the data, not whatever the config happens to say at run time.
    splits = split_for(manifest, cut.fold)
    print(f"dataset {dataset}\nfold {cut.fold} of {cut.folds}, test held out of every fold\n"
          + describe(splits, total=len(manifest), stratify_cols=cut.stratify_cols))

    weights = class_weights(splits["train"], PHASES, power=cfg.training.class_weight_power,
                            cap=cfg.training.class_weight_cap)
    print("class weights: " + "  ".join(f"{name}={value:.2f}"
                                        for name, value in zip(PHASES, weights)))

    fps = float(splits["train"]["analysis_fps"].median())
    model = PhaseSegmenter(model=OmegaConf.to_container(cfg.model, resolve=True),
                           training=OmegaConf.to_container(cfg.training, resolve=True),
                           class_weights=weights.tolist(), fps=fps)
    print(f"{cfg.model.name}: {model.net.n_parameters():,} parameters, "
          f"receptive field {model.net.receptive_field()} samples "
          f"({model.net.receptive_field() / fps:.0f} s at {fps:g} fps)")

    task = None
    if cfg.clearml.enabled:
        task = initialize_clearml_task(project_name=cfg.clearml.project_name,
                                       task_name=run_title(cfg), timeout=cfg.clearml.timeout_s,
                                       cfg=cfg)

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
    print("\ntest, fold %d" % cut.fold)
    for key, value in sorted(results[0].items()):
        print(f"  {key:34s} {value:.2f}")

    # A page of test windows with every run, not on request: a macro F1 does not say whether the
    # breaths came out as breaths, and nobody goes back to draw one for a run that looked fine.
    figure = test_figure(cfg, model, dataset, splits["test"], run_dir)
    if figure:
        print(f"test windows: {figure}")

    # The checkpoint is the artifact, so it travels with the task rather than sitting in a run
    # directory somebody has to find.
    if task is not None and checkpoint.best_model_path:
        task.upload_artifact("best_checkpoint", artifact_object=checkpoint.best_model_path)
        task.upload_artifact("test_metrics", artifact_object=str(run_dir / "test_metrics.csv"))
        if figure:
            task.get_logger().report_image("test windows", cfg.plot.pick, iteration=0,
                                           local_path=str(figure), max_image_history=1)


if __name__ == "__main__":
    main()
