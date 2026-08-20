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
from phase.dataset import WindowDataset, class_weights
from phase.labels import PHASES
from phase.splits import describe, patient_folds, split_for


def loaders(cfg: DictConfig, splits: dict[str, pd.DataFrame]) -> dict[str, DataLoader]:
    root = Path(cfg.data.dir)
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


@hydra.main(version_base=None, config_path="parameter", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed + cfg.data.fold, workers=True)
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    manifest_path = Path(cfg.data.dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no dataset at {manifest_path} - run build_dataset.py first")
    manifest = pd.read_csv(manifest_path)

    manifest = patient_folds(manifest, cfg.data.folds, seed=cfg.seed)
    splits = split_for(manifest, cfg.data.fold, cfg.data.val_fraction, seed=cfg.seed)
    print(f"fold {cfg.data.fold} of {cfg.data.folds}\n{describe(splits)}")

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

    data = loaders(cfg, splits)
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
    print("\ntest, fold %d" % cfg.data.fold)
    for key, value in sorted(results[0].items()):
        print(f"  {key:34s} {value:.2f}")

    # The checkpoint is the artifact, so it travels with the task rather than sitting in a run
    # directory somebody has to find.
    if task is not None and checkpoint.best_model_path:
        task.upload_artifact("best_checkpoint", artifact_object=checkpoint.best_model_path)
        task.upload_artifact("test_metrics", artifact_object=str(run_dir / "test_metrics.csv"))


if __name__ == "__main__":
    main()
