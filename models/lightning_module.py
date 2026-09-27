"""The training step: loss, decoding, and what gets logged.

Loss is weighted cross-entropy plus soft Dice. Cross-entropy alone optimises per-sample
agreement, which on a set that is 60-70% `unknown` is satisfied by a model that never commits;
Dice is computed per class over the whole window, so a class that is small in samples still
carries a full share of it.

Padding is masked out of both. A window padded to the crop length has invented samples in it,
and a model rewarded for predicting `unknown` on them learns the padding, not the breath.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet1d import UNet1D
from phase.labelsources import ALGORITHM, SOURCES
from phase.decode import decode, transition_matrix
from phase.labels import classes_for
from phase.metrics import event_level, per_sample


STOP_AS_EXHALE = "stop_as_exhale"
"""`decoding.allowed` key holding the per-source tables for a dataset with `stop` merged away."""


def allowed_for(decoding: dict, label_source: str, stop_as_exhale: bool = False) -> dict:
    """The transition table for this label set.

    Keyed by source, because the two label sets disagree about the commonest transition of all:
    a person draws inhale straight into exhale, production never does. A flat table is taken as
    written so an older config still loads.
    """
    allowed = decoding["allowed"]
    if stop_as_exhale:
        if STOP_AS_EXHALE not in allowed:
            raise KeyError(f"decoding.allowed has no {STOP_AS_EXHALE} tables, and the dataset "
                           "merges stop into exhale")
        allowed = allowed[STOP_AS_EXHALE]
    if not set(allowed) & set(SOURCES):
        return allowed
    if label_source not in allowed:
        raise KeyError(f"decoding.allowed has no table for labels.source={label_source!r}; "
                       f"it has {sorted(allowed)}")
    return allowed[label_source]


def soft_dice(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    """**1 - mean per-class Dice**, so this is a loss and zero is perfect. Classes absent from a
    batch are skipped. The coefficient itself is logged separately as `dice`."""
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 2, 1).float()
    keep = mask.unsqueeze(1).float()
    probs, onehot = probs * keep, onehot * keep

    intersection = (probs * onehot).sum(dim=(0, 2))
    totals = probs.sum(dim=(0, 2)) + onehot.sum(dim=(0, 2))
    present = onehot.sum(dim=(0, 2)) > 0
    dice = (2 * intersection + eps) / (totals + eps)
    return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0


class PhaseSegmenter(pl.LightningModule):
    """Wraps the U-Net with its loss, its decoder and its metrics.

    Args:
        model: the `model` config block.
        training: the `training` block - loss weights, optimiser, decoding.
        class_weights: per-class loss weights from the training split's own counts, as a list.
        fps: analysis rate, for turning sample errors into seconds in the logs.
    """

    HEADLINE = "macro_f1"
    """The one metric on the charts, for val and test alike. Per-class Dice and per-class F1 are
    the same quantity, so this is also the mean Dice - there is no second thing to reconcile."""

    def __init__(self, model: dict[str, Any], training: dict[str, Any],
                 class_weights: list[float] | None = None, fps: float = 10.0,
                 fold: int = 0, label_source: str = ALGORITHM, stop_as_exhale: bool = False):
        super().__init__()
        # A plain list, not an array: `save_hyperparameters` pickles what it was given, and
        # torch.load defaults to weights_only=True, which refuses a numpy global on reload.
        self.save_hyperparameters()
        self.classes = classes_for(stop_as_exhale)
        self.net = UNet1D(in_channels=model["in_channels"], n_classes=len(self.classes),
                          channels=tuple(model["channels"]), bottleneck=model["bottleneck"],
                          kernel_size=model["kernel_size"], dropout=model.get("dropout", 0.0))
        self.cfg = training
        self.fps = float(fps)
        self.fold = int(fold)
        weights = (torch.ones(len(self.classes)) if class_weights is None
                   else torch.as_tensor(np.asarray(class_weights), dtype=torch.float32))
        self.register_buffer("class_weights", weights)

        decoding = training.get("decoding", {})
        self.cost = (transition_matrix(allowed_for(decoding, label_source, stop_as_exhale),
                                       decoding["switch_penalty"], self.classes)
                     if decoding.get("viterbi") else None)
        self.min_duration = decoding.get("min_duration") if decoding.get("enforce_min") else None
        self._epoch: dict[str, list] = {}
        self._ce: list[float] = []
        self._dice: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # --------------------------------------------------------------------- loss

    def _loss(self, logits, target, mask) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_element = F.cross_entropy(logits, target, weight=self.class_weights,
                                      reduction="none")
        counted = mask.float()
        ce = (per_element * counted).sum() / counted.sum().clamp(min=1.0)
        dice_loss = soft_dice(logits, target, mask)
        total = self.cfg["ce_weight"] * ce + self.cfg["dice_weight"] * dice_loss
        # Both, and named for what they are: `soft_dice` returns 1 - Dice, so logging it as
        # "dice" reads as a coefficient falling towards zero when it is a loss doing its job.
        return total, {"ce": ce.detach(), "dice_loss": dice_loss.detach(),
                       "dice": (1.0 - dice_loss).detach()}

    def _step(self, batch, stage: str):
        logits = self(batch["x"])
        loss, parts = self._loss(logits, batch["y"], batch["mask"])
        self.log(f"{stage}/loss", loss, prog_bar=(stage == "val"), batch_size=len(batch["y"]))
        if stage == "train":
            # Accumulated rather than logged per step: the two training charts are epoch means,
            # and a per-step series of either is unreadable at 60 epochs.
            self._ce.append(float(parts["ce"]))
            self._dice.append(float(parts["dice"]))
        if stage != "train":
            self._collect(stage, logits.detach(), batch["y"], batch["mask"])
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    # ----------------------------------------------------------------- metrics

    def _collect(self, stage: str, logits, target, mask):
        """Decode on CPU and hold the epoch's windows. Viterbi is per window, not batched."""
        store = self._epoch.setdefault(stage, [])
        logits = logits.permute(0, 2, 1).float().cpu().numpy()
        target = target.cpu().numpy()
        mask = mask.cpu().numpy().astype(bool)
        for item in range(logits.shape[0]):
            keep = mask[item]
            store.append((decode(logits[item][keep], self.cost, self.min_duration),
                          target[item][keep]))

    def _report(self, stage: str):
        store = self._epoch.pop(stage, [])
        if not store:
            return
        flat_pred = np.concatenate([pred for pred, _ in store])
        flat_true = np.concatenate([truth for _, truth in store])
        metrics = per_sample(flat_pred, flat_true, classes=self.classes)

        events: dict[str, list[float]] = {}
        for pred, truth in store:
            for key, value in event_level(pred, truth, self.cfg.get("event_iou", 0.5),
                                          self.classes).items():
                events.setdefault(key, []).append(value)
        for key, values in events.items():
            clean = [v for v in values if not np.isnan(v)]
            if clean:
                metrics[key] = float(np.mean(clean))
        if "boundary_mae_samples" in metrics:
            metrics["boundary_mae_sec"] = metrics["boundary_mae_samples"] / self.fps

        # Only the headline. Logging all two dozen put every metric on one chart per split,
        # which is unreadable - and `report_folds.py` recomputes them all from the saved logits
        # anyway, into `raw/` and `viterbi/`.
        self.log(f"{stage}/{self.HEADLINE}", float(metrics.get(self.HEADLINE, 0.0)))

    # ------------------------------------------------------------- clearml charts

    def _chart(self, title: str, value: float, iteration: int) -> None:
        """One chart per (metric, split), one series per fold - the cough convention.

        Fetched off `Task.current_task()` rather than plumbed in, so nothing here depends on
        ClearML being enabled.
        """
        try:
            from clearml import Task
        except ImportError:
            return
        task = Task.current_task()
        if task is None:
            return
        task.get_logger().report_scalar(title=title, series=f"fold {self.fold}",
                                        value=float(value), iteration=int(iteration))

    def _logged(self, key: str) -> float | None:
        value = self.trainer.callback_metrics.get(key) if self.trainer else None
        return None if value is None else float(value)

    def on_train_epoch_end(self):
        loss = self._logged("train/loss")
        if loss is not None:
            self._chart("loss - train", loss, self.current_epoch)
        # `dice` is the coefficient, 1 - the loss term, so it rises towards 1 like every other
        # score on the page rather than falling towards zero.
        if self._dice:
            self._chart("dice - train", sum(self._dice) / len(self._dice), self.current_epoch)
        if self._ce:
            self._chart("ce - train", sum(self._ce) / len(self._ce), self.current_epoch)
        self._ce.clear()
        self._dice.clear()

    def on_validation_epoch_end(self):
        self._report("val")
        for title, key in (("loss - val", "val/loss"),
                           (f"{self.HEADLINE} - val", f"val/{self.HEADLINE}")):
            value = self._logged(key)
            if value is not None:
                self._chart(title, value, self.current_epoch)

    def on_test_epoch_end(self):
        self._report("test")
        # Iteration is the fold, not the epoch: one point per fold, so the chart reads as a
        # comparison across folds rather than a line going nowhere.
        for title, key in (("loss - test", "test/loss"),
                           (f"{self.HEADLINE} - test", f"test/{self.HEADLINE}")):
            value = self._logged(key)
            if value is not None:
                self._chart(title, value, self.fold)

    # --------------------------------------------------------------- optimiser

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"],
                                      weight_decay=self.cfg["weight_decay"])
        if not self.cfg.get("scheduler", "cosine"):
            return optimiser
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.cfg["max_epochs"], eta_min=self.cfg["lr"] * 0.01)
        return {"optimizer": optimiser, "lr_scheduler": scheduler}
