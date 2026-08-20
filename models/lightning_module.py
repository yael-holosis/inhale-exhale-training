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
from phase.decode import decode, transition_matrix
from phase.labels import N_CLASSES, PHASES
from phase.metrics import event_level, per_sample


def soft_dice(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    """1 - mean per-class Dice over the batch. Classes absent from a batch are skipped."""
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, N_CLASSES).permute(0, 2, 1).float()
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

    def __init__(self, model: dict[str, Any], training: dict[str, Any],
                 class_weights: list[float] | None = None, fps: float = 10.0):
        super().__init__()
        # A plain list, not an array: `save_hyperparameters` pickles what it was given, and
        # torch.load defaults to weights_only=True, which refuses a numpy global on reload.
        self.save_hyperparameters()
        self.net = UNet1D(in_channels=model["in_channels"], n_classes=N_CLASSES,
                          channels=tuple(model["channels"]), bottleneck=model["bottleneck"],
                          kernel_size=model["kernel_size"], dropout=model.get("dropout", 0.0))
        self.cfg = training
        self.fps = float(fps)
        weights = (torch.ones(N_CLASSES) if class_weights is None
                   else torch.as_tensor(np.asarray(class_weights), dtype=torch.float32))
        self.register_buffer("class_weights", weights)

        decoding = training.get("decoding", {})
        self.cost = (transition_matrix(decoding["allowed"], decoding["switch_penalty"])
                     if decoding.get("viterbi") else None)
        self.min_duration = decoding.get("min_duration") if decoding.get("enforce_min") else None
        self._epoch: dict[str, list] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # --------------------------------------------------------------------- loss

    def _loss(self, logits, target, mask) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_element = F.cross_entropy(logits, target, weight=self.class_weights,
                                      reduction="none")
        counted = mask.float()
        ce = (per_element * counted).sum() / counted.sum().clamp(min=1.0)
        dice = soft_dice(logits, target, mask)
        total = self.cfg["ce_weight"] * ce + self.cfg["dice_weight"] * dice
        return total, {"ce": ce.detach(), "dice": dice.detach()}

    def _step(self, batch, stage: str):
        logits = self(batch["x"])
        loss, parts = self._loss(logits, batch["y"], batch["mask"])
        self.log(f"{stage}/loss", loss, prog_bar=(stage == "val"), batch_size=len(batch["y"]))
        for name, value in parts.items():
            self.log(f"{stage}/{name}", value, batch_size=len(batch["y"]))
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
        metrics = per_sample(flat_pred, flat_true)

        events: dict[str, list[float]] = {}
        for pred, truth in store:
            for key, value in event_level(pred, truth,
                                          self.cfg.get("event_iou", 0.5)).items():
                events.setdefault(key, []).append(value)
        for key, values in events.items():
            clean = [v for v in values if not np.isnan(v)]
            if clean:
                metrics[key] = float(np.mean(clean))
        if "boundary_mae_samples" in metrics:
            metrics["boundary_mae_sec"] = metrics["boundary_mae_samples"] / self.fps

        for key, value in metrics.items():
            self.log(f"{stage}/{key}", float(value))

    def on_validation_epoch_end(self):
        self._report("val")

    def on_test_epoch_end(self):
        self._report("test")

    # --------------------------------------------------------------- optimiser

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"],
                                      weight_decay=self.cfg["weight_decay"])
        if not self.cfg.get("scheduler", "cosine"):
            return optimiser
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.cfg["max_epochs"], eta_min=self.cfg["lr"] * 0.01)
        return {"optimizer": optimiser, "lr_scheduler": scheduler}
