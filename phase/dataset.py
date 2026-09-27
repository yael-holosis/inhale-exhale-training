"""The torch dataset over the built shards, and the augmentations that are safe on it.

One item is one window: a 1-channel trace and its per-sample target.

**Windows keep their own length.** The network is fully convolutional and takes any length; the
only thing that ever wanted a common one is the tensor a DataLoader stacks, and `collate` solves
that by padding each batch to its own longest member and masking the padding - which the loss and
the metrics already honour. Nothing is discarded.

A fixed `crop` is still available and is what a memory-bound run would use, but it is not the
default and it is not free: 6.3% of windows are longer than 200 samples, and they are the *hard*
ones. The pipeline grows a window by 5 s and retries precisely when it cannot find three breaths
in it, so cropping them back to 200 throws away the slow and irregular breathing first. Measured
on the built set, a 200-sample crop discards 2.9% of samples and all of them come from that 6.3%.

**There is no polarity flip.** The build orients every window to the polarity its reviewer
labelled against - `data.labels.orient_by_reviewer_flip`, applied where `ReviewerFlipped` is set
- so polarity is a convention the dataset holds, not noise to be averaged out. Negating half the
windows at random would destroy exactly that convention, so the augmentation is gone and
`amplitude_range` stays positive for the same reason.

What remains only ever changes the trace, never the labels, with one exception worth knowing:
`time_warp` resamples, so it changes a window's **length**. `LengthBucketSampler` buckets on the
stored length, so the warp reintroduces the padding the bucketing exists to remove - masked out
of the loss, but not hidden from `BatchNorm`. See `notebooks/augmentation.ipynb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from phase.building import shard_path
from phase.labels import UNKNOWN
from phase.preprocess import normalise_window


class WindowDataset(Dataset):
    """Windows named by a manifest slice, read from the shards beside it.

    Args:
        manifest: rows of `manifest.csv` - already filtered to this split.
        root: directory holding the shards.
        crop: samples per item. Longer windows are randomly cropped when training and
            centre-cropped otherwise; shorter ones are padded and masked out.
        train: whether to augment.
        augment: the `training.augmentation` config block.
        normalise: 'window' z-scores each item on its own statistics. That is what the device
            can do at inference - it has no corpus statistics - so it is the default.
    """

    PAD_LABEL = UNKNOWN

    def __init__(self, manifest: pd.DataFrame, root: str | Path, crop: int | None = None,
                 train: bool = False, augment: dict[str, Any] | None = None,
                 normalise: str = "window", seed: int = 0):
        self.rows = manifest.reset_index(drop=True)
        self.root = Path(root)
        self.crop = int(crop) if crop else None
        self.train = bool(train)
        self.augment = dict(augment or {})
        self.normalise = normalise
        self.seed = int(seed)
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _shard(self, name: str) -> dict[str, np.ndarray]:
        if name not in self._cache:
            with np.load(shard_path(self.root, name), allow_pickle=False) as stored:
                self._cache[name] = {"values": stored["values"], "targets": stored["targets"],
                                     "offsets": stored["offsets"]}
        return self._cache[name]

    def _window(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.rows.iloc[index]
        shard = self._shard(str(row["shard"]))
        position = int(row["position"])
        start, end = shard["offsets"][position], shard["offsets"][position + 1]
        return (shard["values"][start:end].astype(np.float32),
                shard["targets"][start:end].astype(np.int64))

    def __getitem__(self, key: int | tuple[int, int]) -> dict[str, torch.Tensor]:
        # `(index, epoch)` from a sampler built with `with_epoch`: persistent workers never see
        # the main process's epoch, so it travels with the index.
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        values, target = self._window(index)
        # Seeded when training too, or two identical runs augment differently and disagree.
        rng = np.random.default_rng((self.seed, epoch, index) if self.train
                                    else self.seed + index)

        if self.train:
            values, target = self._augment(values, target, rng)
        values, target, mask = self._fit(values, target, rng)
        values = self._normalise(values)

        return {"x": torch.from_numpy(values[None, :]),
                "y": torch.from_numpy(target),
                "mask": torch.from_numpy(mask),
                "row": index}

    # ------------------------------------------------------------------ shaping

    def _fit(self, values: np.ndarray, target: np.ndarray,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop or pad to `self.crop`, or leave the window alone when there is no crop."""
        if self.crop is None:
            return values, target, np.ones(values.size, dtype=bool)
        length = values.size
        if length >= self.crop:
            spare = length - self.crop
            start = int(rng.integers(spare + 1)) if self.train else spare // 2
            stop = start + self.crop
            return values[start:stop], target[start:stop], np.ones(self.crop, dtype=bool)
        mask = np.zeros(self.crop, dtype=bool)
        mask[:length] = True
        padded_values = np.zeros(self.crop, dtype=np.float32)
        padded_target = np.full(self.crop, self.PAD_LABEL, dtype=np.int64)
        padded_values[:length] = values
        padded_target[:length] = target
        return padded_values, padded_target, mask

    def _normalise(self, values: np.ndarray) -> np.ndarray:
        return normalise_window(values, self.normalise)

    # ------------------------------------------------------------------ augmentation

    def _augment(self, values: np.ndarray, target: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.augment
        if rng.random() < cfg.get("time_warp_prob", 0.0):
            values, target = time_warp(values, target, cfg.get("time_warp_range", (0.8, 1.25)),
                                       rng)
        if rng.random() < cfg.get("amplitude_prob", 0.0):
            low, high = cfg.get("amplitude_range", (0.5, 2.0))
            values = values * float(rng.uniform(low, high))
        if rng.random() < cfg.get("drift_prob", 0.0):
            # Relative to the window, like the noise below. A raw window's std is ~3e-4, so an
            # absolute 0.3 was 315x the breathing - not a wander under it but a wipe of it.
            drift = baseline_drift(values.size, cfg.get("drift_scale", 0.3), rng)
            values = values + drift * (values.std() or 1.0)
        if rng.random() < cfg.get("noise_prob", 0.0):
            scale = cfg.get("noise_scale", 0.05) * (values.std() or 1.0)
            values = values + rng.normal(0.0, scale, values.size).astype(np.float32)
        return values.astype(np.float32), target


def time_warp(values: np.ndarray, target: np.ndarray, factor_range, rng) -> tuple:
    """Resample the window, so a breath at 12 bpm can stand in for one at 15.

    The trace is interpolated linearly and the target with nearest-neighbour - a class index has
    no midpoint, and a linear blend of INHALE and EXHALE would land on the STOP index.
    """
    low, high = factor_range
    factor = float(rng.uniform(low, high))
    length = max(8, int(round(values.size * factor)))
    source = np.linspace(0.0, values.size - 1, values.size)
    wanted = np.linspace(0.0, values.size - 1, length)
    warped = np.interp(wanted, source, values).astype(np.float32)
    indices = np.clip(np.round(wanted).astype(int), 0, values.size - 1)
    return warped, target[indices]


def baseline_drift(length: int, scale: float, rng) -> np.ndarray:
    """A slow wander under the breathing: two sub-breath-rate sinusoids at random phase."""
    t = np.arange(length, dtype=np.float32) / max(length - 1, 1)
    drift = np.zeros(length, dtype=np.float32)
    for _ in range(2):
        cycles = float(rng.uniform(0.25, 1.0))
        drift += np.sin(2 * np.pi * cycles * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
    return (scale * drift / 2.0).astype(np.float32)


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack items of differing length, padding each batch to its own longest member.

    The padding is masked, and both the loss and the metrics already drop masked samples - so a
    window is never cropped to fit a tensor and never learned from beyond its own end.

    Padded with zeros rather than by replication: a replicated edge is a flat run that looks like
    a held breath, and although it is masked out of the loss it still enters the receptive field
    of the samples before it. Zeros after a mean-removed window are its own baseline.

    **Use `LengthBucketSampler` with this.** 94% of windows are exactly 200 samples, which reads
    as "a random batch is almost always uniform" and is not: with 64 to a batch, the chance every
    one of them is 200 is 1.6%, so 98% of batches are padded and the padding averages 40% of the
    tensor - 65% at worst. That padding is masked out of the loss, but it is *not* hidden from
    `BatchNorm`, which normalises over batch and length together and carries its statistics into
    inference where no padding exists. Bucketing removes the cause instead of compensating for it.
    """
    longest = max(int(item["x"].shape[-1]) for item in batch)
    x = torch.zeros(len(batch), batch[0]["x"].shape[0], longest)
    y = torch.zeros(len(batch), longest, dtype=torch.long)
    mask = torch.zeros(len(batch), longest, dtype=torch.bool)
    for position, item in enumerate(batch):
        length = int(item["x"].shape[-1])
        x[position, :, :length] = item["x"]
        y[position, :length] = item["y"]
        mask[position, :length] = item["mask"]
    return {"x": x, "y": y, "mask": mask,
            "row": torch.tensor([int(item["row"]) for item in batch])}


class LengthBucketSampler(Sampler):
    """Batches drawn from windows of one length, so a batch needs no padding at all.

    Windows come in a handful of discrete lengths - the pipeline's 20 s window, plus the ones it
    grew by 5 s and retried - so grouping by length is exact rather than approximate, and only
    the last batch of each group is short.

    Shuffled twice: within a length group, and over the batches themselves. Without the second,
    every epoch would feed all 2,669 plain windows and then all 96 of the 250-sample ones, which
    is a curriculum nobody chose and a `BatchNorm` update history to match.

    **`drop_last` is a trap here and defaults off.** Dropping the short final batch is the usual
    way to avoid an unstable last step, but once batches are bucketed by length the short batch
    is not a remainder - it *is* the whole of a rare length. Measured on the built set, dropping
    it would discard every window of 300 samples and longer: 82 of them, the slowest and most
    irregular breathing in the set, and the part `crop_samples: null` exists to keep. A batch of
    one 600-sample window still gives BatchNorm 600 positions per channel, which is not an
    unstable estimate.
    """

    def __init__(self, lengths, batch_size: int, shuffle: bool = True, drop_last: bool = False,
                 seed: int = 0, with_epoch: bool = False):
        # See the note above: bucketed, a short batch is a whole rare length, not a remainder.
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.with_epoch = bool(with_epoch)
        self.epoch = 0

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for length in np.unique(self.lengths):
            positions = np.flatnonzero(self.lengths == length)
            if self.shuffle:
                positions = positions[rng.permutation(positions.size)]
            for start in range(0, positions.size, self.batch_size):
                chunk = positions[start:start + self.batch_size].tolist()
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        for batch in self._batches():
            yield [(index, self.epoch) for index in batch] if self.with_epoch else batch
        self.epoch += 1

    def __len__(self) -> int:
        return len(self._batches())


def class_weights(manifest: pd.DataFrame, phases, power: float = 1.0,
                  cap: float = 20.0) -> np.ndarray:
    """Inverse-frequency weights from the manifest's own per-class counts.

    Capped, because on a set where one class is nearly absent an uncapped inverse turns a handful
    of samples into the whole loss. `power` between 0 and 1 softens it - 0.5 is the usual choice
    when the raw inverse over-corrects. A class absent from the split - `stop` once it is merged
    into exhale - is left out of the balance and weighted 1; it never appears as a target.
    """
    counts = np.array([float(manifest[f"n_{name}"].sum()) for name in phases])
    present = counts > 0
    weights = np.ones(len(counts))
    if present.any():
        weights[present] = (counts[present].sum() / (present.sum() * counts[present])) ** power
    return np.clip(weights, 1.0 / cap, cap).astype(np.float32)
