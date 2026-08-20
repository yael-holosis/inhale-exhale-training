"""A dataset that looks like a built one, without AWS. For the tests and for a smoke run.

Breaths are asymmetric on purpose - the rise and the fall differ - because a network trained
under the polarity-flip augmentation has nothing else to tell inhale from exhale. A symmetric
synthetic set would make the task impossible and the smoke run would look like a bug.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase.labels import EXHALE, INHALE, STOP, UNKNOWN, class_counts
from phase.building import MANIFEST_NAME, SHARD_TEMPLATE
from phase.labels import PHASES

ENV = "synthetic"
"""Stands where `ds_algo` / `ds_prod` would be. The shard name carries it because the two real
environments have unrelated RadarSignal ID spaces."""


def breath(rate_bpm: float, fps: float, rise_fraction: float,
           rng) -> tuple[np.ndarray, np.ndarray]:
    """One breath: a rise, a fall, then a pause at the trough."""
    period = max(8, int(round(60.0 / rate_bpm * fps)))
    rise = max(2, int(round(period * rise_fraction)))
    pause = max(1, int(round(period * 0.2)))
    fall = max(2, period - rise - pause)

    values = np.concatenate([
        np.sin(np.linspace(-np.pi / 2, np.pi / 2, rise)),
        np.sin(np.linspace(np.pi / 2, -np.pi / 2, fall)),
        np.full(pause, -1.0)]).astype(np.float32)
    target = np.concatenate([
        np.full(rise, INHALE), np.full(fall, EXHALE), np.full(pause, STOP)]).astype(np.int64)
    # Production emits no phase for the turn at the crest, so the synthetic set carries the same
    # hole - otherwise the transition rules in the decoder describe a set that does not exist.
    target[rise - 1:rise + 1] = UNKNOWN
    return values + rng.normal(0, 0.03, values.size).astype(np.float32), target


def window(samples: int, rate_bpm: float, fps: float, rng) -> tuple[np.ndarray, np.ndarray]:
    values, targets = [], []
    while sum(v.size for v in values) < samples:
        v, t = breath(rate_bpm, fps, rise_fraction=float(rng.uniform(0.25, 0.4)), rng=rng)
        values.append(v)
        targets.append(t)
    values = np.concatenate(values)[:samples]
    targets = np.concatenate(targets)[:samples]
    if rng.random() < 0.5:                     # the arbitrary stored sign
        values = -values
        targets = np.where(targets == INHALE, EXHALE,
                           np.where(targets == EXHALE, INHALE, targets))
    return values.astype(np.float32), targets.astype(np.int64)


def build(root: Path, n_patients: int = 8, signals_per_patient: int = 3,
          windows_per_signal: int = 4, samples: int = 200, fps: float = 10.0,
          seed: int = 0) -> pd.DataFrame:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, signal_id = [], 1000

    for patient in range(n_patients):
        name = f"synth-{patient:03d}"
        rate = float(rng.uniform(8, 22))
        for _ in range(signals_per_patient):
            signal_id += 1
            values, targets = [], []
            for _ in range(windows_per_signal):
                v, t = window(samples, rate * float(rng.uniform(0.85, 1.15)), fps, rng)
                values.append(v)
                targets.append(t)
            lengths = np.array([v.size for v in values], dtype=np.int64)
            offsets = np.concatenate(([0], np.cumsum(lengths)))
            shard = SHARD_TEMPLATE.format(env=ENV, signal_id=signal_id)
            np.savez_compressed(
                root / shard,
                values=np.concatenate(values).astype(np.float32),
                targets=np.concatenate(targets).astype(np.int8),
                offsets=offsets,
                window_index=np.arange(windows_per_signal, dtype=np.int64),
                start_index=np.arange(windows_per_signal, dtype=np.int64) * 50,
                analysis_fps=np.full(windows_per_signal, fps, dtype=np.float32),
                respiration_rate=np.full(windows_per_signal, rate, dtype=np.float32),
                range_bin=np.zeros(windows_per_signal, dtype=np.int64),
                rejected=np.zeros(windows_per_signal, dtype=bool),
                orientation=np.array(["kept"] * windows_per_signal),
                signal_id=np.int64(signal_id), patient=np.str_(name),
                session_id=np.int64(patient), env=np.str_(ENV))
            for position in range(windows_per_signal):
                counts = class_counts(targets[position])
                rows.append({"env": ENV, "shard": shard, "position": position,
                             "RadarSignalID": signal_id, "SessionID": patient,
                             "PatientID": name, "WindowIndex": position,
                             "start_index": position * 50, "samples": samples,
                             "analysis_fps": fps, "respiration_rate": rate, "range_bin": 0,
                             "rejected": False, "orientation": "kept", "n_spans": 0,
                             **{f"n_{phase}": counts[phase] for phase in PHASES}})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / MANIFEST_NAME, index=False)
    return manifest


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data_sets/synthetic")
    frame = build(target)
    print(f"{len(frame)} windows, {frame['PatientID'].nunique()} patients -> {target}")
