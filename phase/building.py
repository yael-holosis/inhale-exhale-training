"""Build the training set: sample signals, run the production algorithm, keep what it called.

The dataset is **distillation**. Every label in it is production's own
`calculate_inhale_exhale_time` answer on a window production itself produced, so the target a
model fits here is the current algorithm, not the breath. Two consequences that belong on the
front of any result read off it:

- Its boundaries are the 10% and 90% amplitude crossings - rise and fall times, a median 65% of
  the true trough-to-crest rise (`inhale-exhale-detection/claude/FINDINGS.md`, finding 5). A
  model trained on them reproduces truncated phases, not full ones.
- Its ceiling is the teacher. Where the algorithm is wrong, the target is wrong. Human labels
  from `BreathPhaseTimeRecord` are the only thing that measures either of them, which is why
  every row here carries `RadarSignalID` and `WindowIndex` - so a labelled window can be joined
  in later and held out.

Nothing is written to `RespirationWindow`. That table is the labelling app's browse list and a
run of this size would bury it; the dataset is its own artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase import bridge
from phase.labels import PHASES, class_counts, spans_to_targets

MANIFEST_NAME = "manifest.csv"
SHARD_TEMPLATE = "signal_{signal_id}.npz"
SUMMARY_NAME = "build_summary.yaml"


@dataclass
class BuildStats:
    signals_planned: int = 0
    signals_built: int = 0
    signals_failed: int = 0
    windows_kept: int = 0
    windows_no_phases: int = 0
    windows_rejected: int = 0
    samples: int = 0
    per_class: dict[str, int] = field(default_factory=lambda: {name: 0 for name in PHASES})
    failures: list[str] = field(default_factory=list)


def plan(env_key: str, patients: list[str] | None, per_patient: int, seed: int,
         labeling_repo: str) -> pd.DataFrame:
    """Which signals this run would build, sampled per patient and spread across sessions."""
    labeling = bridge.require_labeling(labeling_repo)["building"]
    pool = labeling.eligible_signals(env_key, prefixes=patients)
    if pool.empty:
        return pool
    return labeling.sample_signals(pool, per_patient=per_patient, seed=seed)


def windows_of_signal(env_key: str, row: pd.Series, labeling_repo: str,
                      requires_rate: bool = True) -> list[dict[str, Any]]:
    """One signal's windows, each with the samples and production's phases on those samples.

    `building.windows_of` runs the pipeline and turns each window the right way up;
    `suggestion.suggest` then asks the detector what it calls **on that oriented trace**, so the
    spans and the samples describe the same picture. A window the detector cannot answer for is
    kept with an empty span list and counted, not dropped - "the algorithm found nothing here"
    is a training signal, and dropping it would bias the set towards easy breathing.
    """
    bundle = bridge.require_labeling(labeling_repo)
    windows = bundle["building"].windows_of(env_key, row)

    out = []
    for window in windows:
        values = np.asarray(window["values"], dtype=np.float32)
        rate = window.get("respiration_rate")
        spans, note = ([], "no stored rate") if (requires_rate and not rate) else \
            bundle["suggestion"].suggest(values, rate, window["analysis_fps"])
        out.append({**window, "values": values, "spans": spans, "note": note,
                    "target": spans_to_targets(spans, values.size)})
    return out


def _shard_arrays(signal_id: int, patient: str, session_id: int,
                  windows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """One .npz per signal: its windows concatenated, plus the offsets that cut them apart.

    Ragged on purpose. Windows differ in length - the pipeline grows one by 5 s and retries when
    it cannot find three breaths - and padding them to a common width would put invented samples
    in the training set.
    """
    values = np.concatenate([window["values"] for window in windows])
    targets = np.concatenate([window["target"] for window in windows])
    lengths = np.array([window["values"].size for window in windows], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    return {
        "values": values.astype(np.float32),
        "targets": targets.astype(np.int8),
        "offsets": offsets.astype(np.int64),
        "window_index": np.array([w["window_index"] for w in windows], dtype=np.int64),
        "start_index": np.array([w["start_index"] for w in windows], dtype=np.int64),
        "analysis_fps": np.array([w["analysis_fps"] for w in windows], dtype=np.float32),
        "respiration_rate": np.array([np.nan if w["respiration_rate"] is None
                                      else w["respiration_rate"] for w in windows],
                                     dtype=np.float32),
        "range_bin": np.array([-1 if w["range_bin"] is None else w["range_bin"]
                               for w in windows], dtype=np.int64),
        "rejected": np.array([w["rejected"] for w in windows], dtype=bool),
        "orientation": np.array([w.get("orientation", "") for w in windows]),
        "signal_id": np.int64(signal_id),
        "patient": np.str_(patient),
        "session_id": np.int64(session_id),
    }


def _manifest_rows(signal_id: int, patient: str, session_id: int, env_key: str,
                   windows: list[dict[str, Any]], shard: str) -> list[dict[str, Any]]:
    rows = []
    for position, window in enumerate(windows):
        counts = class_counts(window["target"])
        rows.append({
            "env": env_key,
            "shard": shard,
            "position": position,
            "RadarSignalID": signal_id,
            "SessionID": session_id,
            "PatientID": patient,
            "WindowIndex": window["window_index"],
            "start_index": window["start_index"],
            "samples": int(window["values"].size),
            "analysis_fps": window["analysis_fps"],
            "respiration_rate": window["respiration_rate"],
            "range_bin": window["range_bin"],
            "rejected": window["rejected"],
            "orientation": window.get("orientation", ""),
            "n_spans": len(window["spans"]),
            **{f"n_{name}": counts[name] for name in PHASES},
        })
    return rows


def build(env_key: str, patients: list[str] | None, per_patient: int, seed: int,
          out_dir: Path, labeling_repo: str, requires_rate: bool = True,
          limit: int | None = None, log=print) -> tuple[pd.DataFrame, BuildStats]:
    """Run the plan and write the shards. Resumable: a shard already on disk is left alone.

    Resumable for the same reason the labelling repo never rebuilds an uploaded signal -
    `fast_small_kmeans` draws from the unseeded global RNG, so a rebuild is a *different* trace.
    Re-running to extend a set must not silently replace the windows already in it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = plan(env_key, patients, per_patient, seed, labeling_repo)
    stats = BuildStats(signals_planned=0 if chosen.empty else len(chosen))
    if chosen.empty:
        log("no eligible signals for that selection")
        return pd.DataFrame(), stats
    if limit:
        chosen = chosen.head(limit)
        stats.signals_planned = len(chosen)

    log(f"{stats.signals_planned} signals over {chosen['PatientID'].nunique()} patients")
    rows: list[dict[str, Any]] = []
    started = time.time()
    for count, (_, row) in enumerate(chosen.iterrows(), start=1):
        signal_id = int(row["SignalID"])
        shard = SHARD_TEMPLATE.format(signal_id=signal_id)
        path = out_dir / shard
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                rows.extend(_manifest_from_shard(stored, env_key, shard))
            stats.signals_built += 1
            continue
        try:
            windows = windows_of_signal(env_key, row, labeling_repo, requires_rate)
        except Exception as error:                                        # noqa: BLE001
            stats.signals_failed += 1
            stats.failures.append(f"{signal_id}: {type(error).__name__}: {error}")
            log(f"  [{count}/{stats.signals_planned}] signal {signal_id} FAILED: {error}")
            continue

        usable = [window for window in windows if window["values"].size]
        if not usable:
            stats.signals_failed += 1
            stats.failures.append(f"{signal_id}: no windows reached the phase calculation")
            continue

        np.savez_compressed(path, **_shard_arrays(signal_id, str(row["PatientID"]),
                                                  int(row["SessionID"]), usable))
        rows.extend(_manifest_rows(signal_id, str(row["PatientID"]), int(row["SessionID"]),
                                   env_key, usable, shard))
        stats.signals_built += 1
        stats.windows_kept += len(usable)
        stats.windows_no_phases += sum(1 for w in usable if not w["spans"])
        stats.windows_rejected += sum(1 for w in usable if w["rejected"])
        for window in usable:
            stats.samples += int(window["values"].size)
            for name, value in class_counts(window["target"]).items():
                stats.per_class[name] += value
        if count % 25 == 0 or count == stats.signals_planned:
            rate = (time.time() - started) / count
            left = rate * (stats.signals_planned - count)
            log(f"  [{count}/{stats.signals_planned}] {stats.windows_kept} windows, "
                f"{rate:.1f}s/signal, ~{left/60:.0f} min left")

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest.to_csv(out_dir / MANIFEST_NAME, index=False)
    return manifest, stats


def _manifest_from_shard(stored, env_key: str, shard: str) -> list[dict[str, Any]]:
    """Rebuild a shard's manifest rows without re-running anything, for a resumed build."""
    offsets = stored["offsets"]
    targets = stored["targets"]
    rows = []
    for position in range(len(offsets) - 1):
        window_target = targets[offsets[position]:offsets[position + 1]]
        counts = class_counts(window_target.astype(np.int64))
        rate = float(stored["respiration_rate"][position])
        rows.append({
            "env": env_key, "shard": shard, "position": position,
            "RadarSignalID": int(stored["signal_id"]),
            "SessionID": int(stored["session_id"]),
            "PatientID": str(stored["patient"]),
            "WindowIndex": int(stored["window_index"][position]),
            "start_index": int(stored["start_index"][position]),
            "samples": int(offsets[position + 1] - offsets[position]),
            "analysis_fps": float(stored["analysis_fps"][position]),
            "respiration_rate": None if np.isnan(rate) else rate,
            "range_bin": int(stored["range_bin"][position]),
            "rejected": bool(stored["rejected"][position]),
            "orientation": str(stored["orientation"][position]),
            "n_spans": -1,                     # not stored; recoverable from the target
            **{f"n_{name}": counts[name] for name in PHASES},
        })
    return rows
