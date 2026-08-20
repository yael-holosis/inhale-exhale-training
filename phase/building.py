"""Build the training set from the windows already in `RespirationWindow`, labelled by the algorithm.

**No raw scan is read.** The app repo's `upload_windows.py` has already run the production
pipeline on the scans and stored each window it produced - the samples in S3, the row in
`RespirationWindow`. This repo reads those objects and asks production's own
`calculate_inhale_exhale_time` what it calls on them. That is the whole build: two reads and a
function call, no pipeline, no scan download, no disk pressure.

It also means the training windows are **the same objects a labeller sees**, byte for byte.
`RespirationWindowID` travels into the manifest, so a human label written later joins straight
onto the row the network was trained on - no re-derivation, no risk of the two describing
different samples.

To grow the set, upload more windows from the app repo:

    poetry run python upload_windows.py --env ds_algo --patients SL --per-patient 20 --commit

That is deliberately not done from here. Building a window and labelling a window are that
repo's job; this one consumes what it produced.

## What the labels are, and are not

`suggestion.suggest` is production's answer mapped back onto the stored trace. Two properties
carry into every number measured on this set:

- **The boundaries are the 10% and 90% amplitude crossings** - rise and fall times, a median 65%
  of the true trough-to-crest rise (`inhale-exhale-detection/claude/FINDINGS.md`, finding 5).
- **The ceiling is the teacher.** Where the algorithm is wrong, the target is wrong. Only the
  human spans in `BreathPhaseTimeRecord` measure either of them.

`unknown` also means two things here. A labeller marks it where they could not call the trace;
production leaves it in the same places *and* at the turn from inhale to exhale on every breath,
because it has no phase for that turn. Anything reading `unknown` as "no breathing" will be
wrong most of the time it fires.
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
SUMMARY_NAME = "build_summary.yaml"
SHARD_TEMPLATE = "{env}_signal_{signal_id}.npz"
"""Environment first, and it has to be. `ds_algo` and `ds_prod` have unrelated `RadarSignal` ID
spaces - signal 2120091 is a different recording on each - so one filename would silently
overwrite the other's samples. The app repo prefixes its S3 window keys for the same reason."""


@dataclass
class BuildStats:
    signals_planned: int = 0
    signals_built: int = 0
    signals_failed: int = 0
    windows_kept: int = 0
    windows_no_phases: int = 0
    samples: int = 0
    failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------- selection

def catalogue(env_key: str, app_repo: str) -> pd.DataFrame:
    """Every uploaded window with its signal and patient, as the app's sidebar sees it.

    `db.browse` is the association: `RespirationWindow` on the labels instance merged in pandas
    (never in SQL - on prod they are different servers) with `RadarSignal`/`Session` on the
    device instance, keyed on `RadarSignalID`. The patient's display name is per environment -
    a `PatientStudyName` from the labels database on `ds_algo` (`SL0066`), and
    `Session.PatientID` on `ds_prod` (`bs-008`), which has no name column at all.

    Adds `Spans`, the number of human phase records on the window. Zero is the normal case.
    """
    frame = bridge.require_app(app_repo)["db"].browse(env_key).copy()
    frame["Spans"] = frame["Spans"].fillna(0).astype(int)
    return frame


def select(catalogue_frame: pd.DataFrame, patients: list[str] | None = None,
           signals: list[int] | None = None, per_patient: int | None = None,
           unlabelled_only: bool = False, seed: int = 0) -> pd.DataFrame:
    """Narrow the catalogue. Signal-level, because a shard is a signal.

    `per_patient` caps how many **signals** a patient contributes, sampled seeded and spread
    across sessions so one long night does not dominate. `unlabelled_only` holds back the
    windows a person has already labelled, so they can serve as a clean test set.
    """
    frame = catalogue_frame
    if patients:
        wanted = tuple(patients)
        keep = (frame["Patient"].astype(str).str.startswith(wanted)
                | frame["PatientKey"].astype(str).str.startswith(wanted))
        frame = frame[keep]
    if signals:
        frame = frame[frame["RadarSignalID"].isin([int(value) for value in signals])]
    if unlabelled_only:
        labelled = frame.loc[frame["Spans"] > 0, "RadarSignalID"].unique()
        frame = frame[~frame["RadarSignalID"].isin(labelled)]
    if not per_patient:
        return frame

    taken = []
    for patient, rows in frame.groupby("Patient", sort=True):
        by_signal = rows.drop_duplicates("RadarSignalID")[["RadarSignalID", "SessionID"]]
        rng = np.random.default_rng(_patient_seed(seed, str(patient)))
        chosen, sessions = [], {session: list(group["RadarSignalID"])
                                for session, group in by_signal.groupby("SessionID", sort=True)}
        order = list(sessions)
        rng.shuffle(order)
        for values in sessions.values():
            rng.shuffle(values)
        while len(chosen) < per_patient and any(sessions.values()):
            for session in order:                    # one per session, then round again
                if sessions[session] and len(chosen) < per_patient:
                    chosen.append(sessions[session].pop())
        taken.append(rows[rows["RadarSignalID"].isin(chosen)])
    return pd.concat(taken) if taken else frame.iloc[0:0]


def _patient_seed(seed: int, patient: str) -> int:
    """A digest, not `hash()` - Python randomises string hashing per process, so `--seed` would
    otherwise draw a different sample on every invocation."""
    import hashlib

    return int.from_bytes(hashlib.sha256(f"{int(seed)}:{patient}".encode()).digest()[:8],
                          "big") % (2 ** 32)


# ---------------------------------------------------------------------------------- one signal

def windows_of_signal(env_key: str, signal_id: int, app_repo: str,
                      requires_rate: bool = True) -> list[dict[str, Any]]:
    """One signal's stored windows, each with its samples and the algorithm's phases on them.

    The samples are read as stored, never re-oriented: the blob is already the trace the app
    shows and `ReviewerFlipped` describes that object, so turning it over here would put the
    labels on a picture nobody has seen.

    A window the detector cannot answer for is **kept** with an empty span list and counted, not
    dropped - "the algorithm found nothing here" is a training signal, and dropping those windows
    would bias the set towards easy breathing.
    """
    bundle = bridge.require_app(app_repo)
    rows = bundle["db"].windows_for_signal(env_key, int(signal_id))
    out = []
    for _, row in rows.iterrows():
        t_sec, values = bundle["waveforms"].window_samples(str(row["WaveformS3Path"]))
        values = np.asarray(values, dtype=np.float32)
        if not values.size:
            continue
        rate = None if pd.isna(row.get("RespirationRate")) else float(row["RespirationRate"])
        fps = float(row["AnalysisFps"])
        spans, note = ([], "no stored rate") if (requires_rate and not rate) else \
            bundle["suggestion"].suggest(values, rate, fps)
        out.append({
            "window_id": int(row["ID"]),
            "window_index": int(row["WindowIndex"]),
            "start_index": int(row["StartIndex"]),
            "analysis_fps": fps,
            "respiration_rate": rate,
            "range_bin": None if pd.isna(row.get("RangeBin")) else int(row["RangeBin"]),
            "reviewer_flipped": bool(row.get("ReviewerFlipped", False)),
            "system_version": "" if pd.isna(row.get("SystemVersion")) else
                              str(row["SystemVersion"]),
            "spans": spans, "note": note,
            "t_sec": np.asarray(t_sec, dtype=np.float32),
            "values": values,
            "target": spans_to_targets(spans, values.size),
        })
    return out


def _shard_arrays(env_key: str, signal_id: int, patient: str, patient_key: str,
                  session_id: int, windows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """One .npz per signal: its windows concatenated, plus the offsets that cut them apart.

    Ragged on purpose - windows differ in length, and padding them to a common width would put
    invented samples in the training set.
    """
    lengths = np.array([window["values"].size for window in windows], dtype=np.int64)
    return {
        "values": np.concatenate([w["values"] for w in windows]).astype(np.float32),
        "targets": np.concatenate([w["target"] for w in windows]).astype(np.int8),
        "offsets": np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        "window_id": np.array([w["window_id"] for w in windows], dtype=np.int64),
        "window_index": np.array([w["window_index"] for w in windows], dtype=np.int64),
        "start_index": np.array([w["start_index"] for w in windows], dtype=np.int64),
        "analysis_fps": np.array([w["analysis_fps"] for w in windows], dtype=np.float32),
        "respiration_rate": np.array([np.nan if w["respiration_rate"] is None
                                      else w["respiration_rate"] for w in windows],
                                     dtype=np.float32),
        "range_bin": np.array([-1 if w["range_bin"] is None else w["range_bin"]
                               for w in windows], dtype=np.int64),
        "reviewer_flipped": np.array([w["reviewer_flipped"] for w in windows], dtype=bool),
        "n_spans": np.array([len(w["spans"]) for w in windows], dtype=np.int64),
        "system_version": np.array([w["system_version"] for w in windows]),
        "signal_id": np.int64(signal_id),
        "patient": np.str_(patient),
        "patient_key": np.str_(patient_key),
        "session_id": np.int64(session_id),
        "env": np.str_(env_key),
    }


# --------------------------------------------------------------------------------- the build

def build(env_key: str, out_dir: Path, app_repo: str, patients: list[str] | None = None,
          signals: list[int] | None = None, per_patient: int | None = None,
          unlabelled_only: bool = False, seed: int = 0, requires_rate: bool = True,
          limit: int | None = None, log=print) -> tuple[pd.DataFrame, BuildStats]:
    """Read the selected signals' windows and write one shard each.

    Resumable: a shard already on disk is left alone. The window blobs are immutable once
    uploaded - the app repo refuses to overwrite one, because the labels already made against it
    would then describe samples that are not there - so a resumed build cannot mix two versions
    of a window. It re-reads nothing it already has.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = select(catalogue(env_key, app_repo), patients, signals, per_patient,
                    unlabelled_only, seed)
    if chosen.empty:
        log("no uploaded windows match that selection")
        return rebuild_manifest(out_dir), BuildStats()

    by_signal = chosen.drop_duplicates("RadarSignalID")
    if limit:
        by_signal = by_signal.head(limit)
    stats = BuildStats(signals_planned=len(by_signal))
    log(f"{len(chosen)} windows over {stats.signals_planned} signals, "
        f"{chosen['Patient'].nunique()} patients")

    started = time.time()
    for count, (_, row) in enumerate(by_signal.iterrows(), start=1):
        signal_id = int(row["RadarSignalID"])
        shard = SHARD_TEMPLATE.format(env=env_key, signal_id=signal_id)
        if (out_dir / shard).exists():
            stats.signals_built += 1
            continue
        try:
            windows = windows_of_signal(env_key, signal_id, app_repo, requires_rate)
        except Exception as error:                                        # noqa: BLE001
            stats.signals_failed += 1
            stats.failures.append(f"{signal_id}: {type(error).__name__}: {error}")
            log(f"  [{count}/{stats.signals_planned}] signal {signal_id} FAILED: {error}")
            continue
        if not windows:
            stats.signals_failed += 1
            stats.failures.append(f"{signal_id}: no window carried samples")
            continue

        np.savez_compressed(out_dir / shard,
                            **_shard_arrays(env_key, signal_id, str(row["Patient"]),
                                            str(row["PatientKey"]), int(row["SessionID"]),
                                            windows))
        stats.signals_built += 1
        stats.windows_kept += len(windows)
        stats.windows_no_phases += sum(1 for w in windows if not w["spans"])
        stats.samples += sum(int(w["values"].size) for w in windows)
        if count % 25 == 0 or count == stats.signals_planned:
            rate = (time.time() - started) / count
            log(f"  [{count}/{stats.signals_planned}] {stats.windows_kept} windows, "
                f"{rate:.1f}s/signal, ~{rate * (stats.signals_planned - count) / 60:.0f} min left")

    # Rebuilt from every shard in the directory, not from this run's selection. A second run -
    # the other environment, or more patients - must extend the set rather than replace its index
    # with only what it happened to touch.
    manifest = rebuild_manifest(out_dir)
    if not manifest.empty:
        manifest.to_csv(out_dir / MANIFEST_NAME, index=False)
    return manifest, stats


def rebuild_manifest(out_dir: Path) -> pd.DataFrame:
    """The index, read back off the shards. Cheap - only the small arrays are touched."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(out_dir).glob("*_signal_*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            rows.extend(_manifest_from_shard(stored, path.name))
    return pd.DataFrame(rows)


def _manifest_from_shard(stored, shard: str) -> list[dict[str, Any]]:
    """A shard's manifest rows. One per window, and `RespirationWindowID` is the join key.

    That column is what lets a human label written months from now be matched to the exact
    window the network trained on, rather than to a re-derived approximation of it.
    """
    offsets, targets = stored["offsets"], stored["targets"]
    rows = []
    for position in range(len(offsets) - 1):
        counts = class_counts(targets[offsets[position]:offsets[position + 1]].astype(np.int64))
        rate = float(stored["respiration_rate"][position])
        rows.append({
            "env": str(stored["env"]), "shard": shard, "position": position,
            "RespirationWindowID": int(stored["window_id"][position]),
            "RadarSignalID": int(stored["signal_id"]),
            "SessionID": int(stored["session_id"]),
            "PatientID": str(stored["patient"]),
            "PatientKey": str(stored["patient_key"]),
            "WindowIndex": int(stored["window_index"][position]),
            "start_index": int(stored["start_index"][position]),
            "samples": int(offsets[position + 1] - offsets[position]),
            "analysis_fps": float(stored["analysis_fps"][position]),
            "respiration_rate": None if np.isnan(rate) else rate,
            "range_bin": int(stored["range_bin"][position]),
            "reviewer_flipped": bool(stored["reviewer_flipped"][position]),
            "system_version": str(stored["system_version"][position]),
            "n_spans": int(stored["n_spans"][position]),
            **{f"n_{name}": counts[name] for name in PHASES},
        })
    return rows
