"""Build a dataset: one directory per build, carrying the windows, their labels and its own provenance.

A dataset directory is self-describing on purpose. Six months from now the only question that
matters about a checkpoint is what it was trained on, and the answer has to be readable off disk
rather than reconstructed from a config file that has moved on:

    data_sets/phases_algorithm_20260820T165400Z/
      build_params.yaml            every parameter that decided this dataset, plus versions
      windows.csv                  one row per window: provenance, class counts, split columns
      stats.yaml                   hours, class balance, per-patient and per-cohort counts
      ds_algo_signal_1557424.npz   the samples and the per-sample targets, one file per signal

**No raw scan is read.** The pipeline already ran when these windows were uploaded; this reads
each window's samples from S3 and labels them from whichever source `data.labels.source` names -
see `phase/labelsources.py` for what each one is and is not.

**A directory is one label source.** Extending a build with `--into` refuses to mix them: an
algorithm-labelled window and a human-labelled one are different targets, and a set that silently
contained both would train a model against a moving definition.

Every row carries `RespirationWindowID`, so the same window can be found in the database, in the
labelling app, and in this dataset without re-deriving anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from phase import sources
from phase.labelsources import LabelSource
from phase.labels import PHASES, class_counts, spans_to_targets

WINDOWS_NAME = "windows.csv"
PARAMS_NAME = "build_params.yaml"
STATS_NAME = "stats.yaml"
SHARD_TEMPLATE = "{env}_signal_{signal_id}.npz"
"""Environment first, and it has to be. `ds_algo` and `ds_prod` have unrelated `RadarSignal` ID
spaces - signal 2120091 is a different recording on each - so one filename would silently
overwrite the other's samples."""

DIR_TEMPLATE = "{name}_{source}_{stamp}"
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


@dataclass
class BuildStats:
    signals_planned: int = 0
    signals_built: int = 0
    signals_failed: int = 0
    windows_kept: int = 0
    windows_unlabelled: int = 0
    samples: int = 0
    failures: list[str] = field(default_factory=list)


def stamp() -> str:
    """UTC, because the two instances run their clocks in UTC and a local stamp would not sort."""
    return datetime.now(timezone.utc).strftime(STAMP_FORMAT)


def dataset_dir(root: str | Path, name: str, source: str, when: str | None = None) -> Path:
    return Path(root) / DIR_TEMPLATE.format(name=name, source=source, stamp=when or stamp())


LATEST = "latest"


def built_at(directory: Path) -> datetime:
    """The build stamp out of a directory name, for ordering. Unparseable sorts oldest.

    Parsed rather than taken from the whole name: the name begins with the label source, so
    sorting the strings puts every `phases_human_...` after every `phases_algorithm_...`
    whatever their timestamps, and `latest` silently means "human". Not mtime either - that
    moves whenever anything in the directory is rewritten, `make_splits.py` included.
    """
    stamp_part = directory.name.rsplit("_", 1)[-1]
    try:
        return datetime.strptime(stamp_part, STAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def resolve(root: str | Path, wanted: str, source: str | None = None) -> Path:
    """A dataset directory by name, or `latest` for the newest build under `root`.

    `source` narrows `latest` to one label source, so a `human` build sitting beside an
    `algorithm` one does not become what everything trains on by accident.
    """
    root = Path(root)
    if wanted and wanted != LATEST:
        named = Path(wanted)
        return named if named.is_absolute() or named.exists() else root / wanted
    candidates = [path for path in root.glob("*") if (path / WINDOWS_NAME).exists()]
    if source:
        candidates = [path for path in candidates if f"_{source}_" in path.name]
    if not candidates:
        raise FileNotFoundError(
            f"no dataset under {root}" + (f" built with labels.source={source!r}" if source
                                          else "") + " - run build_dataset.py first")
    return max(candidates, key=built_at)


def existing_params(directory: Path) -> dict[str, Any]:
    path = Path(directory) / PARAMS_NAME
    if not path.exists():
        return {}
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


# --------------------------------------------------------------------------------- selection

def catalogue(env_key: str) -> pd.DataFrame:
    """Every uploaded window with its signal and patient. See `phase.sources.catalogue`."""
    return sources.catalogue(env_key)


def select(frame: pd.DataFrame, patients: list[str] | None = None,
           signals: list[int] | None = None, per_patient: int | None = None,
           seed: int = 0) -> pd.DataFrame:
    """Narrow the catalogue. Signal-level, because a shard is a signal."""
    if patients:
        wanted = tuple(patients)
        frame = frame[frame["Patient"].astype(str).str.startswith(wanted)
                      | frame["PatientKey"].astype(str).str.startswith(wanted)]
    if signals:
        frame = frame[frame["RadarSignalID"].isin([int(value) for value in signals])]
    if not per_patient:
        return frame

    taken = []
    for patient, rows in frame.groupby("Patient", sort=True):
        by_signal = rows.drop_duplicates("RadarSignalID")[["RadarSignalID", "SessionID"]]
        rng = np.random.default_rng(_patient_seed(seed, str(patient)))
        sessions = {session: list(group["RadarSignalID"])
                    for session, group in by_signal.groupby("SessionID", sort=True)}
        order = list(sessions)
        rng.shuffle(order)
        for values in sessions.values():
            rng.shuffle(values)
        chosen: list[int] = []
        # One signal per session, then round again - a patient with a thousand sessions must not
        # contribute a thousand near-identical minutes of one night.
        while len(chosen) < per_patient and any(sessions.values()):
            for session in order:
                if sessions[session] and len(chosen) < per_patient:
                    chosen.append(sessions[session].pop())
        taken.append(rows[rows["RadarSignalID"].isin(chosen)])
    return pd.concat(taken) if taken else frame.iloc[0:0]


def _patient_seed(seed: int, patient: str) -> int:
    """A digest, not `hash()` - Python randomises string hashing per process, so a seed would
    otherwise draw a different sample on every invocation."""
    import hashlib

    return int.from_bytes(hashlib.sha256(f"{int(seed)}:{patient}".encode()).digest()[:8],
                          "big") % (2 ** 32)


# ---------------------------------------------------------------------------------- one signal

def windows_of_signal(rows: pd.DataFrame, labels: LabelSource) -> list[dict[str, Any]]:
    """One signal's stored windows, each with its samples and its per-sample target.

    Samples are read as stored, never re-oriented: the blob is already the trace the labelling
    app shows and `ReviewerFlipped` describes *that object*, so turning it over here would put
    the labels on a picture nobody has seen.

    A window the source cannot label is **kept**, empty, and counted - except under `human`,
    where `eligible` has already removed the unlabelled ones. "The algorithm found nothing here"
    is a training signal; dropping those windows would bias the set towards easy breathing.
    """
    out = []
    for _, row in rows.sort_values("WindowIndex").iterrows():
        _, values = sources.window_samples(str(row["WaveformS3Path"]))
        values = np.asarray(values, dtype=np.float32)
        if not values.size:
            continue
        spans, note = labels.rows_for(row, values)
        out.append({
            "window_id": int(row["ID"]),
            "window_index": int(row["WindowIndex"]),
            "start_index": int(row["StartIndex"]),
            "analysis_fps": float(row["AnalysisFps"]),
            "respiration_rate": (None if pd.isna(row.get("RespirationRate"))
                                 else float(row["RespirationRate"])),
            "range_bin": None if pd.isna(row.get("RangeBin")) else int(row["RangeBin"]),
            "reviewer_flipped": bool(row.get("ReviewerFlipped", False)),
            "system_version": ("" if pd.isna(row.get("SystemVersion"))
                               else str(row["SystemVersion"])),
            "human_spans": int(row.get("Spans", 0) or 0),
            "spans": spans, "note": note, "values": values,
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
        "human_spans": np.array([w["human_spans"] for w in windows], dtype=np.int64),
        "system_version": np.array([w["system_version"] for w in windows]),
        "signal_id": np.int64(signal_id),
        "patient": np.str_(patient),
        "patient_key": np.str_(patient_key),
        "session_id": np.int64(session_id),
        "env": np.str_(env_key),
    }


# ------------------------------------------------------------------------------------ the build

def build(env_key: str, out_dir: Path, labels: LabelSource, patients: list[str] | None = None,
          signals: list[int] | None = None, per_patient: int | None = None, seed: int = 0,
          limit: int | None = None, log=print) -> tuple[pd.DataFrame, BuildStats]:
    """Read the selected signals' windows and write one shard each.

    Resumable: a shard already on disk is left alone. A window blob is immutable once uploaded,
    so a shard can never be stale - only absent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = labels.eligible(select(catalogue(env_key), patients, signals, per_patient, seed))
    if chosen.empty:
        log(f"no window on {env_key} matches that selection and can be labelled by "
            f"`{labels.source}`")
        return read_windows(out_dir), BuildStats()

    by_signal = chosen.drop_duplicates("RadarSignalID")
    if limit:
        by_signal = by_signal.head(limit)
    stats = BuildStats(signals_planned=len(by_signal))
    log(f"{len(chosen)} windows over {stats.signals_planned} signals, "
        f"{chosen['Patient'].nunique()} patients, labelled by `{labels.source}`")

    started = time.time()
    for count, (_, row) in enumerate(by_signal.iterrows(), start=1):
        signal_id = int(row["RadarSignalID"])
        shard = SHARD_TEMPLATE.format(env=env_key, signal_id=signal_id)
        if (out_dir / shard).exists():
            stats.signals_built += 1
            continue
        try:
            windows = windows_of_signal(chosen[chosen["RadarSignalID"] == signal_id], labels)
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
        stats.windows_unlabelled += sum(1 for w in windows if not w["spans"])
        stats.samples += sum(int(w["values"].size) for w in windows)
        if count % 25 == 0 or count == stats.signals_planned:
            rate = (time.time() - started) / count
            log(f"  [{count}/{stats.signals_planned}] {stats.windows_kept} windows, "
                f"{rate:.1f}s/signal, ~{rate * (stats.signals_planned - count) / 60:.0f} min left")

    windows_frame = read_windows(out_dir)
    if not windows_frame.empty:
        write_windows(out_dir, windows_frame)
    return windows_frame, stats


def write_windows(out_dir: Path, frame: pd.DataFrame) -> Path:
    path = Path(out_dir) / WINDOWS_NAME
    frame.to_csv(path, index=False)
    return path


def load_windows(directory: Path) -> pd.DataFrame:
    """`windows.csv` as written, split columns included if `make_splits.py` has run."""
    path = Path(directory) / WINDOWS_NAME
    if not path.exists():
        raise FileNotFoundError(f"no {WINDOWS_NAME} in {directory} - run build_dataset.py first")
    return pd.read_csv(path)


def read_windows(out_dir: Path) -> pd.DataFrame:
    """Rebuild the index off the shards, preserving any split columns already assigned.

    Read from the directory rather than from the run's own selection, so a second run - the other
    environment, more patients - extends the set instead of replacing its index with only what it
    happened to touch. Splits are carried over by `RespirationWindowID`: extending a dataset must
    not silently drop the record of where the existing windows went.
    """
    out_dir = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*_signal_*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            rows.extend(_rows_of_shard(stored, path.name))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    previous = out_dir / WINDOWS_NAME
    if previous.exists():
        stored = pd.read_csv(previous)
        carried = [c for c in stored.columns if c == "split" or c.endswith("_split")]
        if carried:
            frame = frame.merge(stored[["RespirationWindowID", *carried]],
                                on="RespirationWindowID", how="left")
    return frame


def _rows_of_shard(stored, shard: str) -> list[dict[str, Any]]:
    """A shard's rows. `RespirationWindowID` is the join key back to the database."""
    offsets, targets = stored["offsets"], stored["targets"]
    rows = []
    for position in range(len(offsets) - 1):
        counts = class_counts(targets[offsets[position]:offsets[position + 1]].astype(np.int64))
        rate = float(stored["respiration_rate"][position])
        labelled = int(stored["n_spans"][position])
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
            "n_spans": labelled,
            "labelled": labelled > 0,
            # A snapshot at build time, unlike every other column here: it counts the human spans
            # on this window when it was read, and labelling continues afterwards.
            "human_spans": int(stored["human_spans"][position])
                           if "human_spans" in stored else 0,
            **{f"n_{name}": counts[name] for name in PHASES},
        })
    return rows


# ------------------------------------------------------------------------------------ provenance

def summarise(frame: pd.DataFrame) -> dict[str, Any]:
    """`stats.yaml`: what is in this dataset, in the terms anybody would ask about it."""
    per_class = {name: int(frame[f"n_{name}"].sum()) for name in PHASES}
    samples = max(sum(per_class.values()), 1)
    out = {
        "windows": int(len(frame)),
        "signals": int(frame["RadarSignalID"].nunique()),
        "patients": int(frame["PatientID"].nunique()),
        "environments": sorted(frame["env"].astype(str).unique()),
        "samples": samples,
        "hours": round(samples / 10.0 / 3600.0, 2),
        "windows_unlabelled": int((~frame["labelled"]).sum()),
        "windows_with_human_spans": int((frame["human_spans"] > 0).sum()),
        "per_class": per_class,
        "per_class_fraction": {name: round(count / samples, 4)
                               for name, count in per_class.items()},
        "per_environment": {str(env): int(n) for env, n in frame["env"].value_counts().items()},
        "windows_per_patient": {str(patient): int(n) for patient, n
                                in frame["PatientID"].value_counts().items()},
    }
    for column in [c for c in frame.columns if c == "split" or c.endswith("_split")]:
        out.setdefault("splits", {})[column] = {
            str(value): int(n) for value, n in frame[column].value_counts().items()}
    return out


def write_provenance(out_dir: Path, params: dict[str, Any], frame: pd.DataFrame) -> None:
    """`build_params.yaml` and `stats.yaml`, rewritten whole on every run.

    `build_params.yaml` accumulates a `runs` list, so a directory extended with a second
    environment records both invocations rather than only the last.
    """
    import copy

    out_dir = Path(out_dir)
    existing = existing_params(out_dir)
    runs = list(existing.get("runs", []))
    runs.append(params)
    # Deep-copied before dumping: `safe_dump` emits a YAML anchor for an object it has already
    # seen, so the labels block would come out as `&id001` at the top and `*id001` in the run -
    # valid, and unreadable.
    document = copy.deepcopy({"labels": params["labels"],
                              "created": existing.get("created") or params["started"],
                              "runs": runs})
    with open(out_dir / PARAMS_NAME, "w") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)
    with open(out_dir / STATS_NAME, "w") as handle:
        yaml.safe_dump(summarise(frame), handle, sort_keys=False)
