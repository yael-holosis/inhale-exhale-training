"""Splits, grouped by patient - the only defensible unit here.

Windows overlap: the pipeline strides 5 s across a 20 s window, so neighbouring windows share
most of their breaths. A random split over windows puts the same breath on both sides and
reports a number that means nothing. A split by signal is not enough either - one patient's
night is one breathing pattern, and a model that has seen it is not being tested on a new
subject when it sees more of it.

So: group by `PatientID`, and report which patients are on which side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def patient_folds(manifest: pd.DataFrame, n_folds: int, seed: int = 0) -> pd.DataFrame:
    """Add a `fold` column: each patient wholly inside one fold.

    `GroupKFold` balances fold sizes rather than shuffling, so the patient order is randomised
    first with the seed - otherwise the same alphabetical patients land together every run.
    """
    manifest = manifest.copy()
    patients = manifest["PatientID"].astype(str)
    unique = np.array(sorted(patients.unique()))
    if len(unique) < n_folds:
        raise ValueError(f"{len(unique)} patients cannot fill {n_folds} folds")

    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(len(unique))]
    order = {patient: position for position, patient in enumerate(shuffled)}
    keyed = patients.map(order).to_numpy()

    manifest["fold"] = -1
    splitter = GroupKFold(n_splits=n_folds)
    for fold, (_, held) in enumerate(splitter.split(keyed, groups=keyed)):
        manifest.iloc[held, manifest.columns.get_loc("fold")] = fold
    return manifest


def split_for(manifest: pd.DataFrame, fold: int,
              val_fraction: float = 0.0, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Train / val / test for one fold. Validation is carved out by patient too.

    With `val_fraction` 0 the validation set *is* the held-out fold, which is honest as long as
    nothing is selected on it. Anything that picks a checkpoint or a threshold needs a real
    third split - pass a fraction and the patients come out of train, never out of test.
    """
    test = manifest[manifest["fold"] == fold]
    rest = manifest[manifest["fold"] != fold]
    if val_fraction <= 0:
        return {"train": rest, "val": test, "test": test}

    patients = np.array(sorted(rest["PatientID"].astype(str).unique()))
    rng = np.random.default_rng(seed + fold)
    n_val = max(1, int(round(len(patients) * val_fraction)))
    held = set(patients[rng.permutation(len(patients))][:n_val])
    in_val = rest["PatientID"].astype(str).isin(held)
    return {"train": rest[~in_val], "val": rest[in_val], "test": test}


def describe(splits: dict[str, pd.DataFrame]) -> str:
    lines = []
    for name, frame in splits.items():
        patients = sorted(frame["PatientID"].astype(str).unique())
        lines.append(f"  {name:5s} {len(frame):6d} windows  {frame['RadarSignalID'].nunique():5d} "
                     f"signals  {len(patients):3d} patients  {', '.join(patients[:8])}"
                     + (" ..." if len(patients) > 8 else ""))
    return "\n".join(lines)
