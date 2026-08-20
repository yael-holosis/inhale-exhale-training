"""Cut the dataset: a test set held out once, then folds that rotate validation inside the rest.

Shaped after `cough/data_utils.py`. One `split` column (train/test) plus one `fold_{i}_split`
column per fold (train/val/test), written into the dataset's own `windows.csv` - so the record of
which window went where survives the run that used it.

**Test is fixed across folds.** Every fold reports against the same held-out patients, which is
what makes fold numbers comparable and gives one headline result. The alternative - rotating the
test set - evaluates every patient once but leaves five numbers measured on five different
populations.

**Grouped by patient, always.** Windows stride 5 s across a 20 s span, so neighbours share most
of their breaths: a random split puts the same breath on both sides and reports nothing. A split
by signal is not enough either - one patient's night is one breathing pattern.

The group key is `(env, PatientID)`, not the name alone. `PatientID` is a *display* name -
`SL0066` from the algo instance, `bs-003` from production - and the two instances have unrelated
identity spaces. Nothing stops a name appearing on both, and if one did, the name alone would
merge two different people into one group and put half of each on both sides.

**Stratification is a configured list of columns**, one composite key per group, honoured by both
the test split and the fold split. `env` is the cohort, and it matters: the instances differ in
acquisition, so a split that ignores it puts folds on different populations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN, VAL, TEST = "train", "val", "test"
SPLIT_COLUMN = "split"
FOLD_COLUMN = "fold_{fold}_split"


def patient_key(frame: pd.DataFrame, columns) -> pd.Series:
    """The unit a split may not cut through, as one string."""
    present = [column for column in columns if column in frame.columns]
    if not present:
        raise KeyError(f"none of {tuple(columns)} is in the manifest")
    key = frame[present[0]].astype(str)
    for column in present[1:]:
        key = key + "/" + frame[column].astype(str)
    return key


def group_table(frame: pd.DataFrame, group_columns, stratify_cols) -> pd.DataFrame:
    """One row per patient: its key, its size in windows, and its stratification key.

    The stratification value is the group's most common - a patient sits on one instance and has
    one cohort, so the mode makes the rule total rather than assuming it.
    """
    keys = patient_key(frame, group_columns)
    rows = pd.DataFrame({"key": keys, "size": 1})
    table = rows.groupby("key", sort=True)["size"].sum().reset_index()
    if stratify_cols:
        labels = {}
        for column in stratify_cols:
            if column not in frame.columns:
                continue
            per_group = frame.groupby(keys)[column].agg(lambda v: str(v.mode().iat[0]))
            labels[column] = per_group
        table["stratum"] = table["key"].map(
            lambda key: "|".join(str(labels[c].get(key, "")) for c in labels)) if labels else ""
    else:
        table["stratum"] = ""
    return table


def _balanced_assignment(groups: pd.DataFrame, n_parts: int, weights, seed: int) -> dict[str, int]:
    """Assign each group to a part, keeping every stratum's window count near its target.

    Greedy, largest group first, each one going to whichever part is **furthest short** of its
    target in that group's stratum. Deficit-first, not nearest-target: scoring by distance from
    the final target sends every early group to the part with the smallest target, because being
    under is penalised like being over - which put 95% of the windows in a 20% test set.

    Chosen over `StratifiedGroupKFold` after measuring both. That splitter balances the number of
    *groups* per stratum, and our groups differ in size by more than an order of magnitude
    (SL0066 has 190 windows, bs-010 has 81), so on this data it produced folds ranging from 0% to
    58% `ds_prod` against an overall 32%, and fold sizes from 11.9% to 24.0%. This balances
    windows, which is what the metrics are averaged over.

    `weights` is the share each part should carry - `[0.2, 0.8]` for a test split, or equal
    shares for folds.
    """
    rng = np.random.default_rng(seed)
    order = groups.sample(frac=1.0, random_state=int(rng.integers(2 ** 31)))
    order = order.sort_values("size", ascending=False, kind="stable")

    strata = sorted(order["stratum"].unique())
    totals = {stratum: float(order.loc[order["stratum"] == stratum, "size"].sum())
              for stratum in strata}
    have = {part: {stratum: 0.0 for stratum in strata} for part in range(n_parts)}
    want = {part: {stratum: totals[stratum] * weights[part] for stratum in strata}
            for part in range(n_parts)}

    assignment: dict[str, int] = {}
    for row in order.itertuples():
        def shortfall(part: int) -> tuple[float, float]:
            # Primary: how far this part is from its target in *this* stratum, so a part short
            # on one cohort attracts that cohort's patients. Secondary: its overall shortfall,
            # which decides between parts that are equally short in the stratum.
            in_stratum = want[part][row.stratum] - have[part][row.stratum]
            overall = sum(want[part][s] - have[part][s] for s in strata)
            return (-in_stratum, -overall)

        best = min(range(n_parts), key=shortfall)
        have[best][row.stratum] += row.size
        assignment[row.key] = best
    return assignment


def assign_test(frame: pd.DataFrame, test_fraction: float, group_columns, stratify_cols,
                seed: int) -> pd.DataFrame:
    """Add the `split` column: train or test, by patient, once and for all folds."""
    frame = frame.copy()
    if test_fraction <= 0:
        frame[SPLIT_COLUMN] = TRAIN
        return frame
    groups = group_table(frame, group_columns, stratify_cols)
    if len(groups) < 2:
        raise ValueError(f"{len(groups)} patients cannot be split into train and test")
    parts = _balanced_assignment(groups, 2, [test_fraction, 1.0 - test_fraction], seed)
    keys = patient_key(frame, group_columns)
    frame[SPLIT_COLUMN] = np.where(keys.map(parts) == 0, TEST, TRAIN)
    return frame


def assign_folds(frame: pd.DataFrame, n_folds: int, group_columns, stratify_cols,
                 seed: int) -> pd.DataFrame:
    """Add `fold_{i}_split`: test stays test, and each fold holds out a slice of train as val."""
    frame = frame.copy()
    trainable = frame[frame[SPLIT_COLUMN] == TRAIN]
    groups = group_table(trainable, group_columns, stratify_cols)
    if len(groups) < n_folds:
        raise ValueError(f"{len(groups)} trainable patients cannot fill {n_folds} folds")

    parts = _balanced_assignment(groups, n_folds, [1.0 / n_folds] * n_folds, seed + 1)
    keys = patient_key(frame, group_columns)
    fold_of = keys.map(parts)
    for fold in range(n_folds):
        column = FOLD_COLUMN.format(fold=fold)
        frame[column] = np.where(frame[SPLIT_COLUMN] == TEST, TEST,
                                 np.where(fold_of == fold, VAL, TRAIN))
    return frame


def make_splits(frame: pd.DataFrame, cfg, ) -> pd.DataFrame:
    """Both passes, from the `data.split` config block."""
    frame = assign_test(frame, float(cfg["test_fraction"]), cfg["group_columns"],
                        cfg["stratify_cols"], int(cfg["seed"]))
    return assign_folds(frame, int(cfg["folds"]), cfg["group_columns"], cfg["stratify_cols"],
                        int(cfg["seed"]))


def split_for(frame: pd.DataFrame, fold: int) -> dict[str, pd.DataFrame]:
    """The three sides of one fold, read off the columns rather than recomputed."""
    column = FOLD_COLUMN.format(fold=fold)
    if column not in frame.columns:
        raise KeyError(f"{column} is not in the manifest - run make_splits.py on this dataset")
    return {name: frame[frame[column] == name] for name in (TRAIN, VAL, TEST)}


def describe(splits: dict[str, pd.DataFrame], total: int | None = None,
             stratify_cols=("env",)) -> str:
    """One line per split: size, share, stratum mix, and the patients in it."""
    total = total or sum(len(frame) for frame in splits.values())
    lines = []
    for name, frame in splits.items():
        if frame.empty:
            lines.append(f"  {name:5s} empty")
            continue
        patients = sorted(frame["PatientID"].astype(str).unique())
        mix = ""
        for column in stratify_cols or ():
            if column in frame.columns:
                counts = frame[column].value_counts()
                mix += "  " + " ".join(f"{value} {100 * n / len(frame):.0f}%"
                                       for value, n in counts.items())
        lines.append(f"  {name:5s} {len(frame):6d} windows ({100 * len(frame) / total:4.1f}%)  "
                     f"{frame['RadarSignalID'].nunique():4d} signals  "
                     f"{len(patients):3d} patients{mix}\n"
                     f"        {', '.join(patients)}")
    return "\n".join(lines)
