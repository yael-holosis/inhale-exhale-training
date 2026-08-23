"""The split: a test set held out once, folds that rotate validation, and no patient in two places.

Everything here runs on a synthetic set with two cohorts, so stratification has something to
balance.
"""

import numpy as np
import pandas as pd
import pytest

from tests import synthetic
from phase.splits import (FOLD_COLUMN, SPLIT_COLUMN, assign_folds, assign_test, group_table,
                          make_splits, patient_key, split_for)

GROUPS = ["env", "PatientID"]
STRATIFY = ["env"]
CONFIG = {"test_fraction": 0.2, "folds": 5, "seed": 7,
          "group_columns": GROUPS, "stratify_cols": STRATIFY}


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    root = tmp_path_factory.mktemp("splits")
    return synthetic.build(root, n_patients=20, signals_per_patient=2, windows_per_signal=3,
                           environments=("ds_algo", "ds_prod"))


@pytest.fixture(scope="module")
def split(frame):
    return make_splits(frame, CONFIG)


# ------------------------------------------------------------------ the group key

def test_the_group_key_includes_the_environment(frame):
    keys = patient_key(frame, GROUPS)
    assert keys.str.contains("/").all()
    assert keys.nunique() == frame.groupby(["env", "PatientID"]).ngroups


def test_identically_named_patients_on_two_instances_stay_separate():
    """`PatientID` is a display name and the instances have unrelated identity spaces.

    Constructed, because no collision exists in the real data - which is exactly why one would go
    unnoticed.
    """
    rows = [{"env": env, "PatientID": "shared-name", "RadarSignalID": i}
            for i, env in enumerate(("ds_algo", "ds_prod"))]
    assert patient_key(pd.DataFrame(rows), GROUPS).nunique() == 2


def test_a_group_carries_one_stratum(frame):
    table = group_table(frame, GROUPS, STRATIFY)
    assert set(table["stratum"]) <= {"ds_algo", "ds_prod"}
    assert len(table) == frame.groupby(["env", "PatientID"]).ngroups


# ------------------------------------------------------------------ the test split

def test_test_is_about_the_requested_fraction(frame):
    out = assign_test(frame, 0.2, GROUPS, STRATIFY, seed=7)
    share = (out[SPLIT_COLUMN] == "test").mean()
    assert 0.1 < share < 0.32, f"test is {share:.0%} of windows"


def test_no_patient_is_in_both_train_and_test(split):
    train = set(patient_key(split[split[SPLIT_COLUMN] == "train"], GROUPS))
    test = set(patient_key(split[split[SPLIT_COLUMN] == "test"], GROUPS))
    assert not train & test


def test_stratification_keeps_the_cohort_mix(split):
    overall = (split["env"] == "ds_prod").mean()
    held = split[split[SPLIT_COLUMN] == "test"]
    assert abs((held["env"] == "ds_prod").mean() - overall) < 0.25


def test_a_test_fraction_of_zero_keeps_everything_trainable(frame):
    out = assign_test(frame, 0.0, GROUPS, STRATIFY, seed=7)
    assert (out[SPLIT_COLUMN] == "train").all()


# ------------------------------------------------------------------ the folds

def test_test_is_the_same_windows_in_every_fold(split):
    """The property that makes fold numbers comparable."""
    held = set(split.loc[split[SPLIT_COLUMN] == "test", "RespirationWindowID"])
    for fold in range(CONFIG["folds"]):
        column = FOLD_COLUMN.format(fold=fold)
        assert set(split.loc[split[column] == "test", "RespirationWindowID"]) == held


def test_every_trainable_patient_is_validation_exactly_once(split):
    trainable = split[split[SPLIT_COLUMN] == "train"]
    keys = patient_key(trainable, GROUPS)
    counted = {key: 0 for key in keys.unique()}
    for fold in range(CONFIG["folds"]):
        column = FOLD_COLUMN.format(fold=fold)
        for key in patient_key(trainable[trainable[column] == "val"], GROUPS).unique():
            counted[key] += 1
    assert set(counted.values()) == {1}


def test_the_three_sides_of_a_fold_are_disjoint_by_patient(split):
    for fold in range(CONFIG["folds"]):
        parts = split_for(split, fold)
        keys = {name: set(patient_key(part, GROUPS)) for name, part in parts.items()}
        assert not keys["train"] & keys["val"]
        assert not keys["train"] & keys["test"]
        assert not keys["val"] & keys["test"]


def test_the_three_sides_of_a_fold_cover_every_window(split):
    for fold in range(CONFIG["folds"]):
        parts = split_for(split, fold)
        assert sum(len(part) for part in parts.values()) == len(split)


def test_no_signal_crosses_a_fold_boundary(split):
    for fold in range(CONFIG["folds"]):
        parts = split_for(split, fold)
        assert not set(parts["train"]["RadarSignalID"]) & set(parts["test"]["RadarSignalID"])
        assert not set(parts["train"]["RadarSignalID"]) & set(parts["val"]["RadarSignalID"])


def test_folds_are_roughly_even(split):
    sizes = [len(split_for(split, fold)["val"]) for fold in range(CONFIG["folds"])]
    assert max(sizes) <= 2.5 * min(sizes), f"validation folds vary too much: {sizes}"


# ------------------------------------------------------------------ failure modes

def test_more_folds_than_patients_is_refused(frame):
    out = assign_test(frame, 0.2, GROUPS, STRATIFY, seed=7)
    with pytest.raises(ValueError):
        assign_folds(out, 99, GROUPS, STRATIFY, seed=7)


def test_reading_a_fold_that_was_never_assigned_says_so(frame):
    with pytest.raises(KeyError, match="make_splits"):
        split_for(frame, 0)


def test_the_same_seed_gives_the_same_split(frame):
    first = make_splits(frame, CONFIG)
    second = make_splits(frame, CONFIG)
    assert first[SPLIT_COLUMN].tolist() == second[SPLIT_COLUMN].tolist()
    assert first[FOLD_COLUMN.format(fold=0)].tolist() == \
        second[FOLD_COLUMN.format(fold=0)].tolist()


def test_a_different_seed_moves_it(frame):
    other = make_splits(frame, {**CONFIG, "seed": 99})
    base = make_splits(frame, CONFIG)
    assert other[SPLIT_COLUMN].tolist() != base[SPLIT_COLUMN].tolist()


def test_stratification_can_be_turned_off(frame):
    out = make_splits(frame, {**CONFIG, "stratify_cols": []})
    assert set(out[SPLIT_COLUMN]) == {"train", "test"}


def test_excluding_a_patient_drops_it_even_when_patients_would_have_taken_it():
    """`SL_QA` is a QA rig. Excluding has to win over selecting, or an unfiltered build keeps it."""
    from phase.building import select

    frame = pd.DataFrame({"Patient": ["SL0001", "SL_QA", "SL0002"],
                          "PatientKey": ["a", "qa", "b"],
                          "RadarSignalID": [1, 2, 3], "SessionID": [1, 2, 3]})

    assert set(select(frame, exclude_patients=["SL_QA"])["Patient"]) == {"SL0001", "SL0002"}
    # Wanted and excluded at once - excluded wins.
    assert select(frame, patients=["SL_"], exclude_patients=["SL_QA"]).empty
    assert set(select(frame, patients=["SL"], exclude_patients=["SL_QA"])["Patient"]) == \
        {"SL0001", "SL0002"}
    assert len(select(frame)) == 3
