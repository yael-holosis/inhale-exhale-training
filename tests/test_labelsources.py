"""Waveform orientation against `ReviewerFlipped`. Pins the switch, not the measurement."""

import numpy as np
import pandas as pd
import pytest

from phase.labelsources import ALGORITHM, HUMAN, LabelSource
from phase.sources import REVIEWER_FLIPPED

TRACE = np.array([0.0, 1.0, 2.0, 1.0, -1.0], dtype=np.float32)


def source(**cfg) -> LabelSource:
    """Explicit either way: the point of these tests is the switch, not the shipped default."""
    return LabelSource("ds_algo", {"source": ALGORITHM, "orient_by_reviewer_flip": False, **cfg})


def window(flipped: bool) -> pd.Series:
    return pd.Series({"ID": 1, REVIEWER_FLIPPED: flipped})


def test_unset_means_off_so_orientation_is_never_an_accident():
    bare = LabelSource("ds_algo", {"source": ALGORITHM})
    assert bare.orient_by_reviewer_flip is False
    assert bare.orient(window(True), TRACE).tolist() == TRACE.tolist()


@pytest.mark.parametrize("flipped", [True, False])
def test_off_leaves_every_window_alone(flipped):
    assert source().orient(window(flipped), TRACE).tolist() == TRACE.tolist()


def test_on_negates_only_the_flagged_window():
    oriented = source(orient_by_reviewer_flip=True)
    assert oriented.orient(window(True), TRACE).tolist() == (-TRACE).tolist()
    assert oriented.orient(window(False), TRACE).tolist() == TRACE.tolist()


def test_on_is_amplitude_only_so_sample_order_is_untouched():
    """The flip is a sign change, never a reversal in time - indices carry the human spans."""
    flipped = source(orient_by_reviewer_flip=True).orient(window(True), TRACE)
    assert flipped.size == TRACE.size
    assert np.argmax(np.abs(flipped)) == np.argmax(np.abs(TRACE))


def test_orienting_twice_returns_the_original():
    oriented = source(orient_by_reviewer_flip=True)
    once = oriented.orient(window(True), TRACE)
    assert oriented.orient(window(True), once).tolist() == TRACE.tolist()


def test_a_window_with_no_flag_column_is_left_alone():
    """`row.get` default matters: a catalogue without the column must not negate everything."""
    bare = pd.Series({"ID": 1})
    assert source(orient_by_reviewer_flip=True).orient(bare, TRACE).tolist() == TRACE.tolist()


@pytest.mark.parametrize("which", [ALGORITHM, HUMAN])
def test_the_dataset_records_the_choice_for_either_source(which):
    """`build_params.yaml` has to say how a directory was oriented, or it is not self-describing."""
    described = LabelSource("ds_algo", {"source": which,
                                        "orient_by_reviewer_flip": True}).describe()
    assert described["orient_by_reviewer_flip"] is True


def test_excluded_labellers_are_read_from_config_and_recorded():
    """`build_params.yaml` has to say who was dropped, or the dataset cannot be reproduced."""
    described = LabelSource("ds_algo", {"source": HUMAN,
                                        "exclude_labeler_ids": [7]}).describe()
    assert described["exclude_labeler_ids"] == [7]
    # Absent means nobody excluded, not a crash on None.
    assert LabelSource("ds_algo", {"source": HUMAN}).exclude_labeler_ids == []


def test_the_exclusion_sql_drops_only_the_named_labellers():
    from phase.sources import _not_labeller

    assert _not_labeller(None) == ""
    assert _not_labeller([]) == ""
    assert _not_labeller([7]) == " AND r.LabelerID NOT IN (7)"
    assert _not_labeller([7, 8]) == " AND r.LabelerID NOT IN (7, 8)"
    # Ints, so a value from YAML cannot carry SQL through.
    assert _not_labeller(["7"]) == " AND r.LabelerID NOT IN (7)"
