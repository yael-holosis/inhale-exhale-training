"""Stop merged into exhale: a three-class model, end to end - no fourth output, row or panel."""

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from models.lightning_module import PhaseSegmenter, allowed_for
from phase import durations, figures
from phase.decode import decode, transition_matrix
from phase.labels import (EXHALE, INHALE, MERGED_PHASES, PHASES, STOP, UNKNOWN, classes_for,
                          classes_of)
from phase.metrics import duration_agreement, duration_pairs, event_level, per_sample

MODEL = {"in_channels": 1, "channels": [4, 8], "bottleneck": 8, "kernel_size": 5,
         "dropout": 0.0}


def _training():
    return OmegaConf.to_container(OmegaConf.load("parameter/training/default.yaml"), resolve=True)


def _breaths(n: int = 3) -> np.ndarray:
    return np.concatenate([np.full(4, UNKNOWN)]
                          + [np.r_[np.full(8, INHALE), np.full(12, EXHALE)]] * n)


def test_the_merged_vocabulary_is_a_prefix_so_indices_keep_their_meaning():
    assert MERGED_PHASES == ("unknown", "inhale", "exhale")
    assert PHASES[:len(MERGED_PHASES)] == MERGED_PHASES
    assert classes_for(True) == MERGED_PHASES and classes_for(False) == PHASES
    assert classes_of(3) == MERGED_PHASES and classes_of(4) == PHASES
    with pytest.raises(ValueError):
        classes_of(5)


def test_a_merged_model_has_three_outputs_and_a_three_class_decoder():
    model = PhaseSegmenter(model=MODEL, training=_training(), label_source="human",
                           stop_as_exhale=True)
    assert model.classes == MERGED_PHASES
    assert model(torch.randn(2, 1, 64)).shape == (2, 3, 64)
    assert model.cost.shape == (3, 3)
    assert model.class_weights.shape == (3,)
    target = torch.as_tensor(_breaths()[:64][None].repeat(2, 0))
    loss, _ = model._loss(model(torch.randn(2, 1, 64)), target, torch.ones(2, 64, dtype=bool))
    assert torch.isfinite(loss)


def test_an_unmerged_model_is_unchanged():
    model = PhaseSegmenter(model=MODEL, training=_training(), label_source="human")
    assert model(torch.randn(1, 1, 32)).shape == (1, 4, 32)
    assert model.cost.shape == (4, 4)


def test_a_table_naming_a_class_outside_the_set_is_refused():
    with pytest.raises(KeyError):
        transition_matrix({"exhale": ["stop"]}, 2.0, MERGED_PHASES)


def test_logits_and_table_of_different_widths_are_refused():
    cost = transition_matrix(allowed_for(_training()["decoding"], "human"), 2.0)
    with pytest.raises(ValueError):
        decode(np.zeros((10, 3)), cost)


def test_three_wide_logits_decode_under_the_merged_table_with_floors_on():
    decoding = _training()["decoding"]
    cost = transition_matrix(allowed_for(decoding, "human", stop_as_exhale=True),
                             decoding["switch_penalty"], MERGED_PHASES)
    truth = _breaths()
    logits = np.eye(3)[truth] * 6.0
    path = decode(logits, cost, decoding["min_duration"])
    assert path.tolist() == truth.tolist()


def test_the_metrics_carry_no_stop_key():
    truth = _breaths()
    scores = {**per_sample(truth, truth, classes=MERGED_PHASES),
              **event_level(truth, truth, classes=MERGED_PHASES)}
    assert not [key for key in scores if key.endswith("_stop")]
    assert scores["macro_f1"] == 1.0
    pairs = [duration_pairs(truth, truth, 10.0, MERGED_PHASES)]
    assert set(duration_agreement(pairs, MERGED_PHASES)) == {"inhale", "exhale"}


def test_the_duration_reports_have_no_stop_column(tmp_path):
    truth = _breaths(5)
    items = [{durations.LABEL: truth, durations.MODEL: truth, "fps": 10.0, "env": "e",
              "window_id": 1, "signal": 7, "patient": "P"}]
    columns = durations.columns_for(MERGED_PHASES)
    assert columns == ("inhale", "exhale", durations.RATIO)
    assert durations.columns_for() == durations.COLUMNS

    pairs, coverage = durations.span_errors(items, 0.5, MERGED_PHASES)
    assert "stop" not in set(coverage["phase"])
    stats = durations.span_agreement(pairs, coverage, columns)
    assert tuple(stats.index) == columns
    figures.span_duration_report(pairs, stats, tmp_path / "spans.png")

    medians = durations.signal_medians(items)
    signal_stats = durations.signal_agreement(medians, columns)
    assert tuple(signal_stats.index) == columns
    figures.signal_duration_report(medians, signal_stats, tmp_path / "signals.png")


def test_the_fold_report_draws_three_classes(tmp_path):
    truth = _breaths()
    metrics = per_sample(truth, truth, classes=MERGED_PHASES)
    per_fold = pd.DataFrame([{**metrics, "fold": f"fold_{i}"} for i in range(2)]).set_index("fold")
    path = figures.fold_report(per_fold, metrics, (0.9, 1.0), np.eye(3),
                               tmp_path / "fold_report.png")
    assert path.exists()


def test_a_merged_target_never_holds_stop():
    from phase.corrections import Corrections

    drawn = np.r_[np.full(3, INHALE), np.full(3, EXHALE), np.full(3, STOP)]
    merged, _ = Corrections(merge_stop_into_exhale=True).apply(drawn)
    assert merged.max() < len(MERGED_PHASES)
