"""The package-name collision that this repo is laid out to avoid.

`respiration-phase-labeling` calls its package `utils` and its config `parameters`. Both
checkouts end up on `sys.path` during a dataset build, so if this repo used either name, every
`from utils import ...` inside that repo would resolve to *our* module and the build would fail
somewhere far from the cause. This is the test that says so out loud.
"""

import importlib
import sys
from pathlib import Path

import pytest

from phase import bridge

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_this_repo_does_not_define_utils_or_parameters():
    for forbidden in ("utils", "parameters"):
        assert not (REPO_ROOT / forbidden).exists(), (
            f"{forbidden}/ would shadow the labelling repo's own package on sys.path")


def test_our_package_is_importable_under_its_own_name():
    for name in ("labels", "bridge", "building", "dataset", "splits", "decode", "metrics"):
        module = importlib.import_module(f"phase.{name}")
        assert str(REPO_ROOT) in module.__file__


def test_a_missing_checkout_is_reported_rather_than_raised():
    bundle, reason = bridge.labeling("/nowhere/at/all")
    assert bundle is None
    assert "no respiration-phase-labeling checkout" in reason


@pytest.mark.skipif(not Path("~/respiration-phase-labeling").expanduser().exists(),
                    reason="the labelling checkout is not on this machine")
def test_the_bridge_reaches_their_modules_not_ours():
    bundle = bridge.require_labeling("~/respiration-phase-labeling")
    assert "respiration-phase-labeling" in bundle["building"].__file__
    assert "respiration-phase-labeling" in bundle["suggestion"].__file__
    for name in ("eligible_signals", "sample_signals", "windows_of"):
        assert hasattr(bundle["building"], name)
    assert hasattr(bundle["suggestion"], "suggest")


def test_versions_names_what_decides_the_numerics():
    assert set(bridge.versions()) == {"holosissystem", "holosis-aws-manager"}


def test_a_shard_name_carries_its_environment():
    """`ds_algo` and `ds_prod` have unrelated RadarSignal ID spaces.

    Signal 2120091 is a different recording on each, so a name without the environment in it
    would have one run silently overwrite the other's samples.
    """
    from phase.building import SHARD_TEMPLATE

    algo = SHARD_TEMPLATE.format(env="ds_algo", signal_id=2120091)
    prod = SHARD_TEMPLATE.format(env="ds_prod", signal_id=2120091)
    assert algo != prod
    assert "ds_algo" in algo and "ds_prod" in prod
