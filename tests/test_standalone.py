"""This repo stands on `holosissystem` and the house AWS library, and on no sibling checkout.

The point of these is that a regression is loud. An import of a neighbouring repo, or a package
named so it shadows one, fails here rather than three weeks later on somebody else's laptop.
"""

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OUR_MODULES = ("labels", "building", "dataset", "splits", "decode", "metrics", "production",
               "sources")

SIBLING_REPOS = ("respiration_phase_labeling", "respiration-phase-labeling",
                 "inhale_exhale", "inhale-exhale-detection")
"""Repos this one used to reach into. It reads their databases and their S3 objects; it does not
import their code."""


def _tracked_python():
    return [path for path in REPO_ROOT.rglob("*.py")
            if ".venv" not in str(path) and "outputs" not in str(path)]


def test_no_module_imports_a_sibling_checkout():
    offenders = []
    for path in _tracked_python():
        text = path.read_text()
        for name in SIBLING_REPOS:
            if f"import {name}" in text or f"from {name}" in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {name}")
    assert not offenders, "\n".join(offenders)


def test_nothing_puts_another_repo_on_sys_path():
    # Assembled rather than written out, so this file does not match its own search.
    needles = ("sys.path." + verb for verb in ("append", "insert"))
    needles = tuple(needles)
    offenders = [str(path.relative_to(REPO_ROOT)) for path in _tracked_python()
                 if any(needle in path.read_text() for needle in needles)]
    assert not offenders, f"sys.path is manipulated in {offenders}"


def test_this_repo_does_not_define_utils_or_parameters():
    # The labelling app calls its package `utils` and its config `parameters`. Nothing puts them
    # on the path any more, but the names stay reserved: they are the collision that hid a bug
    # once, and re-taking them invites it back.
    for forbidden in ("utils", "parameters"):
        assert not (REPO_ROOT / forbidden).exists()


def test_our_package_is_importable_under_its_own_name():
    for name in OUR_MODULES:
        module = importlib.import_module(f"phase.{name}")
        assert str(REPO_ROOT) in module.__file__


def test_the_sources_config_names_both_environments():
    from phase import sources

    assert set(sources.config()["environments"]) == {"ds_algo", "ds_prod"}
    for env in ("ds_algo", "ds_prod"):
        assert sources.env_config(env)["labels"]["database"] == "edge_data_extras"


def test_an_unknown_environment_fails_loudly():
    from phase import sources

    with pytest.raises(KeyError):
        sources.env_config("staging")


def test_production_is_declared_read_only():
    from phase import sources

    assert sources.env_config("ds_prod")["device"]["read_only"] is True


def test_no_password_is_written_down_in_a_tracked_file():
    """The app repo carries production's read-only password in a tracked YAML. This one does not,
    and that is a property worth enforcing rather than remembering."""
    tracked = [path for path in REPO_ROOT.rglob("*")
               if path.is_file() and ".venv" not in str(path) and ".git/" not in str(path)
               and "outputs" not in str(path) and path.suffix in (".py", ".yaml", ".yml", ".md")]
    offenders = []
    for path in tracked:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("password:", "secret_key:", "access_key:")):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {stripped[:40]}")
    assert not offenders, "\n".join(offenders)


def test_only_reads_are_allowed_through_the_query_helper():
    from phase.sources import ReadOnly, frame

    with pytest.raises(ReadOnly):
        frame("ds_algo", "labels", "DELETE FROM RespirationWindow")


def test_latest_is_the_newest_build_not_the_last_name_alphabetically(tmp_path):
    """`phases_human_...` sorts after `phases_algorithm_...` whatever the timestamps.

    Sorting the names made `latest` mean "human" - it picked a 25-window evaluation set over the
    2,848-window training set. It failed loudly, because that set had no splits yet; it would not
    have, once both were split.
    """
    from phase.building import WINDOWS_NAME, resolve

    for name in ("phases_algorithm_20260820T143456Z", "phases_human_20260820T120000Z"):
        (tmp_path / name).mkdir()
        (tmp_path / name / WINDOWS_NAME).write_text("env\n")

    assert resolve(tmp_path, "latest").name == "phases_algorithm_20260820T143456Z"
    # And narrowing to a source picks the newest of that source, not of everything.
    assert resolve(tmp_path, "latest", "human").name == "phases_human_20260820T120000Z"


def test_an_unbuilt_source_is_reported_rather_than_silently_falling_back(tmp_path):
    from phase.building import WINDOWS_NAME, resolve

    (tmp_path / "phases_algorithm_20260820T143456Z").mkdir()
    (tmp_path / "phases_algorithm_20260820T143456Z" / WINDOWS_NAME).write_text("env\n")
    with pytest.raises(FileNotFoundError, match="human"):
        resolve(tmp_path, "latest", "human")
