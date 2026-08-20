"""Reach the two repos that own the pipeline, rather than carrying a copy of either.

`respiration-phase-labeling` owns signal selection, sampling, the raw-scan run and the
orientation decision; it in turn reaches `inhale-exhale-detection`, which owns the wrapper
that runs `holosissystem` unmodified. Nothing about which window exists, which way up it is,
or where a phase boundary falls is decided here - this module only imports.

Both checkouts are located by config with an environment-variable override, so a second
machine does not need a tracked file edited.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

LABELING_REPO_ENV = "RESPIRATION_PHASE_LABELING_REPO"
DETECTION_REPO_ENV = "INHALE_EXHALE_DETECTION_REPO"


def _repo(configured: str, env_var: str) -> Path:
    return Path(os.environ.get(env_var) or configured).expanduser()


@lru_cache(maxsize=1)
def labeling(configured: str = "~/respiration-phase-labeling") -> Any:
    """The labelling repo's build modules, or an explanation of why they are unreachable.

    Returns ``(bundle, reason)``. A missing checkout is reported as a missing checkout - a
    dataset build that silently produces nothing is the worst possible answer.
    """
    repo = _repo(configured, LABELING_REPO_ENV)
    if not repo.exists():
        return None, f"no respiration-phase-labeling checkout at {repo}"
    # Appended, not prepended, and the names below are **theirs**. Two collisions are avoided
    # by construction rather than by ordering: this repo's package is `phase`, not `utils`, and
    # its config lives under `parameter/`, not `parameters/`. Rename either back and these
    # imports silently resolve to our own modules instead of the labelling repo's.
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    try:
        from utils import building as labeling_building
        from utils import suggestion as labeling_suggestion
        from parameters.params import load_params as labeling_params
    except Exception as error:                                            # noqa: BLE001
        return None, f"cannot import the window builder from {repo}: {error}"
    return {"building": labeling_building, "suggestion": labeling_suggestion,
            "params": labeling_params, "root": repo}, None


def require_labeling(configured: str = "~/respiration-phase-labeling") -> dict[str, Any]:
    bundle, reason = labeling(configured)
    if bundle is None:
        raise RuntimeError(reason)
    return bundle


def versions() -> dict[str, str]:
    """What produced this dataset. Recorded in the manifest, since the numerics follow it."""
    import importlib.metadata as metadata

    out = {}
    for package in ("holosissystem", "holosis-aws-manager"):
        try:
            out[package] = metadata.version(package)
        except Exception:                                                 # noqa: BLE001
            out[package] = "unknown"
    return out
