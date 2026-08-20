"""Reach `respiration-phase-labeling` - the labelling **app** repo - rather than copying it.

That repo is the one thing this one is built on. It owns the connections to both instances, the
`RespirationWindow` table, the window blobs in S3, and the call into production's phase
calculation. It in turn reaches `inhale-exhale-detection`, which owns the wrapper that runs
`holosissystem` unmodified. Nothing about which window exists, which way up it is, or where a
phase boundary falls is decided here - this module only imports.

Located by config with an environment-variable override, so a second machine does not need a
tracked file edited.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

APP_REPO_ENV = "RESPIRATION_PHASE_LABELING_REPO"
DEFAULT_APP_REPO = "~/respiration-phase-labeling"


@lru_cache(maxsize=1)
def labeling_app(configured: str = DEFAULT_APP_REPO) -> Any:
    """The app repo's modules, or an explanation of why they are unreachable.

    Returns ``(bundle, reason)``. A missing checkout is reported as a missing checkout - a
    dataset build that silently produces nothing is the worst possible answer.
    """
    repo = Path(os.environ.get(APP_REPO_ENV) or configured).expanduser()
    if not repo.exists():
        return None, f"no respiration-phase-labeling checkout at {repo}"
    # Appended, not prepended, and the names below are **theirs**. Two collisions are avoided by
    # construction rather than by ordering: this repo's package is `phase`, not `utils`, and its
    # config lives under `parameter/`, not `parameters/`. Rename either back and these imports
    # silently resolve to our own modules instead of the app's.
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    try:
        from utils import db
        from utils import suggestion
        from utils import waveforms
        from parameters.params import load_params
    except Exception as error:                                            # noqa: BLE001
        return None, f"cannot import the app repo at {repo}: {error}"
    return {"db": db, "suggestion": suggestion, "waveforms": waveforms,
            "params": load_params, "root": repo}, None


def require_app(configured: str = DEFAULT_APP_REPO) -> dict[str, Any]:
    bundle, reason = labeling_app(configured)
    if bundle is None:
        raise RuntimeError(reason)
    return bundle


def versions() -> dict[str, str]:
    """What produced these labels. Recorded in the manifest, since the numerics follow it."""
    import importlib.metadata as metadata

    out = {}
    for package in ("holosissystem", "holosis-aws-manager"):
        try:
            out[package] = metadata.version(package)
        except Exception:                                                 # noqa: BLE001
            out[package] = "unknown"
    return out
