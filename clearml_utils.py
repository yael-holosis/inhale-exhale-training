"""ClearML wiring, the same shape as `cough/clearml_utils.py`.

Credentials come from `~/clearml.conf` and are never written down here. The server is reachable
or it is not; a run must not die because it is not, so every failure falls back to running
without logging and says so.

`DISABLE_CLEARML=true` turns it off entirely.
"""

from __future__ import annotations

import os
import re
import socket
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

from omegaconf import OmegaConf

CONFIG_PATHS = ("~/clearml.conf", "./clearml.conf", "~/.clearml/clearml.conf")
DEFAULT_API_PORT = 8008


def parse_clearml_config() -> Optional[dict[str, Any]]:
    """Server details from whichever config file exists first."""
    for candidate in CONFIG_PATHS:
        path = os.path.expanduser(candidate)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                content = handle.read()
        except OSError as error:
            print(f"warning: cannot read ClearML config at {path}: {error}")
            continue
        found = re.search(r"api_server:\s*([^\s\n]+)", content)
        if not found:
            continue
        url = found.group(1).strip()
        parsed = urlparse(url)
        return {"api_server_url": url, "host": parsed.hostname,
                "port": parsed.port or DEFAULT_API_PORT, "config_path": path}
    return None


def check_clearml_availability(timeout: int = 3) -> Tuple[bool, Optional[Any]]:
    """Whether the server answers, without creating a task."""
    try:
        from clearml import Task
    except ImportError:
        print("ClearML not installed")
        return False, None

    config = parse_clearml_config()
    if not config:
        print(f"no ClearML config found in {', '.join(CONFIG_PATHS)}")
        return False, None

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    reachable = sock.connect_ex((config["host"], config["port"])) == 0
    sock.close()
    if reachable:
        print(f"ClearML reachable at {config['host']}:{config['port']}")
        return True, Task
    print(f"ClearML unreachable at {config['host']}:{config['port']} - running without logging")
    return False, None


def initialize_clearml_task(project_name: str, task_name: str, timeout: int = 10,
                            cfg: Any = None) -> Optional[Any]:
    """Start the task, or return None and let the run continue."""
    if os.getenv("DISABLE_CLEARML", "false").lower() == "true":
        print("ClearML disabled via DISABLE_CLEARML")
        return None

    available, Task = check_clearml_availability()
    if not available or Task is None:
        return None

    try:
        os.environ["CLEARML_API_DEFAULT_REQ_TIMEOUT"] = str(timeout)
        # Both autocaptures off, for the same reason: everything here is reported explicitly.
        # TensorBoard capture would put `val/loss` on a "val" chart keyed by step *and* leave the
        # explicit `loss - val` chart keyed by epoch - the same number twice, and the automatic
        # one cannot separate the folds because they share a task.
        task = Task.init(project_name=project_name, task_name=task_name,
                         auto_connect_frameworks={"matplotlib": False, "tensorboard": False})
        print(f"ClearML task: {project_name} / {task_name}")
        if cfg is not None:
            try:
                from omegaconf import OmegaConf
                # Hydra's autocapture records the config pre-override; replace it with what the
                # run actually used.
                task.connect_configuration(name="OmegaConf",
                                           configuration=OmegaConf.to_container(cfg,
                                                                                resolve=True))
            except Exception as error:                                    # noqa: BLE001
                print(f"warning: could not refresh the OmegaConf section: {error}")
        return task
    except Exception as error:                                            # noqa: BLE001
        print(f"ClearML init failed ({type(error).__name__}: {error}) - continuing without it")
        return None


def _compact(value: float) -> str:
    """`0.0001` -> `1e-4`, `0.001` -> `1e-3`, `0.1` -> `0.1`. Readable in a task list."""
    if value == 0:
        return "0"
    text = f"{value:g}"
    return text.replace("e-0", "e-").replace("e+0", "e")


def parameter_count(model_cfg: Any) -> int | None:
    """How many weights this model config makes. None if the net cannot be built here."""
    try:
        import torch  # noqa: F401

        from models.unet1d import UNet1D

        net = UNet1D(in_channels=int(model_cfg.in_channels),
                     channels=[int(c) for c in model_cfg.channels],
                     bottleneck=int(model_cfg.bottleneck),
                     kernel_size=int(model_cfg.kernel_size),
                     dropout=float(model_cfg.get("dropout", 0.0)))
        return sum(p.numel() for p in net.parameters())
    except Exception:                                                     # noqa: BLE001
        return None


def run_title(cfg: Any) -> str:
    """A name that says what the run was, so the ClearML list is readable without opening rows.

    Everything in here changes what the numbers mean, so two runs that differ in any of it must
    not share a name: what it was trained against (label source, corrections), how much capacity
    it had (shape and weight count), and how hard that capacity was held back (weight decay,
    dropout). A list of identically-named rows hides exactly the comparison being run.
    """
    model = cfg.model
    labels = cfg.data.labels
    shape = "-".join(str(int(c)) for c in model.channels) + f"x{int(model.bottleneck)}"
    parts = [cfg.data.name, str(labels.source), model.name, shape, f"k{model.kernel_size}"]
    weights = parameter_count(model)
    if weights:
        parts.append(f"{weights / 1000:.1f}kp")
    parts.append(f"wd{_compact(float(cfg.training.weight_decay))}")
    parts.append(f"do{_compact(float(model.get('dropout', 0.0)))}")
    corrections = OmegaConf.to_container(labels.corrections, resolve=True) \
        if "corrections" in labels else {}
    marks = []
    if corrections.get("blank_edge_spans"):
        marks.append("edgeblank")
    above = corrections.get("all_unknown_above")
    if above is not None:
        marks.append(f"blank{round(float(above) * 100):g}")
    return "_".join(parts + (marks or ["asdrawn"]))
