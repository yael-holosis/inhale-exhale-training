"""Run the phase segmenter on the waveforms in this folder. **This is the file the device runs.**

    cd /data/edge_test && /opt/env/bin/python run_inference.py --runs 10

No Lightning, no Hydra, no ClearML, no AWS, no database, no dataset - numpy and torch only, which
is what `holosissystem` already installs.

There is one copy of this file. `pack.py` fills the folder around it (weights, waveforms, and the
`phase/` and `models/` modules that decide the numerics) and the whole folder is what goes to the
device, so nothing here is ever duplicated into a build directory.

Two questions, needing different amounts of work:

- **Is the inference right?** One pass answers it. `pack.py` records the host's own logits beside
  every waveform, so the device has something exact to reproduce - a wrong BLAS or a truncated
  file shows up as a logit difference rather than as a slightly worse F1 nobody can interpret.
  Checked on every pass, so nondeterminism cannot hide either.
- **How long does it take?** One pass answers nothing. `--runs N` walks the whole set N times and
  keeps every measurement; `--warmup` throws away the first passes, where torch is still
  allocating and the caches are cold. Every window is the same duration by construction
  (`pack.py --seconds`), so the spread that comes out is the device's, not the data's.

Writes three artifacts here: `results.json` (the summary), `timings.csv` (one row per window per
run - the distribution itself, nothing pre-aggregated) and `predictions.npz` (the decoded labels
next to the reference, for `report.py` to draw on the host).

Exit status is 0 only if every window reproduced the host inside `LOGIT_TOLERANCE` and decoded to
the same labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from phase.decode import decode, transition_matrix
from phase.labels import N_CLASSES, PHASES
from phase.metrics import event_level, per_sample
from phase.preprocess import normalise_window
from models.unet1d import UNet1D

def as_typed(path: Path) -> Path:
    """The path the way the shell has it, not the way the kernel does.

    `resolve()` follows symlinks and `os.getcwd()` has already followed them, so on a device where
    `/root` is a link into `/data` both turn the directory you are standing in into a physical path
    you have never typed - and that is not a path anybody can paste back. The shell keeps what you
    typed in `$PWD`; this prefers it whenever it names the same directory.
    """
    typed = os.environ.get("PWD")
    if typed:
        candidate = Path(typed)
        try:
            if candidate.is_dir() and os.path.samefile(candidate, path):
                return candidate
        except OSError:
            pass
    return path


FOLDER = as_typed(Path(__file__).absolute().parent)
"""Everything is beside this file, so the runner needs no path argument - there is no shell history
on a device to fish one out of."""

# ------------------------------------------------------------------- the folder's own layout
# Named here rather than in a module of their own: `pack.py` and `report.py` import them from
# this file, which is the only one of the three that also travels.

MANIFEST = "manifest.json"
WEIGHTS = "weights.pt"
WAVEFORMS_DIR = "waveforms"
WAVEFORM_TEMPLATE = "{env}_window_{window_id}.npz"
RESULTS = "results.json"
TIMINGS = "timings.csv"
PREDICTIONS = "predictions.npz"

ARTIFACTS = (RESULTS, TIMINGS, PREDICTIONS)
"""What a device run leaves behind."""

OUT_DIR = "out"
"""Results go in here, not in the folder root - on the device as much as on the host. Somebody
who has just run this looks for its output where they are standing, and the previous version told
them to look in a directory that only existed on the machine that packed the folder."""

RETURN_FILES = (*ARTIFACTS, MANIFEST)
"""What to copy back. The manifest goes with them so a returned run is self-describing: it names
its own checkpoint, dataset and decoding table, and `report.py` needs nothing from the folder that
produced it. Without that, re-packing this folder silently relabels every past run."""

NUMERIC_MODULES = ("phase/__init__.py", "phase/labels.py", "phase/decode.py",
                   "phase/metrics.py", "phase/preprocess.py",
                   "models/__init__.py", "models/unet1d.py")
"""What the inference itself needs. numpy and torch only, which is why `phase/dataset.py` and
`models/lightning_module.py` are not here: they pull in pandas, Lightning and the database
layer."""

PLOT_MODULES = ("phase/figures.py", "phase/durations.py", "phase/splits.py")
"""What the figures need on top of that: matplotlib and pandas, both of which `holosissystem`
installs. Same drawing code as the training run makes, so a device page and a training page read
identically."""

MODULES = (*NUMERIC_MODULES, *PLOT_MODULES)
"""Copied in by `pack.py` from the training repo rather than reimplemented - a second
implementation of a decoder or a panel is a second thing to keep in step."""

LOGIT_TOLERANCE = 1e-3
"""Max absolute logit difference between host and device before the run is called a failure.
Both are float32 CPU torch, so the difference should be at the last bit; the floor is here for a
different BLAS, not for a different model."""

DEVICE_USER = "root"
"""Only used to print the commands below."""

HOST_FOLDER = "edge_test"
"""This folder's name in the training repo. `report.py` reads `edge_test/out`, so fetching a
device's `out/` into `edge_test/` is the whole round trip - and it is written down once, here,
because printing it in three files is how the three drifted apart."""


def fetch_command(device_folder, ip: str = "<ip>") -> str:
    """How to bring a run back to the machine that packed the folder. Run from the repo root.

    `device_folder` is whatever the caller knows it to be - the runner passes its own resolved
    path, `pack.py` passes `--device-dir` or a placeholder. Nothing here invents one.
    """
    return (f"scp -r {DEVICE_USER}@{ip}:{device_folder}/{OUT_DIR} {HOST_FOLDER}/\n"
            f"  poetry run python -m {HOST_FOLDER}.report")

STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
"""UTC, and the same shape as a dataset directory's stamp. It names the run: `report.py` puts a
run's figures in a directory of this name, so a second run never lands on the first one's."""

PERCENTILES = (50, 90, 95, 99)
TIMING_COLUMNS = ("run", "window_id", "env", "patient", "samples", "seconds",
                  "forward_ms", "decode_ms", "total_ms")


# --------------------------------------------------------------------------------- the model

def out_dir(folder: Path) -> Path:
    """Where this run's results go. Created on demand."""
    path = folder / OUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model(folder: Path, manifest: dict) -> UNet1D:
    """The net alone, from a state dict. `weights_only=True`, so nothing in the file is executed."""
    stored = torch.load(folder / WEIGHTS, map_location="cpu", weights_only=True)
    net = UNet1D(n_classes=N_CLASSES, **manifest["model"])
    net.load_state_dict(stored["state_dict"])
    net.eval()
    return net


def decoder(manifest: dict) -> tuple[np.ndarray | None, dict | None]:
    """The two decoding passes as the training run had them, rebuilt from the recorded table."""
    decoding = manifest["decoding"]
    cost = (transition_matrix(decoding["allowed"], decoding["switch_penalty"])
            if decoding["viterbi"] else None)
    return cost, (decoding["min_duration"] if decoding["enforce_min"] else None)


def waveforms(folder: Path) -> list[dict]:
    """Every window in the folder, in the order `pack.py` wrote them."""
    out = []
    for path in sorted((folder / WAVEFORMS_DIR).glob("*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            out.append({"name": path.stem,
                        "values": stored["values"].astype(np.float32),
                        "targets": stored["targets"].astype(np.int64),
                        "host_logits": stored["host_logits"].astype(np.float32),
                        "host_prediction": stored["host_prediction"].astype(np.int64),
                        "window_id": int(stored["window_id"]),
                        "patient": str(stored["patient"]),
                        "env": str(stored["env"]),
                        "fps": float(stored["fps"])})
    return out


@torch.no_grad()
def infer(net: UNet1D, values: np.ndarray, normalise: str,
          cost, minimum) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One window through the net and the decoder, timed separately.

    Normalisation is inside the forward timing: it is work the device would have to do too.
    """
    started = time.perf_counter()
    x = torch.from_numpy(normalise_window(values, normalise)[None, None, :])
    logits = net(x)[0].permute(1, 0).numpy()
    forward = (time.perf_counter() - started) * 1e3

    started = time.perf_counter()
    prediction = decode(logits, cost, minimum)
    return logits, prediction, forward, (time.perf_counter() - started) * 1e3


def warm_up(net, items: list[dict], normalise: str, cost, minimum, passes: int) -> None:
    """Discarded passes. The first forward on a cold interpreter allocates every workspace torch
    will reuse, and on a net this small that cost is the whole measurement."""
    for _ in range(passes):
        for item in items:
            infer(net, item["values"], normalise, cost, minimum)


# ------------------------------------------------------------------------------- measurement

def correctness(item: dict, logits: np.ndarray, prediction: np.ndarray) -> dict:
    """This window against its labels, and against what the host got on it."""
    scores = {**per_sample(prediction, item["targets"]),
              **event_level(prediction, item["targets"])}
    return {"window_id": item["window_id"], "patient": item["patient"], "env": item["env"],
            "fps": item["fps"], "samples": int(item["values"].size),
            "macro_f1": float(scores["macro_f1"]),
            "f1_inhale": float(scores.get("f1_inhale", float("nan"))),
            "f1_exhale": float(scores.get("f1_exhale", float("nan"))),
            "f1_stop": float(scores.get("f1_stop", float("nan"))),
            "event_f1_inhale": float(scores.get("event_f1_inhale", float("nan"))),
            "event_f1_exhale": float(scores.get("event_f1_exhale", float("nan"))),
            "logit_drift": float(np.abs(logits - item["host_logits"]).max()),
            "labels_match_host": bool(np.array_equal(prediction, item["host_prediction"])),
            "agreement_with_host": float(np.mean(prediction == item["host_prediction"])),
            "classes": {name: int((prediction == index).sum())
                        for index, name in enumerate(PHASES)}}


def spread(values) -> dict[str, float]:
    """What a distribution is worth reporting as. The percentiles are the point, not the mean."""
    array = np.asarray(values, dtype=np.float64)
    out = {"n": int(array.size), "mean": float(array.mean()),
           "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
           "min": float(array.min()), "max": float(array.max())}
    for percentile, value in zip(PERCENTILES, np.percentile(array, PERCENTILES)):
        out[f"p{percentile}"] = float(value)
    return out


def measure(net, items: list[dict], normalise: str, cost, minimum,
            runs: int) -> tuple[list[dict], list[dict]]:
    """Every window, `runs` times: one timing row per (run, window), one check per window.

    The check is the worst case over the runs, so a device that answers differently on the
    second pass cannot hide behind the first.
    """
    timings, checks = [], {}
    for run in range(runs):
        for item in items:
            logits, prediction, forward, decode_ms = infer(net, item["values"], normalise,
                                                           cost, minimum)
            timings.append({"run": run, "window_id": item["window_id"], "env": item["env"],
                            "patient": item["patient"], "samples": int(item["values"].size),
                            "seconds": round(item["values"].size / item["fps"], 2),
                            "forward_ms": forward, "decode_ms": decode_ms,
                            "total_ms": forward + decode_ms})
            check = correctness(item, logits, prediction)
            previous = checks.get(item["window_id"])
            if previous is not None:
                check["logit_drift"] = max(check["logit_drift"], previous["logit_drift"])
                check["labels_match_host"] = (check["labels_match_host"]
                                              and previous["labels_match_host"])
            checks[item["window_id"]] = check
    return timings, [checks[item["window_id"]] for item in items]


# --------------------------------------------------------------------------------- reporting

def stamp() -> str:
    return time.strftime(STAMP_FORMAT, time.gmtime())


def previous(folder: Path) -> str | None:
    """When the run whose artifacts are about to be replaced happened, if there is one.

    Said out loud rather than silently overwritten: `--runs 10` twice is two experiments, and the
    first one's raw rows are gone the moment the second finishes.
    """
    path = folder / OUT_DIR / RESULTS
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text()).get("finished", "an earlier run"))
    except (ValueError, OSError):
        return "an earlier run"


def environment() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "machine": platform.machine(),
            "platform": platform.platform(), "threads": torch.get_num_threads()}


def write_timings(out: Path, timings: list[dict]) -> Path:
    out = out / TIMINGS
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_COLUMNS)
        writer.writeheader()
        writer.writerows(timings)
    return out


def write_predictions(out: Path, items: list[dict], net, normalise: str,
                      cost, minimum) -> Path:
    """The decoded labels beside the reference, ragged like a shard. Nothing here plots -
    matplotlib is not part of the device's budget; `report.py` draws these on the host."""
    predictions = [infer(net, item["values"], normalise, cost, minimum)[1] for item in items]
    lengths = np.array([item["values"].size for item in items], dtype=np.int64)
    out = out / PREDICTIONS
    np.savez_compressed(
        out,
        values=np.concatenate([item["values"] for item in items]).astype(np.float32),
        targets=np.concatenate([item["targets"] for item in items]).astype(np.int8),
        prediction=np.concatenate(predictions).astype(np.int8),
        offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        window_id=np.array([item["window_id"] for item in items], dtype=np.int64),
        patient=np.array([item["patient"] for item in items]),
        env=np.array([item["env"] for item in items]),
        fps=np.array([item["fps"] for item in items], dtype=np.float32))
    return out


def report(checks: list[dict], timings: list[dict], manifest: dict, env: dict,
           runs: int, warmup: int) -> dict:
    """Print both tables, return what goes in `results.json`."""
    tolerance = manifest["logit_tolerance"]
    print(f"\n{len(checks)} windows x {runs} runs ({warmup} warm-up passes discarded)")
    print(f"{manifest['dataset']}, fold {manifest['fold']}, {manifest['split']} split, "
          f"labelled by `{manifest['label_source']}`")
    print(f"device: python {env['python']}, torch {env['torch']}, numpy {env['numpy']}, "
          f"{env['machine']}, {env['threads']} threads")
    print(f"host:   python {manifest['host']['python']}, torch {manifest['host']['torch']}, "
          f"numpy {manifest['host']['numpy']}, {manifest['host']['machine']}")

    print("\n--- inference, against the labels and against the host " + "-" * 24)
    header = (f"{'window':>9} {'env':>8} {'patient':>10} {'macro F1':>9} {'inh':>6} {'exh':>6} "
              f"{'stop':>6} {'host agree':>11} {'d logit':>10}")
    print(header)
    print("-" * len(header))
    for row in sorted(checks, key=lambda item: item["macro_f1"]):
        flag = "" if row["labels_match_host"] else "  MISMATCH"
        print(f"{row['window_id']:>9} {row['env']:>8} {row['patient']:>10} "
              f"{row['macro_f1']:>9.2f} {row['f1_inhale']:>6.2f} {row['f1_exhale']:>6.2f} "
              f"{row['f1_stop']:>6.2f} {row['agreement_with_host']:>10.2%} "
              f"{row['logit_drift']:>10.2e}{flag}")

    macro = spread([row["macro_f1"] for row in checks])
    drift = max(row["logit_drift"] for row in checks)
    mismatched = [row["window_id"] for row in checks if not row["labels_match_host"]]
    print(f"\nmacro F1 mean {macro['mean']:.2f}, median {macro['p50']:.2f}, "
          f"range {macro['min']:.2f}-{macro['max']:.2f}   "
          f"(host recorded {manifest['host_macro_f1']:.2f})")
    print(f"largest logit difference from the host {drift:.2e}, tolerance {tolerance:.0e}")
    if mismatched:
        print(f"windows decoding differently from the host: {mismatched}")

    print("\n--- time per window, over every run " + "-" * 43)
    per_window: dict[int, list[float]] = {}
    for row in timings:
        per_window.setdefault(row["window_id"], []).append(row["total_ms"])
    header = (f"{'window':>9} {'runs':>6} {'median':>9} {'mean':>9} {'sd':>8} {'min':>8} "
              f"{'p95':>8} {'max':>8}")
    print(header)
    print("-" * len(header))
    for window_id, values in sorted(per_window.items(), key=lambda pair: -np.median(pair[1])):
        stats = spread(values)
        print(f"{window_id:>9} {stats['n']:>6} {stats['p50']:>9.2f} {stats['mean']:>9.2f} "
              f"{stats['sd']:>8.2f} {stats['min']:>8.2f} {stats['p95']:>8.2f} "
              f"{stats['max']:>8.2f}")

    total = spread([row["total_ms"] for row in timings])
    forward = spread([row["forward_ms"] for row in timings])
    decoding = spread([row["decode_ms"] for row in timings])
    seconds = float(np.mean([row["seconds"] for row in timings]))
    print(f"\nall {total['n']} measurements, ms per {seconds:g} s window:")
    for name, stats in (("forward", forward), ("decode", decoding), ("total", total)):
        print(f"  {name:8} mean {stats['mean']:7.2f}  sd {stats['sd']:6.2f}  "
              f"min {stats['min']:7.2f}  p50 {stats['p50']:7.2f}  p95 {stats['p95']:7.2f}  "
              f"p99 {stats['p99']:7.2f}  max {stats['max']:7.2f}")
    print(f"  {seconds:g} s of signal in {total['p50']:.2f} ms median - "
          f"{seconds * 1e3 / total['p50']:.0f}x real time, "
          f"{seconds * 1e3 / total['max']:.0f}x at the worst measurement")

    passed = drift <= tolerance and not mismatched
    print("\nPASS - the device reproduces the host" if passed else
          "\nFAIL - the device does not reproduce the host")

    return {"passed": passed, "finished": stamp(), "runs": runs, "warmup": warmup,
            "environment": env,
            "windows": checks, "macro_f1": macro,
            "latency_ms": {"total": total, "forward": forward, "decode": decoding},
            "per_window_median_ms": {str(window): float(np.median(values))
                                     for window, values in per_window.items()},
            "window_seconds": seconds, "host_macro_f1": manifest["host_macro_f1"],
            "max_logit_drift": drift, "tolerance": tolerance,
            "mismatched_windows": mismatched,
            "checkpoint": manifest["checkpoint"], "dataset": manifest["dataset"],
            "packed": manifest["created"]}


def draw(out: Path) -> bool:
    """Draw this run's figures, here, into `<out>/report/<the run's stamp>/`.

    The plot is the artifact somebody actually looks at, so it is made where the run happened
    rather than waiting for a copy back to a laptop. `report.py` travels with the folder and needs
    matplotlib and pandas, both of which `holosissystem` installs - if they are missing this says
    so and the run's data is still complete, ready to be drawn anywhere else.
    """
    try:
        try:
            from edge_test import report                 # a checkout of the training repo
        except ImportError:
            import report                               # beside this file, on a device
    except ImportError as error:
        print(f"\nno figures drawn here: {error}. The run itself is complete - draw it on a "
              f"machine with matplotlib:\n  poetry run python -m {HOST_FOLDER}.report")
        return False
    report.build(out)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", type=Path, default=FOLDER,
                        help="where the waveforms are (default: this file's own folder)")
    parser.add_argument("--runs", type=int, default=10,
                        help="passes over the whole window set; every measurement is kept")
    parser.add_argument("--warmup", type=int, default=2,
                        help="passes to discard first, while torch is still allocating")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads; the default is torch's own")
    parser.add_argument("--no-figures", action="store_true",
                        help="write the data only; do not draw here")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    folder = as_typed(args.folder.absolute())
    if not (folder / MANIFEST).exists():
        print(f"{folder / MANIFEST} is missing - run `python -m edge_test.pack` first")
        return 2
    manifest = json.loads((folder / MANIFEST).read_text())
    items = waveforms(folder)
    if not items:
        print(f"no waveform in {folder / WAVEFORMS_DIR}")
        return 2

    net = load_model(folder, manifest)
    cost, minimum = decoder(manifest)
    normalise = manifest["normalise"]
    runs = max(1, args.runs)

    replacing = previous(folder)
    if replacing:
        print(f"note: this replaces the run of {replacing} in {folder / OUT_DIR} - copy it out "
              f"first if you still want it")

    warm_up(net, items, normalise, cost, minimum, args.warmup)
    timings, checks = measure(net, items, normalise, cost, minimum, runs)

    results = report(checks, timings, manifest, environment(), runs, args.warmup)
    out = out_dir(folder)
    (out / RESULTS).write_text(json.dumps(results, indent=2))
    write_timings(out, timings)
    write_predictions(out, items, net, normalise, cost, minimum)
    # The manifest goes with them, so what comes back names its own checkpoint, dataset and
    # decoding table - one directory, self-describing, and one recursive copy fetches it.
    shutil.copyfile(folder / MANIFEST, out / MANIFEST)

    print(f"\nrun {results['finished']} - results are in {out}")
    for name in RETURN_FILES:
        print(f"  {name}")
    if not args.no_figures:
        draw(out)
    print(f"\nto fetch this run, from the repo root on the machine that packed this folder:\n"
          f"  {fetch_command(folder)}")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
