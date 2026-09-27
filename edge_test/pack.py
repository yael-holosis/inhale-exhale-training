"""Fill this folder with a checkpoint, a few held-out windows, and the host's own answer.

    poetry run python -m edge_test.pack --checkpoint outputs/<date>/<time>/fold_0/checkpoints/best-epoch=NN.ckpt
    poetry run python -m edge_test.pack --probe 10.0.0.7        # what is on a device, first

The folder it fills is the one it lives in - `edge_test/` is what goes to the device, so
`run_inference.py` exists once and is never copied into a build directory. What gets written here
is data and the numeric modules: `weights.pt`, `waveforms/`, `manifest.json`, and copies of
`edge_test.run_inference: MODULES` from the training repo. All of it is gitignored; the three
scripts beside it are the tracked part.

The device has python 3.12, numpy and torch 2.9.1 - `holosissystem` installs them - and nothing
else this repo uses, so the Lightning checkpoint is unwrapped into a plain state dict.

**The host's own logits go in with the waveforms.** The device's job is to reproduce them, which is
what makes this a test rather than a demo.

Windows come from the held-out split of the dataset the checkpoint was trained on, read from
`dataset.txt` beside it, so the test is never on data the model has seen.

Nothing is downloaded and nothing outside this folder is written.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from edge_test import run_inference as runner
from evaluate import load_model, window_samples
from models.lightning_module import allowed_for
from phase import figures
from phase.building import load_windows, stamp
from phase.decode import decode
from phase.preprocess import WINDOW, normalise_window
from omegaconf import OmegaConf
from phase.splits import split_for

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_NAME = "dataset.txt"
"""Written beside every fold by `train.py` - the dataset a checkpoint belongs to."""

SECONDS = 20.0
"""One window length only, by default. A latency distribution over mixed lengths measures the
lengths, not the device: the net is fully convolutional, so a 25 s window costs 25% more than a
20 s one. 20 s is the pipeline's own first window length and most of the held-out set is exactly
that."""

DEVICE_USER = "root"

PROBE_PYTHONS = ("/opt/env/bin/python", "/data/env/bin/python", "/usr/bin/python3")
"""Candidates `--probe` *tests for*, from `provision_device.sh` and `scp_entrypoints.sh` in the
HolosisSystem repo. It reports which of them exists; nothing here assumes one does."""

DEVICE_DIR = "<folder on the device>"
DEVICE_IP = "<ip>"
DEVICE_PYTHON = "<the python --probe found>"
"""Placeholders. A device's paths are not this repo's to know - `--device-dir`, `--device-ip` and
`--device-python` fill them in when you want the printed commands to be copy-pasteable, and
`--probe` is how you find out what to pass."""

PROBE = """
import platform, sys
print('  python  ', sys.version.split()[0], platform.machine())
for name in ('numpy', 'torch'):
    try:
        print(f'  {name:8}', __import__(name).__version__)
    except Exception as error:
        print(f'  {name:8} MISSING ({type(error).__name__})')
"""


# ------------------------------------------------------------------------------------- probing

def probe(ip: str) -> int:
    """What is on a device, before anything is copied to it. Read-only: installs nothing."""
    ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
           f"{DEVICE_USER}@{ip}"]
    print(f"=== {ip} ===")
    subprocess.run([*ssh, "uname -m -r; df -h /data 2>/dev/null | tail -1"], check=False)
    for python in PROBE_PYTHONS:
        print(f"\n--- {python}")
        subprocess.run([*ssh, f"test -x {python} && {python} - <<'EOF'{PROBE}EOF"], check=False)
    print("\n--- holosissystem")
    subprocess.run([*ssh, "systemctl is-active holosissystem 2>/dev/null || true"], check=False)
    return 0


# ------------------------------------------------------------------------------ what gets packed

def dataset_of(checkpoint: Path, override: str | None) -> Path:
    """The dataset that trained this checkpoint, not whatever `latest` now means."""
    if override:
        return Path(override)
    marker = checkpoint.parent.parent / DATASET_NAME
    if not marker.exists():
        raise FileNotFoundError(f"{marker} is missing, so the dataset behind {checkpoint.name} "
                                f"is unknown - pass --dataset")
    return Path(marker.read_text().strip())


def labelled_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop windows the reference left entirely unknown.

    They score either 1.00 or near zero - the only class in play is `unknown` - so in a set of ten
    they spend slots saying nothing about whether the phases came out right. They belong in a
    metric over the whole split; not in a folder whose purpose is to eyeball ten predictions.
    """
    return frame[frame["n_unknown"] < frame["samples"]]


def stride_seconds(frame: pd.DataFrame) -> float | None:
    """How often production cuts a window, measured rather than configured.

    Nothing in `parameter/` sets this - the windows come from the database as production wrote
    them, so the cadence is a property of the recording and the only honest source for it is the
    recording. Consecutive `WindowIndex` within one signal, the modal gap in `start_index`.
    A dataset of one window per signal has nothing to measure and gets None.
    """
    gaps = []
    for _, group in frame.sort_values("WindowIndex").groupby("RadarSignalID"):
        if len(group) < 2:
            continue
        steps = np.diff(group["WindowIndex"].to_numpy())
        starts = np.diff(group["start_index"].to_numpy())
        fps = float(group["analysis_fps"].iloc[0])
        gaps.extend(starts[steps == 1] / fps)
    return round(float(pd.Series(gaps).mode().iloc[0]), 3) if gaps else None


def signal_seconds(frame: pd.DataFrame) -> float | None:
    """How long one radar signal runs, measured the same way and for the same reason as the
    stride. The end of the last window of a signal, modal across signals."""
    ends = ((frame["start_index"] + frame["samples"]) / frame["analysis_fps"])
    per_signal = ends.groupby(frame["RadarSignalID"]).max().round()
    return round(float(per_signal.mode().iloc[0]), 3) if len(per_signal) else None


def one_length(frame: pd.DataFrame, seconds: float) -> pd.DataFrame:
    """Only windows of exactly this duration. `samples` and `analysis_fps` decide it, not a
    nominal length - a window the pipeline grew is a different number of samples."""
    if not seconds:
        return frame
    return frame[frame["samples"] == (frame["analysis_fps"] * seconds).round()]


@torch.no_grad()
def answer(model, values: np.ndarray, normalise: str) -> tuple[np.ndarray, np.ndarray]:
    """The host's logits and decoded labels for one window - what the device has to reproduce."""
    x = torch.from_numpy(normalise_window(values, normalise)[None, None, :])
    logits = model(x)[0].permute(1, 0).cpu().numpy()
    return logits.astype(np.float32), decode(logits, model.cost, model.min_duration)


def net_state(model) -> dict[str, torch.Tensor]:
    """The U-Net's own tensors, unprefixed. The class weights buffer stays behind - it is a
    property of the training split, and nothing at inference reads it."""
    prefix = "net."
    return {key[len(prefix):]: value.cpu() for key, value in model.state_dict().items()
            if key.startswith(prefix)}


def clear(folder: Path) -> None:
    """Remove what a previous pack or run left, and nothing else. The tracked scripts stay.

    `report/` is deliberately not touched: those directories are the record of runs that already
    happened, each stamped and each naming the checkpoint it was made from.
    """
    for name in (runner.WEIGHTS, runner.MANIFEST):
        (folder / name).unlink(missing_ok=True)
    for directory in (runner.WAVEFORMS_DIR, runner.OUT_DIR,
                      *{Path(name).parts[0] for name in runner.MODULES}):
        shutil.rmtree(folder / directory, ignore_errors=True)


STRAY = ("report",)
"""Directories that must not be in the folder when it is copied. `report/` used to live here, and
a device ended up carrying two of this laptop's runs in it - which read exactly like its own."""


def warn_about_strays(folder: Path) -> None:
    for name in STRAY:
        if (folder / name).exists():
            print(f"WARNING: {folder / name} is a leftover from the old layout. Reports now go "
                  f"to out/edge_reports/. Move it out before copying this folder to a device - "
                  f"it would arrive looking like the device's own output.")


def copy_modules(folder: Path) -> None:
    """The modules that decide the numerics, byte for byte from the training repo."""
    for name in runner.MODULES:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, target)


def write_waveforms(folder: Path, chosen: list[dict]) -> list[dict]:
    directory = folder / runner.WAVEFORMS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    summary = []
    for item in chosen:
        row = item["row"]
        name = runner.WAVEFORM_TEMPLATE.format(env=row["env"],
                                               window_id=int(row["RespirationWindowID"]))
        np.savez_compressed(
            directory / name,
            values=item["values"].astype(np.float32),
            targets=item["reference"].astype(np.int8),
            host_logits=item["logits"].astype(np.float32),
            host_prediction=item["prediction"].astype(np.int8),
            window_id=np.int64(row["RespirationWindowID"]),
            patient=np.str_(row["PatientID"]),
            env=np.str_(row["env"]),
            fps=np.float32(row["analysis_fps"]))
        summary.append({"file": f"{runner.WAVEFORMS_DIR}/{name}",
                        "window_id": int(row["RespirationWindowID"]),
                        "patient": str(row["PatientID"]), "env": str(row["env"]),
                        "samples": int(item["values"].size),
                        "fps": float(row["analysis_fps"]),
                        "host_macro_f1": round(float(item["score"]), 4)})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", metavar="IP",
                        help="report a device's python / torch / numpy and exit")
    parser.add_argument("--device-ip", default=DEVICE_IP,
                        help="only fills in the commands printed at the end")
    parser.add_argument("--device-dir", default=DEVICE_DIR,
                        help="where the folder will live on the device; only fills in the "
                             "commands printed at the end")
    parser.add_argument("--device-python", default=DEVICE_PYTHON,
                        help="the interpreter --probe found; only fills in the commands printed "
                             "at the end")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", default=None,
                        help=f"overrides the {DATASET_NAME} beside the checkpoint")
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--n", type=int, default=10, help="windows to pack")
    parser.add_argument("--seconds", type=float, default=SECONDS,
                        help="keep only windows of exactly this duration; 0 keeps every length")
    parser.add_argument("--all-unknown", action="store_true",
                        help="also pack windows the reference left entirely unknown; they score "
                             "1.00 or near zero on the one class in play and show nothing")
    parser.add_argument("--pick", default=figures.SPREAD, choices=figures.PICKS)
    parser.add_argument("--normalise", default=WINDOW,
                        help="`data.normalise`; the test pass has always used per-window")
    parser.add_argument("--seed", type=int, default=0, help="only read by --pick random")
    args = parser.parse_args()

    if args.probe:
        return probe(args.probe)
    if not args.checkpoint:
        parser.error("--checkpoint is required (or --probe IP)")

    dataset = dataset_of(args.checkpoint, args.dataset)
    model = load_model(str(args.checkpoint))
    fold = int(model.hparams["fold"])
    label_source = str(model.hparams["label_source"])
    every = split_for(load_windows(dataset), fold)[args.split]
    frame = one_length(every, args.seconds)
    print(f"{args.checkpoint}\n  dataset {dataset}, fold {fold}, {len(every)} "
          f"{args.split} windows, labelled by `{label_source}`")
    if args.seconds:
        print(f"  {len(frame)} of them are exactly {args.seconds:g} s - the others are dropped, "
              f"so a latency spread is the device's and not the length's")
    if not args.all_unknown:
        kept = labelled_only(frame)
        print(f"  {len(frame) - len(kept)} of those carry no phase at all and are dropped too "
              f"(--all-unknown keeps them)")
        frame = kept
    if frame.empty:
        print(f"nothing left to pack from the {args.split} split of {dataset}")
        return 2

    items = []
    for _, row in frame.iterrows():
        values, reference = window_samples(dataset, row)
        logits, prediction = answer(model, values, args.normalise)
        items.append({"values": values, "reference": reference, "logits": logits,
                      "prediction": prediction, "row": row,
                      "score": figures.score(prediction, reference)})
    chosen = figures.choose(items, args.n, args.pick, args.seed)

    folder = Path(__file__).absolute().parent
    clear(folder)
    warn_about_strays(folder)
    decoding = dict(model.hparams["training"]["decoding"])
    manifest = {
        "created": stamp(),
        "checkpoint": str(args.checkpoint),
        "dataset": str(dataset),
        "fold": fold,
        "split": args.split,
        "label_source": label_source,
        # The device builds a net this wide and scores against these names.
        "classes": list(model.classes),
        "normalise": args.normalise,
        "fps": float(model.hparams["fps"]),
        "n_windows": len(chosen),
        "window_seconds": args.seconds or None,
        "window_stride_seconds": stride_seconds(every),
        "signal_seconds": signal_seconds(every),
        "n_parameters": int(model.net.n_parameters()),
        "receptive_field": int(model.net.receptive_field()),
        "model": {"in_channels": int(model.hparams["model"]["in_channels"]),
                  "channels": [int(c) for c in model.hparams["model"]["channels"]],
                  "bottleneck": int(model.hparams["model"]["bottleneck"]),
                  "kernel_size": int(model.hparams["model"]["kernel_size"]),
                  "dropout": float(model.hparams["model"].get("dropout", 0.0))},
        # Resolved here, not on the device: the per-source table lives in
        # `models.lightning_module`, which needs Lightning to import.
        "decoding": {"viterbi": bool(decoding.get("viterbi")),
                     "switch_penalty": float(decoding["switch_penalty"]),
                     "allowed": allowed_for(decoding, label_source,
                                            bool(model.hparams.get("stop_as_exhale", False))),
                     "enforce_min": bool(decoding.get("enforce_min")),
                     "min_duration": dict(decoding["min_duration"])},
        "logit_tolerance": runner.LOGIT_TOLERANCE,
        # The palette travels rather than being read from `parameter/` at draw time: the figures
        # are drawn where the run happened, and Hydra is not on a device.
        "plot": OmegaConf.to_container(
            OmegaConf.load(REPO_ROOT / "parameter" / "config.yaml").plot, resolve=True),
        "host": {"python": sys.version.split()[0], "torch": torch.__version__,
                 "numpy": np.__version__, "machine": platform.machine(),
                 "platform": platform.platform()},
        "host_macro_f1": round(float(np.mean([item["score"] for item in chosen])), 4),
        "mean_samples": round(float(np.mean([item["values"].size for item in chosen])), 1),
        "class_balance": {name: int(sum(int((item["reference"] == index).sum())
                                        for item in chosen))
                          for index, name in enumerate(model.classes)},
    }

    torch.save({"state_dict": net_state(model), "model": manifest["model"]},
               folder / runner.WEIGHTS)
    manifest["windows"] = write_waveforms(folder, chosen)
    copy_modules(folder)
    (folder / runner.MANIFEST).write_text(json.dumps(manifest, indent=2))

    size = sum(path.stat().st_size for path in folder.rglob("*")
               if path.is_file() and "__pycache__" not in str(path))
    print(f"\n{len(chosen)} windows, {manifest['n_parameters']:,} parameters, "
          f"host macro F1 {manifest['host_macro_f1']:.2f} ({args.pick} of {len(items)})")
    for window in manifest["windows"]:
        print(f"  {window['env']:>8} {window['window_id']:>8} {window['patient']:>10} "
              f"{window['samples']:>5} samples  macro F1 {window['host_macro_f1']:.2f}")
    print(f"\n{folder}  ({size / 1024:.0f} KiB) - copy this whole folder to the device\n")
    where = f"{DEVICE_USER}@{args.device_ip}"
    print(f"  scp -r {folder.name} {where}:{args.device_dir}")
    print(f"  ssh {where} 'cd {args.device_dir} && {args.device_python} "
          f"run_inference.py --runs 10'")
    print(f"  {runner.fetch_command(args.device_dir, args.device_ip)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
