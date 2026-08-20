"""Build the training set from the windows already uploaded to `RespirationWindow`.

    poetry run python build_dataset.py --env ds_algo --patients SL
    poetry run python build_dataset.py --env ds_prod

Reads each window's samples from S3 and asks production's own inhale/exhale calculation what it
calls on them. **No raw scan is downloaded** - the app repo already ran the pipeline when it
uploaded these windows, and this reads what it stored. One `.npz` per signal plus a manifest row
per window, and every row carries `RespirationWindowID` so a human label written later joins
straight onto the window the network trained on.

Options
-------

`--env ds_algo | ds_prod`   Which instance. `ds_algo` is the data-science cohort - the `SL`
    sleep-lab nights; `ds_prod` is the pilots (`bs-`, `RM-`). Defaults to `data.env`.

    **The two are separate runs into the same directory.** Their `RadarSignal` ID spaces are
    unrelated, so a shard is named `<env>_signal_<id>.npz` and the manifest is rebuilt from
    every shard present.

`--patients PREFIX [...]`   Match the display name (`SL0066`, `SL`) or the patient key
    (`algo-p089`, `bs-`). Omit for every patient with uploaded windows.

`--signals ID [...]`   Exactly these radar signals, no sampling.

`--per-patient N`   Cap the **signals** one patient contributes, sampled seeded and spread
    across sessions. Omit to take everything uploaded.

`--unlabelled-only`   Hold back signals a person has already labelled, so they stay a clean test
    set. Off by default: the human spans are read against these windows either way, and holding
    them out of training costs data while there are only a handful of them.

`--limit N`   Stop after N signals. For a first look.

The pool is whatever is in `RespirationWindow`. Growing it means uploading more windows, which
is a write - and **this repo has no writer**, by design. Reads only, both instances.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from phase import production, sources
from phase.building import SUMMARY_NAME, build, catalogue
from phase.labels import PHASES

CONFIG_DIR = Path(__file__).parent / "parameter"


def load_config():
    """The Hydra tree, read directly - a build is not a sweep and does not need the launcher."""
    root = OmegaConf.load(CONFIG_DIR / "config.yaml")
    data = OmegaConf.load(CONFIG_DIR / "data" / f"{root.defaults[0]['data']}.yaml")
    return OmegaConf.merge(root, {"data": data})


def parse_args(cfg):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=cfg.data.env)
    parser.add_argument("--patients", nargs="*", default=None, metavar="PREFIX")
    parser.add_argument("--signals", nargs="*", type=int, default=None, metavar="ID")
    parser.add_argument("--per-patient", type=int, default=cfg.data.signals_per_patient)
    parser.add_argument("--unlabelled-only", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.data.sample_seed)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--catalogue", action="store_true",
                        help="report what is uploaded on this instance and write nothing")
    return parser.parse_args()


def show_catalogue(env_key: str) -> int:
    frame = catalogue(env_key)
    if frame.empty:
        print(f"no windows uploaded on {env_key}")
        return 1
    labelled = frame[frame["Spans"] > 0]
    print(f"{env_key}: {len(frame):,} windows, {frame['RadarSignalID'].nunique()} signals, "
          f"{frame['Patient'].nunique()} patients")
    print(f"  {len(labelled)} windows carry human spans "
          f"({labelled['RadarSignalID'].nunique()} signals)")
    per = frame.groupby("Patient").agg(windows=("ID", "size"),
                                       signals=("RadarSignalID", "nunique"),
                                       labelled=("Spans", lambda s: int((s > 0).sum())))
    print(per.sort_values("windows", ascending=False).to_string())
    return 0


def main() -> int:
    cfg = load_config()
    args = parse_args(cfg)
    if not sources.ensure_session():
        print(f"cannot reach AWS as {sources.profile()} - "
              f"run: aws sso login --profile {sources.profile()}")
        return 1
    if args.catalogue:
        return show_catalogue(args.env)

    patients = args.patients if args.patients is not None else list(cfg.data.patients or [])
    out_dir = Path(args.out or cfg.data.dir)
    print(f"env {args.env} | patients {patients or 'all'} "
          f"| per-patient {args.per_patient or 'all'} -> {out_dir}")

    manifest, stats = build(env_key=args.env, out_dir=out_dir,
                            patients=patients or None, signals=args.signals,
                            per_patient=args.per_patient,
                            unlabelled_only=args.unlabelled_only, seed=args.seed,
                            requires_rate=bool(cfg.data.requires_rate), limit=args.limit)
    if manifest.empty:
        print("nothing built")
        return 1

    per_class = {name: int(manifest[f"n_{name}"].sum()) for name in PHASES}
    samples = max(sum(per_class.values()), 1)
    summary = {
        "run": {"env": args.env, "patients": patients, "per_patient": args.per_patient,
                "seed": args.seed, "unlabelled_only": args.unlabelled_only,
                "signals": {"planned": stats.signals_planned, "built": stats.signals_built,
                            "failed": stats.signals_failed},
                "windows": {"kept": stats.windows_kept,
                            "no_phases": stats.windows_no_phases},
                "failures": stats.failures[:50]},
        "dataset": {
            "environments": sorted(manifest["env"].astype(str).unique()),
            "patients": int(manifest["PatientID"].nunique()),
            "signals": int(manifest["RadarSignalID"].nunique()),
            "windows": int(len(manifest)),
            "windows_without_phases": int((manifest["n_spans"] == 0).sum()),
            "samples": samples,
            "hours": round(samples / 10.0 / 3600.0, 2),
            "per_class": per_class,
            "per_class_fraction": {name: round(count / samples, 4)
                                   for name, count in per_class.items()},
        },
        "holosissystem": production.version(),
    }
    with open(out_dir / SUMMARY_NAME, "w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    data = summary["dataset"]
    print(f"\nthis run: {stats.signals_built} signals, {stats.signals_failed} failed")
    print(f"dataset:  {data['signals']} signals, {data['windows']} windows "
          f"({data['windows_without_phases']} with no phases), {samples:,} samples "
          f"({data['hours']} h), {data['patients']} patients over "
          f"{', '.join(data['environments'])}")
    for name in PHASES:
        print(f"  {name:8s} {per_class[name]:9,d}  {100 * per_class[name] / samples:5.1f}%")
    if stats.signals_failed:
        print(f"\n{stats.signals_failed} signals failed - first few:")
        for line in stats.failures[:5]:
            print(f"  {line}")
    print(f"\nmanifest: {out_dir}/manifest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
