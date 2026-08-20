"""Build the training set from production's own algorithm, for one environment.

Samples signals per patient, runs the production pipeline on each raw scan, and stores every
window the inhale/exhale calculation was given together with the phases it called on that
window. One `.npz` per signal plus a manifest row per window.

    poetry run python build_dataset.py --env ds_prod --patients bs- RM- --per-patient 100

Options
-------

`--env ds_algo | ds_prod`   Which instance. `ds_prod` reads production's patients and raw scans
    and **writes nothing to production**; `ds_algo` is the data-science cohort, which is where
    the sleep-lab nights are. Defaults to `data.env`.

`--patients PREFIX [...]`   Patients by prefix, matching either the study name (`SL0066`, `SL`)
    or the patient key (`bs-`, `RM-`). Omit for the config's list; `--patients` with no value
    for every patient.

`--per-patient N`   Signals sampled from each patient, **not in total**. Sampling is seeded and
    spread across sessions, so a patient with a thousand sessions does not contribute a thousand
    near-identical minutes of one night.

`--limit N`   Stop after N signals. For a first run - the full one takes hours.

`--out DIR`   Where the shards go. Defaults to `data.dir`.

Resumable: a signal whose shard is already on disk is not rebuilt. That is correctness rather
than economy - `fast_small_kmeans` draws from the unseeded global RNG, so a rebuild produces a
different trace, and a set half-built from each would be two datasets in one directory.

Roughly three to four seconds per signal, most of it the raw-scan download. The scans are
discarded after use by the labelling repo's own rule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from phase import bridge
from phase.building import SUMMARY_NAME, build
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
    parser.add_argument("--per-patient", type=int, default=cfg.data.signals_per_patient)
    parser.add_argument("--seed", type=int, default=cfg.data.sample_seed)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    cfg = load_config()
    args = parse_args(cfg)
    patients = args.patients if args.patients is not None else list(cfg.data.patients or [])
    out_dir = Path(args.out or cfg.data.dir)
    labeling_repo = str(cfg.repos.labeling)

    available, reason = bridge.labeling(labeling_repo)
    if available is None:
        print(f"cannot build: {reason}")
        return 1

    print(f"env {args.env} | patients {patients or 'all'} | {args.per_patient}/patient "
          f"| seed {args.seed} -> {out_dir}")
    manifest, stats = build(env_key=args.env, patients=patients or None,
                            per_patient=args.per_patient, seed=args.seed, out_dir=out_dir,
                            labeling_repo=labeling_repo,
                            requires_rate=bool(cfg.data.requires_rate), limit=args.limit)

    if manifest.empty:
        print("nothing built")
        return 1

    total = max(stats.samples, 1)
    summary = {
        "env": args.env, "patients": patients, "per_patient": args.per_patient,
        "seed": args.seed,
        "signals": {"planned": stats.signals_planned, "built": stats.signals_built,
                    "failed": stats.signals_failed},
        "windows": {"kept": stats.windows_kept, "no_phases": stats.windows_no_phases,
                    "rejected_by_production": stats.windows_rejected},
        "samples": stats.samples,
        "hours": round(stats.samples / 10.0 / 3600.0, 2),
        "per_class": stats.per_class,
        "per_class_fraction": {name: round(count / total, 4)
                               for name, count in stats.per_class.items()},
        "versions": bridge.versions(),
        "failures": stats.failures[:50],
    }
    with open(out_dir / SUMMARY_NAME, "w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    print(f"\n{stats.signals_built} signals, {stats.windows_kept} windows, "
          f"{stats.samples:,} samples ({summary['hours']} h)")
    for name in PHASES:
        print(f"  {name:8s} {stats.per_class[name]:9,d}  "
              f"{100 * stats.per_class[name] / total:5.1f}%")
    if stats.signals_failed:
        print(f"\n{stats.signals_failed} signals failed - first few:")
        for line in stats.failures[:5]:
            print(f"  {line}")
    print(f"\nmanifest: {out_dir}/manifest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
