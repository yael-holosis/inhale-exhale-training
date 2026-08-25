"""Build a dataset directory from the windows already in `RespirationWindow`.

    poetry run python build_dataset.py --catalogue          # what is there, no writes
    poetry run python build_dataset.py                      # a new dataset, every instance
    poetry run python build_dataset.py --env ds_algo        # one instance only

Each build writes a **new timestamped directory**, self-describing on disk:

    data_sets/phases_algorithm_20260820T165400Z/
      build_params.yaml            every parameter that decided it, and the versions
      windows.csv                  one row per window: provenance and class counts
      stats.yaml                   hours, class balance, per-patient and per-cohort counts
      ds_algo_signal_1557424.npz   samples and per-sample targets, one file per signal

Then assign the splits, which is a separate step so a re-split needs no rebuild:

    poetry run python make_splits.py --dataset latest

Labels come from `data.labels.source` - `algorithm` (production's own phase calculation, every
window) or `human` (`BreathPhaseTimeRecord`, a few dozen windows). A directory holds one source;
`--into` refuses to mix them.

**No raw scan is downloaded.** About 1.7 s per signal, all of it S3 and the database.

Options
-------

`--env ds_algo [ds_prod ...]`   Which instances, all into one directory. Defaults to
    `data.env`, which is both. `ds_algo` is the data-science cohort - the `SL` sleep-lab nights;
    `ds_prod` is the pilots. Production is only ever read.

`--into DIR`   Add to an existing dataset instead of starting one. `latest` resolves to the
    newest.

`--patients PREFIX [...]`   Match the display name (`SL0066`, `SL`) or the patient key (`bs-`).

`--exclude PREFIX [...]`   Drop these patients even if `--patients` would have taken them.
    Defaults to `data.exclude_patients`, which holds the QA rig.

`--signals ID [...]`   Exactly these radar signals, no sampling.

`--per-patient N`   Cap the signals one patient contributes, sampled seeded and spread across
    sessions. Omit to take everything uploaded.

`--labels algorithm | human`   Override `data.labels.source` for this run.

`--limit N`   Stop after N signals. For a first look.

`--refresh-cache`   Re-download every window. The cache holds raw blobs only, and a blob
    is immutable, so this is for a corrupted cache rather than for picking up new labels - labels
    and `ReviewerFlipped` are read from the database on every build regardless.

`--catalogue`   Report what is uploaded on an instance and write nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from phase import sources
from phase.building import (PARAMS_NAME, STATS_NAME, WINDOWS_NAME, build, catalogue,
                            dataset_dir, existing_params, resolve, stamp, summarise,
                            write_provenance)
from phase.labels import PHASES
from phase.labelsources import SOURCES, LabelSource

CONFIG_DIR = Path(__file__).parent / "parameter"
# Above this share of failed signals the build is partial, not merely small.
FAILURE_LIMIT = 0.05


def load_config():
    """The Hydra tree, read directly - a build is not a sweep and does not need the launcher."""
    root = OmegaConf.load(CONFIG_DIR / "config.yaml")
    data = OmegaConf.load(CONFIG_DIR / "data" / f"{root.defaults[0]['data']}.yaml")
    return OmegaConf.merge(root, {"data": data})


def as_list(value) -> list[str]:
    """`data.env` takes one instance or several; a bare string is not a list of characters."""
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def parse_args(cfg):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", nargs="+", default=as_list(cfg.data.env), metavar="KEY",
                        help="one or more instances, built into a single dataset directory")
    parser.add_argument("--into", default=None,
                        help="extend this dataset directory instead of starting a new one")
    parser.add_argument("--labels", choices=list(SOURCES), default=None,
                        help=f"override data.labels.source (default {cfg.data.labels.source})")
    parser.add_argument("--patients", nargs="*", default=None, metavar="PREFIX")
    parser.add_argument("--exclude", nargs="*", default=None, metavar="PREFIX",
                        help="patient prefixes to drop (default data.exclude_patients)")
    parser.add_argument("--signals", nargs="*", type=int, default=None, metavar="ID")
    parser.add_argument("--per-patient", type=int, default=cfg.data.signals_per_patient)
    parser.add_argument("--seed", type=int, default=cfg.data.sample_seed)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true",
                        help="re-download every window instead of using the cache")
    parser.add_argument("--catalogue", action="store_true")
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
          f"({labelled['RadarSignalID'].nunique()} signals) - the whole `--labels human` pool")
    per = frame.groupby("Patient").agg(windows=("ID", "size"),
                                       signals=("RadarSignalID", "nunique"),
                                       human=("Spans", lambda s: int((s > 0).sum())))
    print(per.sort_values("windows", ascending=False).to_string())
    return 0


def target_directory(args, cfg, source: str) -> Path:
    """A new timestamped directory, or an existing one that was built the same way."""
    if not args.into:
        return dataset_dir(cfg.data.root, cfg.data.name, source)
    directory = resolve(cfg.data.root, args.into)
    previous = existing_params(directory).get("labels", {}).get("source")
    if previous and previous != source:
        raise SystemExit(
            f"{directory} was built with labels.source={previous!r} and this run is {source!r}. "
            f"They are different targets; build a new dataset rather than mixing them.")
    return directory


def main() -> int:
    cfg = load_config()
    args = parse_args(cfg)
    if not sources.ensure_session():
        print(f"cannot reach AWS as {sources.profile()} - "
              f"run: aws sso login --profile {sources.profile()}")
        return 1
    if args.catalogue:
        return max(show_catalogue(env_key) for env_key in args.env)

    label_cfg = OmegaConf.to_container(cfg.data.labels, resolve=True)
    if args.labels:
        label_cfg["source"] = args.labels

    # One directory for every instance: the split stratifies on `env`, so both must be present.
    source = LabelSource(args.env[0], label_cfg).source
    out_dir = target_directory(args, cfg, source)
    patients = args.patients if args.patients is not None else list(cfg.data.patients or [])
    excluded = (args.exclude if args.exclude is not None
                else list(cfg.data.get("exclude_patients") or []))
    print(f"env {', '.join(args.env)} | labels {source} | patients {patients or 'all'}"
          f"{' | excluding ' + ', '.join(excluded) if excluded else ''} "
          f"| per-patient {args.per_patient or 'all'}\n-> {out_dir}")

    frame, built, failed = None, 0, 0
    for env_key in args.env:
        labels = LabelSource(env_key, label_cfg)
        started = stamp()
        print(f"\n[{env_key}]")
        frame, stats = build(env_key=env_key, out_dir=out_dir, labels=labels,
                             patients=patients or None, signals=args.signals,
                             per_patient=args.per_patient, seed=args.seed, limit=args.limit,
                             exclude_patients=excluded or None,
                             refresh_cache=args.refresh_cache)
        built, failed = built + stats.signals_built, failed + stats.signals_failed
        if frame.empty:
            continue
        # One entry per instance, so a directory records every invocation that filled it.
        write_provenance(out_dir, {
            "started": started, "finished": stamp(), "env": env_key, "patients": patients,
            "exclude_patients": excluded,
            "per_patient": args.per_patient, "signals": args.signals, "sample_seed": args.seed,
            "limit": args.limit, "labels": labels.describe(),
            "signals_planned": stats.signals_planned, "signals_built": stats.signals_built,
            "signals_failed": stats.signals_failed, "failures": stats.failures[:50],
        }, frame)

    if frame is None or frame.empty:
        print("nothing built")
        return 1

    facts = summarise(frame)
    cached = sources.CACHE_HITS
    print(f"\nthis run: {built} signals, {failed} failed"
          + (f", {cached} windows from cache, {sources.CACHE_MISSES} downloaded"
             if sources.cache_root() else ", cache off"))
    print(f"dataset:  {facts['signals']} signals, {facts['windows']} windows "
          f"({facts['windows_unlabelled']} carry no label), {facts['samples']:,} samples "
          f"({facts['hours']} h), {facts['patients']} patients over "
          f"{', '.join(facts['environments'])}")
    for name in PHASES:
        share = 100 * facts["per_class"][name] / facts["samples"]
        print(f"  {name:8s} {facts['per_class'][name]:9,d}  {share:5.1f}%")
    if failed:
        print(f"\n{failed} signals failed - first few:")
        for line in stats.failures[:5]:
            print(f"  {line}")
    # A build that lost a large share of its signals is a partial dataset, not a small one, and
    # exiting 0 lets it flow into training as though it were complete. Credentials expiring
    # mid-download is the way this actually happens.
    planned = built + failed
    if planned and failed / planned > FAILURE_LIMIT:
        print(f"\nREFUSING: {failed} of {planned} signals failed "
              f"({100 * failed / planned:.0f}%, limit {100 * FAILURE_LIMIT:.0f}%). "
              f"{out_dir} is partial - fix the cause and build again.")
        return 2
    print(f"\n{out_dir}/  ({WINDOWS_NAME}, {PARAMS_NAME}, {STATS_NAME})")
    print(f"next: poetry run python make_splits.py --dataset {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
