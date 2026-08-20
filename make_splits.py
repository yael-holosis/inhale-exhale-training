"""Assign the splits for a dataset, and write them into its own `windows.csv`.

    poetry run python make_splits.py --dataset latest
    poetry run python make_splits.py --dataset data_sets/phases_algorithm_20260820T165400Z

Separate from the build on purpose: re-splitting - a new seed, different stratification, more
folds - is a second of work, and having to re-download a dataset to change how it is cut would
mean nobody ever changes how it is cut.

Adds one `split` column (train / test) and one `fold_{i}_split` per fold (train / val / test).
**Test is held out once and stays test in every fold**, so every fold reports against the same
patients and the numbers are comparable. Folds rotate only the validation set.

Everything about how is in `data.split`: `test_fraction`, `folds`, `seed`, `group_columns` (the
unit a split may not cut through) and `stratify_cols` (what is balanced across the parts).
Overridable per run with the flags below.

Refuses to overwrite an existing assignment without `--force`, because a model has probably been
trained against it and a silent re-split makes its reported numbers describe a split that no
longer exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from phase.building import (PARAMS_NAME, STATS_NAME, existing_params, load_windows, resolve,
                            summarise, write_windows)
from phase.splits import FOLD_COLUMN, SPLIT_COLUMN, describe, make_splits, split_for

CONFIG_DIR = Path(__file__).parent / "parameter"


def load_config():
    root = OmegaConf.load(CONFIG_DIR / "config.yaml")
    data = OmegaConf.load(CONFIG_DIR / "data" / f"{root.defaults[0]['data']}.yaml")
    return OmegaConf.merge(root, {"data": data})


def main() -> int:
    cfg = load_config()
    cut = OmegaConf.to_container(cfg.data.split, resolve=True)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=cfg.data.dir)
    parser.add_argument("--test-fraction", type=float, default=cut["test_fraction"])
    parser.add_argument("--folds", type=int, default=cut["folds"])
    parser.add_argument("--seed", type=int, default=cut["seed"])
    parser.add_argument("--stratify", nargs="*", default=None, metavar="COLUMN",
                        help=f"override split.stratify_cols (default {cut['stratify_cols']}); "
                             f"pass with no values to disable stratification")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing assignment")
    args = parser.parse_args()

    directory = resolve(cfg.data.root, args.dataset)
    frame = load_windows(directory)
    already = [c for c in frame.columns if c == SPLIT_COLUMN or c.endswith("_split")]
    if already and not args.force:
        print(f"{directory} already carries {', '.join(already)}.\n"
              f"A model has probably been trained against it - re-splitting silently would make "
              f"its numbers describe a split that no longer exists.\nPass --force to replace it.")
        return 1
    frame = frame.drop(columns=already)

    cut.update({"test_fraction": args.test_fraction, "folds": args.folds, "seed": args.seed})
    if args.stratify is not None:
        cut["stratify_cols"] = list(args.stratify)

    frame = make_splits(frame, cut)
    write_windows(directory, frame)

    facts = summarise(frame)
    params = existing_params(directory)
    params["split"] = cut
    with open(Path(directory) / PARAMS_NAME, "w") as handle:
        yaml.safe_dump(params, handle, sort_keys=False)
    with open(Path(directory) / STATS_NAME, "w") as handle:
        yaml.safe_dump(facts, handle, sort_keys=False)

    print(f"{directory}\n  grouped by {'+'.join(cut['group_columns'])}"
          + (f", stratified on {'+'.join(cut['stratify_cols'])}"
             if cut["stratify_cols"] else ", not stratified")
          + f", seed {cut['seed']}")
    held = frame[frame[SPLIT_COLUMN] == "test"]
    print(f"\ntest, held out of every fold: {len(held)} windows "
          f"({100 * len(held) / len(frame):.1f}%), "
          f"{held['PatientID'].nunique()} patients\n"
          f"  {', '.join(sorted(held['PatientID'].astype(str).unique()))}")
    for fold in range(cut["folds"]):
        print(f"\nfold {fold}")
        print(describe(split_for(frame, fold), total=len(frame),
                       stratify_cols=cut["stratify_cols"]))
    print(f"\nwritten: {directory}/windows.csv "
          f"({SPLIT_COLUMN}, {FOLD_COLUMN.format(fold='0..%d' % (cut['folds'] - 1))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
