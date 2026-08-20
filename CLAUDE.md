# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

Training for per-sample inhale/exhale/stop/unknown segmentation of radar respiration waveforms
at 10 fps. Research repo - nothing here is deployed, and nothing here is called by
CloudAnalytics.

Labels come from `data.labels.source`: `algorithm` (production's own phase calculation, every
window) or `human` (`BreathPhaseTimeRecord`, a few dozen). The first datasets are `algorithm`,
because that is what there is enough of - so every metric measured against them is **imitation
rather than correctness**. `README.md` states the caveats; do not write a summary that drops them.

## Standalone, and it has to stay that way

No sibling checkout is imported. The dependencies are `holosissystem` **0.6.6** (pip, from the
release index - it decides the numerics of every label) and `holosis_aws_manager`. Databases and
bucket are in `parameter/sources/default.yaml`.

`tests/test_standalone.py` enforces it: no import of a neighbouring repo, no `sys.path`
manipulation, no `password:` in a tracked file, and reads only through `phase.sources.frame`.

**This repo reads. It never writes** - not to a database, not to S3. Growing the window pool is
somebody else's job.

`phase/production.py` is a port of the phase call and span pairing that `inhale-exhale-detection`
validated. It was checked window for window against that implementation before the dependency was
cut - 60/60 identical labels on real windows. If the upstream phase code changes, this file has to
follow it.

## Conventions

- Parameters live in `parameter/` (Hydra), not in code. Nothing about the build, the labels or the
  split is decided in a module.
- **A dataset directory is the unit and it is self-describing.**
  `data_sets/<name>_<source>_<UTC stamp>/` carries `build_params.yaml` (every parameter, one entry
  per run), `windows.csv` (a row per window, including which split it went to), `stats.yaml`, and
  the shards. A run pins itself to one directory and writes its name into the run dir.
- **Splits live in the dataset, not in the config.** `make_splits.py` writes `split` and
  `fold_i_split` into `windows.csv`; `train.py` reads those columns and never recomputes them.
  Re-splitting an existing dataset needs `--force`, because a model has probably been trained
  against the current assignment.
- No repeated literals. Phase names and indices come from `phase/labels.py`; nothing else defines
  them.
- Generated artifacts go to `data_sets/`, `outputs/` (Hydra runs) or `out/`. Never the repo root.
- Keep comments short; a comment says *why*, not what the line does.
- Tests must not need AWS, a built dataset or a GPU. `tests/synthetic.py` stands in for one, and
  can emit two cohorts so stratification has something to balance.

## Traps

- **A window blob is immutable.** `fast_small_kmeans` is unseeded, so a rebuild is a different
  trace that stored labels would no longer describe. Nothing overwrites one; shards inherit that,
  so a shard on disk is never rebuilt.
- **There is no "labelled" column.** `Spans` counts the phase records on a window. Filtering on a
  column that is not there silently keeps every row - that mistake reported 1,933 labelled
  windows where there were 21.
- **Production's durations are means over ALL pairs**, including the ones `_valid` rejects. Filter
  first and the pairing looks broken when it is not.
- **`inhale_time_sec` describes the trace production RETURNED.** Where that is the negation of its
  input, our spans - which are mapped back onto the input - have inhale and exhale exchanged
  relative to it. `tests/test_production.py` asserts exactly that correspondence.
- **A span's `EndIndex` is inclusive; a window's is exclusive.** Off by one turns every human
  boundary into a systematic bias.
- **Balanced assignment is deficit-first, not nearest-target.** Scoring a greedy split by distance
  from the final target sends every early group to the part with the smallest target - being under
  is penalised like being over - which put 95% of the windows in a 20% test set. Assign to
  whichever part is furthest short.
- **`human_spans` in `windows.csv` is a snapshot at build time**, unlike every other column:
  labelling continues after the dataset is built.
- **A span's `EndIndex` is inclusive; a window's is exclusive.** Adjacent spans share a boundary
  sample. Off by one turns every human boundary into a systematic bias.
- **`unknown` is structural, not only "uncallable".** Production emits no phase for the turn from
  inhale to exhale, so the class sits at the crest of most breaths in this dataset.
- **The decoder's allowed transitions describe the label set**, not physiology. They change if
  the labels ever come from people.
- **`save_hyperparameters` pickles what it is handed**, and `torch.load` defaults to
  `weights_only=True` - pass class weights as a list, never a numpy array, or the checkpoint
  cannot be reloaded.
