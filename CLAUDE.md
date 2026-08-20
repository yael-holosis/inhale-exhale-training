# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

Training for per-sample inhale/exhale/stop/unknown segmentation of radar respiration waveforms
at 10 fps. Research repo - nothing here is deployed, and nothing here is called by
CloudAnalytics.

The first dataset is the **production algorithm's own output**, so every metric measured against
it is imitation rather than correctness. `README.md` states the caveats; do not write a summary
that drops them.

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

- Parameters live in `parameter/` (Hydra), not in code. Named `parameter/` singular so it cannot
  shadow the labelling repo's `parameters` package when that checkout joins `sys.path`.
- No repeated literals. Phase names and indices come from `phase/labels.py`; nothing else defines
  them.
- Generated artifacts go to `data_sets/` (shards), `outputs/` (Hydra runs) or `out/`. Never the
  repo root.
- Keep comments short; a comment says *why*, not what the line does. Rationale belongs in the
  PR/commit/Jira.
- Tests must not need AWS, a built dataset or a GPU. `tests/synthetic.py` stands in for one.

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
- **A span's `EndIndex` is inclusive; a window's is exclusive.** Adjacent spans share a boundary
  sample. Off by one turns every human boundary into a systematic bias.
- **`unknown` is structural, not only "uncallable".** Production emits no phase for the turn from
  inhale to exhale, so the class sits at the crest of most breaths in this dataset.
- **The decoder's allowed transitions describe the label set**, not physiology. They change if
  the labels ever come from people.
- **`save_hyperparameters` pickles what it is handed**, and `torch.load` defaults to
  `weights_only=True` - pass class weights as a list, never a numpy array, or the checkpoint
  cannot be reloaded.
