# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

Training for per-sample inhale/exhale/stop/unknown segmentation of radar respiration waveforms
at 10 fps. Research repo - nothing here is deployed, and nothing here is called by
CloudAnalytics.

The first dataset is the **production algorithm's own output**, so every metric measured against
it is imitation rather than correctness. `README.md` states the caveats; do not write a summary
that drops them.

## One dependency: the labelling app repo

Nothing about the signal is decided here. `respiration-phase-labeling` owns the connections to
both instances, the `RespirationWindow` table, the window blobs in S3 and the call into
production's phase calculation; it reaches `inhale-exhale-detection`, which wraps `holosissystem`
**0.6.6** unmodified. Reached through `phase/bridge.py`.

This repo **reads what that one built**. It does not run the pipeline, does not download a raw
scan, and does not write to any database. To get more windows, run `upload_windows.py` there.

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

- **A window blob is immutable.** The app repo refuses to overwrite one, because
  `fast_small_kmeans` is unseeded and a rebuild is a different trace that stored labels would no
  longer describe. Shards inherit that: a shard on disk is never rebuilt.
- **`browse()` has no "labelled" column.** It has `Spans` (phase records on the window),
  `Labelers` and `LastLabelled`. Filtering on a column that is not there silently keeps every
  row - that mistake reported 1,933 labelled windows where there were 21.
- **A span's `EndIndex` is inclusive; a window's is exclusive.** Adjacent spans share a boundary
  sample. Off by one turns every human boundary into a systematic bias.
- **`unknown` is structural, not only "uncallable".** Production emits no phase for the turn from
  inhale to exhale, so the class sits at the crest of most breaths in this dataset.
- **The decoder's allowed transitions describe the label set**, not physiology. They change if
  the labels ever come from people.
- **`save_hyperparameters` pickles what it is handed**, and `torch.load` defaults to
  `weights_only=True` - pass class weights as a list, never a numpy array, or the checkpoint
  cannot be reloaded.
