# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

Training for per-sample inhale/exhale/stop/unknown segmentation of radar respiration waveforms
at 10 fps. Research repo - nothing here is deployed, and nothing here is called by
CloudAnalytics.

The first dataset is the **production algorithm's own output**, so every metric measured against
it is imitation rather than correctness. `README.md` states the caveats; do not write a summary
that drops them.

## Three repos, one pipeline

Nothing about the signal is decided here. This repo imports:

- `respiration-phase-labeling` - signal selection, sampling, the raw-scan run, orientation, and
  production's phases per window. Reached through `phase/bridge.py`.
- `inhale-exhale-detection` - the wrapper that runs `holosissystem` unmodified.
- `holosissystem` **0.6.6**, from the release index - it decides the numerics of every window.

Do not reimplement any of that here. If something is missing, add it there.

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

- **A built shard is never rebuilt.** `fast_small_kmeans` is unseeded, so a rebuild is a
  different trace. Resumption exists for correctness, not economy.
- **`unknown` is structural, not only "uncallable".** Production emits no phase for the turn from
  inhale to exhale, so the class sits at the crest of most breaths in this dataset.
- **The decoder's allowed transitions describe the label set**, not physiology. They change if
  the labels ever come from people.
- **`save_hyperparameters` pickles what it is handed**, and `torch.load` defaults to
  `weights_only=True` - pass class weights as a list, never a numpy array, or the checkpoint
  cannot be reloaded.
