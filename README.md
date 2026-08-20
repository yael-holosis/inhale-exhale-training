# inhale-exhale-training

Train a per-sample **inhale / exhale / stop / unknown** segmenter on radar respiration
waveforms, at 10 fps, any length in and one class per sample out. Small enough for the edge
device: 18,188 parameters at the default shape.

The training set is **the production algorithm's own output** over the windows already stored in
`RespirationWindow`. Human labelling is under way but still small - a few dozen windows out of
several thousand - so the network starts by learning what the device already does, and moves onto
human labels as they arrive.

**This repo is standalone.** It reads two databases and an S3 bucket, and calls
`holosissystem`'s own phase function. It imports no sibling checkout, writes nothing anywhere,
and `tests/test_standalone.py` fails if either stops being true.

## Read this before quoting a number

**This is distillation, and its ceiling is the teacher.** Every label in the built dataset is
`calculate_inhale_exhale_time`'s answer on a window production itself produced. A model that
scores 1.00 here has cloned the current algorithm, wherever it is right and wherever it is not.

Two specific inheritances:

- **The boundaries are rise and fall times, not phase durations.** Production measures the 10%
  and 90% amplitude crossings, which capture a median 65% of the true trough-to-crest rise
  (`inhale-exhale-detection/claude/FINDINGS.md`, finding 5). A model trained on them reproduces
  truncated phases.
- **`unknown` means two different things in this dataset.** A labeller marks it where they could
  not call the trace. Production leaves it in the same places *and* at the turn from inhale to
  exhale on every breath - it has no phase for that turn, so no span is emitted there. Anything
  that reads `unknown` as "no breathing" will be wrong most of the time it fires.

What the exercise buys, if it works: one fully convolutional pass over a whole signal in place
of 20 s windows, k-means over range bins, growth retries and three sign decisions - and a
starting point that only needs fine-tuning once the human set is large enough to train on.
`evaluate.py --human` is the run that says whether either happened.

## Setup

Python **3.12** exactly (`holosissystem` requires below 3.13, and a caret range builds the
environment on whichever newer interpreter is on the machine).

```bash
poetry install
poetry run python -m pytest tests/ -q
```

Three things have to be reachable, and each is reported by name if it is not:

| What | Where | Override |
| --- | --- | --- |
| `respiration-phase-labeling` checkout | `repos.labeling_app` | `RESPIRATION_PHASE_LABELING_REPO` |
| ClearML credentials | `~/clearml.conf` | `DISABLE_CLEARML=true` to run without |

The app repo is the only dependency: it owns both connections, the `RespirationWindow` table,
the window blobs in S3 and the call into production's phase calculation, and it reaches
`inhale-exhale-detection` itself.

### Credentials and profiles

All of it is in `parameter/sources/default.yaml`, and **no password is in this repo**.
Credentials come from Secrets Manager through `holosis_aws_manager`; a test fails the build if a
`password:` key ever appears in a tracked file.

| | `ds_algo` (SL, sleep lab) | `ds_prod` (pilots) |
| --- | --- | --- |
| Windows + labels | data-science MySQL, SM `sm-data-science-01-db-password-mysql-6g2w8dxe`, db `edge_data_extras` | data-science replica, SM `sm-data-science-01-db-password-edge-data-endpoint-0owgu5br` |
| Signals + patients | same server, db `edge_data` | **production's** MySQL, user `edge_data_user_ro`, read only |
| Window samples | \multicolumn - one bucket for both: `s3-data-science-01-holosis-health-system-sessions` | |

- **Production is read-only by the server's rules**, not only by ours: `edge_data_user_ro` is
  granted SELECT and nothing else. And `phase.sources.frame` refuses anything that is not a
  read, so a writer added later fails in a test rather than on production.
- **Production's password is not in this repo.** It is read from Secrets Manager in the
  production account under `holosis-prod-admin`. Whoever has no access there sets
  `INHALE_EXHALE_TRAINING_PROD_RO_PASSWORD`, or drops it in `secrets/` - see
  [secrets/README.md](secrets/README.md).
- The two sides of an environment **never join in SQL**. On prod they are different servers, so
  the window and signal frames are merged in pandas and one code path serves both instances.

`build_dataset.py` signs in for you if the session has expired; to do it by hand:

```bash
aws sso login --profile holosis-datascience-algo
```

## Building a dataset

```bash
poetry run python build_dataset.py --catalogue --env ds_algo   # what is there, no writes
poetry run python build_dataset.py --env ds_algo               # a new dataset
poetry run python build_dataset.py --env ds_prod --into latest # the other cohort beside it
poetry run python make_splits.py --dataset latest              # assign the splits
```

Each build writes a **new timestamped directory**, self-describing on disk:

```
data_sets/phases_algorithm_20260820T143456Z/
  build_params.yaml            every parameter that decided it, plus versions; one entry per run
  windows.csv                  one row per window: provenance, class counts, split columns
  stats.yaml                   hours, class balance, per-patient / per-cohort / per-split counts
  ds_algo_signal_1557424.npz   samples and per-sample targets, one file per signal
```

Six months from now the only question that matters about a checkpoint is what it was trained on,
and the answer has to be readable off disk rather than reconstructed from a config that has moved
on. `data.dir: latest` resolves to the newest build; name a directory to pin a run to one dataset.

**No raw scan is downloaded.** The pipeline already ran when these windows were uploaded; this
reads each window's samples from S3 (`phase/sources.py`) and labels them. About 1.7 s per signal,
all of it S3 and the database, so there is no disk pressure at all.

**Two cohorts, two runs, one directory.** `ds_algo` is the data-science instance with the `SL`
sleep-lab nights; `ds_prod` is the pilots. Their `RadarSignal` ID spaces are unrelated - signal
2120091 is a different recording on each - so a shard is named `<env>_signal_<id>.npz` and
`windows.csv` is rebuilt from every shard present, carrying any split already assigned.

**Resumable**, and it costs nothing to be: a window blob is immutable once uploaded, so a shard
already on disk can never be stale.

### Where the labels come from

`data.labels.source`, and it is the choice the whole repo turns on:

| | `algorithm` | `human` |
| --- | --- | --- |
| What | production's own `calculate_inhale_exhale_time`, re-run on the stored window | `BreathPhaseTimeRecord` - the spans a person drew |
| How much | every window (2,848 today) | a few dozen |
| What it measures | imitation of the current algorithm | correctness |

`algorithm` is what there is enough of to train on, and it is **distillation**: the ceiling is the
current algorithm, and its boundaries are the 10% and 90% amplitude crossings rather than phase
durations.

**Every sample gets a class.** The detector returns spans, not a covering, so whatever it does not
claim is `unknown` - there are no gaps and no unlabelled samples in the dataset. Three things end
up there, and they are worth telling apart:

```
window 49, first 60 samples   (. unknown, I inhale, E exhale, S stop)
.....IIIII....EEEEEEEEESSSSSIIIIII..EEEEEEEEEESSSIIIIIIIII..
^^^^^      ^^^^                   ^^
edges      the crest              between breaths
```

- **The crest of every breath.** Production emits no phase for the turn from inhale to exhale, so
  `unknown` is structural there rather than a sign of a bad window.
- **Window edges**, before the first boundary and after the last.
- **Whole windows the detector found nothing in** - 248 of 2,848, kept deliberately. "The
  algorithm found nothing here" is a training signal, and dropping them would bias the set
  towards easy breathing.

That makes `unknown` 26.4% of the samples inside labelled windows and 32.9% overall. Anything
reading it as "no breathing" will be wrong most of the time it fires. `human` is the real target and is what the set becomes
once enough windows carry a label.

A directory holds **one** source. `--into` refuses to mix them: two different targets in one
directory would train a model against a moving definition. Override per run with
`--labels human`. Under `human`, windows nobody has labelled are not in the dataset at all -
"nobody looked at this" and "somebody looked and could not call it" are different facts, and the
labelling rule already writes the second one down as `unknown`.

## Splits

`make_splits.py` writes them into the dataset's own `windows.csv`, so the record of which window
went where survives the run that used it. Separate from the build, because re-splitting - a new
seed, different stratification, more folds - should not mean re-downloading a dataset.

```
split         train | test          held out once, by patient, never trained on in any fold
fold_0_split  train | val | test    test stays test; folds rotate only the validation set
fold_1_split  ...
```

**Test is fixed across folds**, so every fold reports against the same patients and one headline
number means something. `train.py` reads these columns and never recomputes them - the split that
trained a model has to be the one recorded beside the data.

Everything about how is in `data.split`:

| Key | What it does |
| --- | --- |
| `test_fraction` | share of windows held out, 0.2 |
| `folds`, `fold`, `seed` | how many validation folds, which one to train, and the seed |
| `group_columns` | the unit a split may not cut through - `[env, PatientID]` |
| `stratify_cols` | what is balanced across test and across folds - `[env]`; empty disables it |

**Grouped by patient, always.** Windows stride 5 s across a 20 s span, so neighbours share most of
their breaths: a random split puts the same breath on both sides and reports nothing. A split by
signal is not enough either - one patient's night is one breathing pattern.

The group key includes `env` because `PatientID` is a *display* name and the two instances have
unrelated identity spaces. Nothing stops a name appearing on both, and if one did, the name alone
would merge two different people into one group.

Stratification is a **greedy deficit-first assignment**, not `StratifiedGroupKFold`. Both were
measured: that splitter balances the number of *groups* per stratum, and our groups differ in size
by more than an order of magnitude (SL0066 has 190 windows, bs-010 has 81), so it produced folds
ranging from 0% to 58% `ds_prod` against an overall 32%, with sizes from 11.9% to 24.0%. Balancing
windows is what the metrics are averaged over.

## Training

```bash
poetry run python train.py                                       # fold 0 of the latest dataset
poetry run python train.py data.split.fold=3 training.max_epochs=100
poetry run python train.py data.dir=phases_algorithm_20260820T143456Z
DISABLE_CLEARML=true poetry run python train.py                  # local only
```

Hydra owns the config (`parameter/`), Lightning the loop, ClearML the record - project
**`inhale-exhale-phase`**, credentials from `~/clearml.conf`, scalars through the TensorBoard
logger, config through `connect_configuration`, and the best checkpoint uploaded as an artifact.
A server that does not answer downgrades the run to local logging rather than killing it.

## The model

1D U-Net with depthwise-separable convolutions in place of dense ones, shaped after **U-Time**.

### References

| Paper | Why it is the reference |
| --- | --- |
| Perslev, Jensen, Darkner, Jennum, Igel, **"U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging"**, NeurIPS 2019 — [arXiv:1910.11162](https://arxiv.org/abs/1910.11162) | The architecture this one is a scaled-down copy of. Fully convolutional encoder/decoder emitting one class per input sample, no recurrence, no fixed input length - exactly the shape of this problem. |
| Perslev, Darkner, Kempfner, Nikolic, Jennum, Igel, **"U-Sleep: resilient high-frequency sleep staging"**, npj Digital Medicine 4:72 (2021) — [doi:10.1038/s41746-021-00440-5](https://doi.org/10.1038/s41746-021-00440-5) · [open access](https://www.nature.com/articles/s41746-021-00440-5) | The same net at sub-epoch resolution, trained across many cohorts without per-dataset tuning. The evidence that this shape generalises across acquisition setups, which is the `ds_algo` / `ds_prod` question here. |
| Ravichandran, Murugesan, Balakarthikeyan, Ram, Preejith, Joseph, Sivaprakasam, **"RespNet: A deep learning model for extraction of respiration from photoplethysmogram"**, EMBC 2019 — [arXiv:1902.04236](https://arxiv.org/abs/1902.04236) | A 1D U-Net on the respiration waveform itself rather than on EEG. Nearest published work by signal type. |
| Ronneberger, Fischer, Brox, **"U-Net: Convolutional Networks for Biomedical Image Segmentation"**, MICCAI 2015 — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | The original, for the skip-connection idea the whole family rests on. |
| Bai, Kolter, Koltun, **"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"**, 2018 — [arXiv:1803.01271](https://arxiv.org/abs/1803.01271) | The dilated TCN, the alternative below if the device ever needs causal streaming. |

| Shape | Parameters | Receptive field |
| --- | --- | --- |
| `[16,24,32]` / 48 (**default**) | 18,188 | 353 samples, 35 s |
| `[16,24,32,48]` / 64 | 35,932 | 737 samples, 74 s |

The default sees a whole 20 s window plus margin; the deeper one is for running on a whole
signal. Both are under 150 KB in fp32 and quantize to int8 without ceremony. Any length in, same
length out - the input is padded to a multiple of `2 ** depth` inside `forward` and cropped back.

Alternative if the device needs **causal streaming**: a dilated TCN (Bai, Kolter, Koltun, above)
reaches the same receptive field with fewer parameters and runs off a ring buffer. It cannot see past a boundary, so a boundary is
only callable after it has gone by. Not built - the device already analyses in windows.

### The polarity flip is the augmentation that matters

`select_waveform` returns whichever of a range bin's real or imaginary part has the larger
standard deviation, and the bin is re-chosen per window, so the stored sign of a window is
arbitrary. Negating the trace and exchanging inhale with exhale therefore produces a sample that
is exactly as real as the one it came from - the same measurement written the other way up.

That has a consequence worth stating plainly: **the model cannot use absolute polarity to decide
direction**, because the training set contains both. Direction has to come from breath shape.
That is the correct constraint - the device does not know a window's polarity either - but
whether shape alone is sufficient is an **open empirical question** on this data, not a settled
one. The dataset is built from windows the labelling repo has already turned the right way up
using `inhale-exhale-detection`'s orientation rule, so the labels are consistent; whether the
network can reproduce that decision from the trace is what the first runs will show.

### What the model is not given

`in_channels` is 1. The obvious second channel is the **phase anchor** in
`inhale-exhale-detection/inhale_exhale/utils/displacement.py` - orientation read off the complex
range profile rather than off breath shape, gated at |corr| >= 0.3. It is a physical cue for
exactly the decision above and it is deliberately left out of the first runs, so that the
shape-alone question gets a clean answer before a second input muddies it.

### Decoding

Argmax will happily emit inhale, exhale, inhale over three samples. Two cheap passes run after
it, both configured in `training.decoding` and both on the device's budget:

- **Viterbi** over the four classes - impossible transitions cost infinity, staying is free.
- **Minimum duration** - a run shorter than its class's floor is absorbed into the longer
  neighbour.

The allowed transitions are a claim about *this label set*, not about physiology:
`inhale -> unknown -> exhale` is the normal path here and `inhale -> exhale` is not, because
production emits no phase for the turn. **Re-derive them if the labels ever come from people.**

## Evaluating

```bash
poetry run python evaluate.py --checkpoint outputs/<date>/<time>/checkpoints/best-epoch=NN.ckpt
poetry run python evaluate.py --checkpoint ... --human --env ds_prod
```

Without `--human` the reference is production's own answer, so the score measures imitation.
With it the reference is `BreathPhaseTimeRecord` and three rows are printed:

```
model   vs human      what the network gets right
teacher vs human      what production gets right on the same windows
model   vs teacher    how much of the gap is the network's own
```

A student cannot beat its teacher by imitating it. If the first row is not at least the second,
the network has not earned its place yet. The human set is small, so read it as a sanity check
rather than a verdict.

## Layout

The repo root holds only things you can run.

- `build_dataset.py` - sample signals, run the production algorithm, write the shards.
- `train.py` - one fold, Hydra + Lightning + ClearML.
- `evaluate.py` - a checkpoint against the algorithm's labels, or against human ones.
- `clearml_utils.py` - the same wiring as `cough`; no credentials in the repo.
- `build_dataset.py` - read the windows, label them, write a dataset directory.
- `make_splits.py` - assign `split` and `fold_i_split` into that directory's `windows.csv`.
- `parameter/` - the Hydra config tree, plus `sources/default.yaml` (databases, bucket, tables).
- `phase/` - `sources` (the two databases and S3), `production` (production's phase call and the
  pairing of its boundary arrays), `labelsources` (algorithm or human), `labels` (the vocabulary),
  `building` (the dataset directory), `dataset`, `splits`, `decode`, `metrics`.

`phase/production.py` is the file to watch. It is a faithful port of the phase call and span
pairing that `inhale-exhale-detection` validated, running against `holosissystem` directly - and
it was checked window for window against that implementation before the dependency was cut
(60/60 identical labels on real windows). If the upstream phase code changes, that file has to
follow it.
- `models/` - `unet1d`, `lightning_module`.
- `tests/` - 46 tests, no AWS and no built dataset; `tests/synthetic.py` also builds a fake set
  for a smoke run:

```bash
PYTHONPATH=. poetry run python tests/synthetic.py data_sets/synthetic
DISABLE_CLEARML=true poetry run python train.py \
    data.dir=data_sets/synthetic data.name=synthetic training.max_epochs=25 data.num_workers=0
```

## Open

- **The pool is whatever has been uploaded, and it grows under you.** Measured 2026-08-20:
  1,933 windows on `ds_algo` (293 signals, 19 `SL` patients) and 915 on `ds_prod` (113 signals,
  11 `bs-` patients) - 2,848 windows, 16.3 hours. `ds_prod` went from 45 windows to 915 during a
  single afternoon's work, so re-read the catalogue rather than quoting a number from here.
- **Human labels are the scarce thing**: 23 windows on `ds_algo`, 6 on `ds_prod`. Enough to sanity
  check a model, nowhere near enough to train or to fine-tune on.
- **The two instances differ in acquisition.** `ds_algo`'s eligible signals are ~51% 300 fps
  two-antenna and 49% 200 fps; `ds_prod` is 100% 200 fps. The preprocessing differs, so a model
  trained on one cohort is not obviously transferable to the other - worth measuring.
- Whether breath shape alone settles direction under the polarity flip. If it does not, the
  phase anchor becomes a second input channel.
- Whether the human set is large enough for a fine-tune rather than only an evaluation.
- Nothing here is exported for the device yet - no int8 quantization, no ONNX, no timing on
  target hardware.
