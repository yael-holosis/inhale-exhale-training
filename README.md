# inhale-exhale-training

Train a per-sample **inhale / exhale / stop / unknown** segmenter on radar respiration
waveforms, at 10 fps, any length in and one class per sample out. Small enough for the edge
device: 18,188 parameters at the default shape.

The first training set is **the production algorithm's own output**, over the windows that
[`respiration-phase-labeling`](https://github.com/Holosis-Health/respiration-phase-labeling) -
the labelling app - has already uploaded. Human labelling there is under way but still small: of
1,933 uploaded windows on `ds_algo`, **21 carry a human span**. So the network starts by learning
what the device already does, and moves onto human labels as they arrive.

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

**This repo defines none of them.** There is no `aws:` block in `parameter/`, no secret name and
no profile - every connection is opened by `respiration-phase-labeling/utils/db.py` and
`building.scan_reader`, from that repo's `parameters/parameters.yaml`. It is stated here only so
nobody goes looking for a second copy.

| | `ds_algo` (SL, sleep lab) | `ds_prod` (bs- / RM- pilots) |
| --- | --- | --- |
| Signals + patients | data-science MySQL, Secrets Manager `sm-data-science-01-db-password-mysql-6g2w8dxe`, db `edge_data` | **production's** MySQL, user `edge_data_user_ro`, db `edge_data` |
| Human labels | same server, db `edge_data_extras` | data-science replica, Secrets Manager `sm-data-science-01-db-password-edge-data-endpoint-0owgu5br` |
| Raw scans | profile `holosis-datascience-algo`, bucket `s3-data-science-01-holosis-health-system-sessions` | profile `holosis-prod-admin`, bucket `s3-prod-01-holosis-health-system-sessions` |

Two things follow from that table:

- **The scan buckets are in different accounts**, and the data-science profile gets a 403 on
  production's. `scan_reader` switches the process's `AWS_PROFILE` to reach them, which is why
  the two cohorts are two runs and not one.
- **Production is read-only by the server's rules**, not only by ours: `edge_data_user_ro` is
  granted SELECT and nothing else. Nothing in this repo writes anywhere near it.

Production's read-only database password is carried **in plaintext** in the labelling repo's
tracked `parameters/parameters.yaml`, and therefore in its git history. That is a deliberate
choice made there and documented there - most labellers have no Secrets Manager access in the
production account, and the app was unusable for them without it - but it is now a dependency of
this repo too, so it is named rather than left to be discovered. `RESPIRATION_PHASE_LABELING_PROD_RO_PASSWORD`
or `secrets/prod_edge_ro_password.txt` override it if it is ever moved out.

```bash
aws sso login --profile holosis-datascience-algo
aws sso login --profile holosis-prod-admin        # ds_prod only
```

## Building the dataset

```bash
poetry run python build_dataset.py --catalogue --env ds_algo   # what is uploaded, no writes
poetry run python build_dataset.py --env ds_algo               # the SL sleep-lab nights
poetry run python build_dataset.py --env ds_prod               # the pilots
```

**No raw scan is read.** The app repo already ran the production pipeline when it uploaded these
windows; this reads each window's samples from S3 and asks production's own
`calculate_inhale_exhale_time` what it calls on them. Two reads and a function call - about 4.5 s
per signal, and no disk pressure at all.

That also means the training windows are **the same objects a labeller sees**, byte for byte.
`RespirationWindowID` goes into the manifest, so a human label written later joins straight onto
the row the network trained on, with no re-derivation and no risk of the two describing different
samples.

**To grow the pool, upload more windows from the app repo.** Building a window is its job:

```bash
poetry run python upload_windows.py --env ds_algo --patients SL --per-patient 20 --commit
```

**Two cohorts, two runs, one directory.** `ds_algo` is the data-science instance, where the `SL`
sleep-lab nights are and where nearly every uploaded window lives; `ds_prod` is the pilots. Their
`RadarSignal` ID spaces are unrelated - signal 2120091 is a different recording on each - so a
shard is named `<env>_signal_<id>.npz` and the manifest is rebuilt from every shard present.

**Resumable**, and it costs nothing to be: a window blob is immutable once uploaded, because the
app repo refuses to overwrite one that labels already point at.

## Training

```bash
poetry run python train.py                                  # fold 0
poetry run python train.py data.fold=3 training.max_epochs=100
DISABLE_CLEARML=true poetry run python train.py             # local only
```

Hydra owns the config (`parameter/`), Lightning the loop, ClearML the record - project
**`inhale-exhale-phase`**, credentials from `~/clearml.conf`, scalars through the TensorBoard
logger, config through `connect_configuration`, and the best checkpoint uploaded as an artifact.
A server that does not answer downgrades the run to local logging rather than killing it.

**Splits are grouped by patient.** Windows stride 5 s across a 20 s span, so neighbours share
most of their breaths - a random split over windows puts the same breath on both sides and
reports a number that means nothing. A split by signal is not enough either: one patient's night
is one breathing pattern.

## The model

1D U-Net, shaped after **U-Time** (Perslev et al., *A Fully Convolutional Network for Time
Series Segmentation Applied to Sleep Staging*, NeurIPS 2019, [arXiv:1910.11162](https://arxiv.org/abs/1910.11162)),
with depthwise-separable convolutions in place of dense ones. Related: **U-Sleep** (Perslev et
al., npj Digital Medicine 4:72, 2021), the same architecture at sub-epoch resolution, and
**RespNet** (Ravichandran et al., EMBC 2019), a 1D U-Net on the respiration waveform itself.

| Shape | Parameters | Receptive field |
| --- | --- | --- |
| `[16,24,32]` / 48 (**default**) | 18,188 | 353 samples, 35 s |
| `[16,24,32,48]` / 64 | 35,932 | 737 samples, 74 s |

The default sees a whole 20 s window plus margin; the deeper one is for running on a whole
signal. Both are under 150 KB in fp32 and quantize to int8 without ceremony. Any length in, same
length out - the input is padded to a multiple of `2 ** depth` inside `forward` and cropped back.

Alternative if the device needs **causal streaming**: a dilated TCN (Bai, Kolter, Koltun,
[arXiv:1803.01271](https://arxiv.org/abs/1803.01271)) reaches the same receptive field with
fewer parameters and runs off a ring buffer. It cannot see past a boundary, so a boundary is
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
- `parameter/` - the Hydra config tree.
- `phase/` - `bridge` (reach the two repos), `labels` (the vocabulary), `building` (the
  dataset), `dataset`, `splits`, `decode`, `metrics`.

**Two directory names are load-bearing.** The labelling repo calls its package `utils` and its
config `parameters`, and both checkouts sit on `sys.path` during a build - so this repo uses
`phase/` and `parameter/`. Rename either back and every `from utils import ...` inside that repo
silently resolves to one of ours. `tests/test_bridge.py` fails if it happens.
- `models/` - `unet1d`, `lightning_module`.
- `tests/` - 46 tests, no AWS and no built dataset; `tests/synthetic.py` also builds a fake set
  for a smoke run:

```bash
PYTHONPATH=. poetry run python tests/synthetic.py data_sets/synthetic
DISABLE_CLEARML=true poetry run python train.py \
    data.dir=data_sets/synthetic data.name=synthetic training.max_epochs=25 data.num_workers=0
```

## Open

- **The pool is the uploaded windows, and it is small**: 1,933 on `ds_algo` over 293 signals and
  19 `SL` patients, 45 on `ds_prod` over 5 signals of one patient. Roughly 11 hours of trace in
  total. Growing it is an `upload_windows.py` run in the app repo, not a change here.
- **`ds_prod` is one patient.** Every pilot cohort - 44 patients, 65,671 eligible signals - is
  reachable, but only `bs-008` has had windows built. Until more are uploaded, the pilots are not
  represented in training at all.
- **The two instances differ in acquisition.** `ds_algo`'s eligible signals are ~51% 300 fps
  two-antenna and 49% 200 fps; `ds_prod` is 100% 200 fps. The preprocessing differs between them,
  so a model trained on one cohort is not obviously transferable to the other - worth measuring
  before assuming.
- Whether breath shape alone settles direction under the polarity flip. If it does not, the
  phase anchor becomes a second input channel.
- Whether the human set is large enough for a fine-tune rather than only an evaluation.
- Nothing here is exported for the device yet - no int8 quantization, no ONNX, no timing on
  target hardware.
