# inhale-exhale-training

Train a per-sample **inhale / exhale / stop / unknown** segmenter on radar respiration
waveforms, at 10 fps, any length in and one class per sample out. Small enough for the edge
device: 18,188 parameters at the default shape.

Two label sources. **`algorithm`** is the production detector's own output over every stored
window - imitation, and what there is most of. **`human`** is the spans a person drew: 603 windows
over 35 patients as of 2026-08-24, and the only thing that measures correctness. The human set is
now large enough to train on directly, which is what the current runs do.

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
| Signals + patients | same server, db `edge_data` | **production's** MySQL, SM `sm-prod-01-clinical-dashboard-edge-ro` under `holosis-prod-admin`, read only |
| Window samples | \multicolumn - one bucket for both: `s3-data-science-01-holosis-health-system-sessions` | |

- **Production is read-only by the server's rules**, not only by ours: its account is granted
  SELECT and nothing else. And `phase.sources.frame` refuses anything that is not a read, so a
  writer added later fails in a test rather than on production.
- **Secrets Manager is the only credential source.** Host, user and password all come out of the
  secret named in the config - there is no password file and no environment variable to set.
  Production's secret is read under `holosis-prod-admin`, so reading that side needs Secrets
  Manager access in the production account.
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
| How much | every window | 603 windows, 35 patients (2026-08-24) |
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

### Taking the labelling at less than its word

`data.labels.corrections` rewrites the target before it reaches a shard, so it is what the network
is trained on, what it is scored against, and what every figure draws. Both are off by default.

| | What it does | Why |
| --- | --- | --- |
| `blank_edge_spans` | a labelled span that runs to the window boundary becomes `unknown`, whole | the boundary cut that breath, and whether a labeller marks the fragment it leaves is inconsistent |
| `all_unknown_above` | a window more than that fraction `unknown` becomes entirely `unknown` | a window that is mostly uncallable should not claim the phases it does have |

The fraction is measured **after** the edge spans are blanked - blanking them adds `unknown`, so
it can be what pushes a window over.

**Touching the boundary is the whole test.** Where a window opens or closes with `unknown` the
labeller looked at that stretch and declined it, so nothing there was truncated and the phase
beside it is left as drawn. 78% of windows start with a phase at sample 0 and 77% end with one;
65 labelled windows are untouched by this entirely.

What each setting costs on the 1,081-window human set:

| `blank_edge_spans` | `all_unknown_above` | `unknown` | windows entirely unknown |
| --- | --- | --- | --- |
| off | off | 25.9% | 155 |
| off | 0.5 | 27.3% | 193 |
| on | off | 32.1% | 155 |
| on | 0.5 | 33.9% | 202 |

Blanking takes 8.4% of the labelled samples.

**A score measured under one setting cannot be compared with one measured under another.** Both
corrections move samples into `unknown`, which is the easy class, so the headline rises without
the model improving.

Each shard keeps the labelling **as drawn** next to the corrected target, so a new setting is
derived from an existing dataset rather than rebuilt:

```bash
poetry run python build_dataset.py --recorrect latest \
    data.labels.corrections.blank_edge_spans=true \
    data.labels.corrections.all_unknown_above=0.5
```

That writes a new dataset directory in about two seconds and carries the split columns across -
the windows and the patients did not move, only the target. A rebuild from the database costs a
round-trip per window, about twelve minutes on the current set. Any trailing `key=value` argument
overrides the config tree the same way.

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

### Where the decoding comes from

**U-Time itself does not decode** - it takes per-sample argmax - and no published work pairs a
1-D U-Net with Viterbi for *respiration phase* specifically. The pattern itself is established
though: the first two papers below are a 1-D CNN and an HMM decoded by Viterbi, doing exactly
this job on a neighbouring problem.

| Reference | What it supports |
| --- | --- |
| Yang, Wu, Wang, Bao, Wang, **"A single-channel EEG based automatic sleep stage classification method leveraging deep one-dimensional convolutional neural network and hidden Markov model"**, Biomedical Signal Processing and Control 68:102581, 2021 — [doi:10.1016/j.bspc.2021.102581](https://doi.org/10.1016/j.bspc.2021.102581) | **The closest published precedent.** A 1-D CNN classifies each epoch, then "HMM works as a post-processing step to correct the sleep stage sequence output from 1D-CNN, thereby correcting unreasonable sleep stage transitions" - the same architecture and the same division of labour as here. Reported as the first pairing of a 1-D CNN with an HMM for sleep staging. |
| Pan, Kuo, Zeng, Liang, **"A transition-constrained discrete hidden Markov model for automatic sleep staging"**, BioMedical Engineering OnLine 11:52, 2012 — [doi:10.1186/1475-925X-11-52](https://doi.org/10.1186/1475-925X-11-52) | **The `allowed` table, published.** "To rule out impossible sleep stage transitions, the a<sub>ij</sub> corresponding to the impossible transition was set to zero according to the sleep stage transition diagram", decoded with Viterbi. That is precisely what `decoding.allowed` does - and, like ours, their transitions are read off the label set rather than learned. |
| Viterbi, **"Error bounds for convolutional codes and an asymptotically optimum decoding algorithm"**, IEEE Trans. Information Theory 13(2), 1967 — [doi:10.1109/TIT.1967.1054010](https://doi.org/10.1109/TIT.1967.1054010) | The algorithm itself. Exact MAP inference over a first-order chain in O(T·K²) - the globally best label sequence, not a locally smoothed argmax. |
| Rabiner, **"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"**, Proc. IEEE 77(2), 1989 — [doi:10.1109/5.18626](https://doi.org/10.1109/5.18626) | Why decoding a sequence beats per-frame argmax, and what a transition matrix is doing. |
| Hinton, Deng, Yu, Dahl, Mohamed, Jaitly, Senior, Vanhoucke, Nguyen, Sainath, Kingsbury, **"Deep Neural Networks for Acoustic Modeling in Speech Recognition"**, IEEE Signal Processing Magazine 29(6), 2012 — [doi:10.1109/MSP.2012.2205597](https://doi.org/10.1109/MSP.2012.2205597) | The hybrid shape used here: a neural net supplies emissions, an HMM supplies transitions, decoding is separate from training. |
| Lafferty, McCallum, Pereira, **"Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"**, ICML 2001 | The structured-prediction framing, and the principled upgrade - see the caveat below. |
| Huang, Xu, Yu, **"Bidirectional LSTM-CRF Models for Sequence Tagging"**, 2015 — [arXiv:1508.01991](https://arxiv.org/abs/1508.01991) | The same architecture with **learned** transitions and a structured loss. |

**The honest caveat.** Our transitions are hand-written and the loss is not structured, so the
model is trained to be right per sample and then decoded under constraints it never saw. A wrong
table cannot be corrected by the data - it can only be caught by eye, which is exactly how the
`inhale -> exhale` omission was found. A CRF layer (learned transitions, CRF loss) would remove
both problems and is the principled version of what is here.

| Shape | Parameters | Receptive field |
| --- | --- | --- |
| `[16,24,32]` / 48 (**default**) | 18,188 | 353 samples, 35 s |
| `[16,24,32,48]` / 64 | 35,932 | 737 samples, 74 s |

The default sees a whole 20 s window plus margin; the deeper one is for running on a whole
signal. Both are under 150 KB in fp32 and quantize to int8 without ceremony.

**Any length in, same length out**, and that holds in training as well as at inference. The input
is padded to a multiple of `2 ** depth` inside `forward` and cropped back, so a window keeps its
own length; `phase.dataset.collate` pads each batch to its own longest member and masks the
padding, which the loss and the metrics already drop. Nothing is cropped to fit a tensor.

### Batching, and what it means at inference

The weights are convolution kernels and per-channel norms - `(16, 1, 9)`, `(16, 16, 1)`, `(16,)`.
Not one of them has a length dimension, and there is no `Linear` or flatten anywhere: the head is
a 1x1 convolution. So the same 18,188 parameters apply to a 137-sample window and a 12,000-sample
signal, and the gradient from a batch of 200-sample windows is a gradient for the same weights a
600-sample window will use. Length never enters the parameter shapes; only the batch tensor
wanted a common one.

**Training batches are bucketed by length** (`LengthBucketSampler`), so a batch is drawn from one
length group and needs no padding. That is a correctness fix, not an optimisation: 94% of windows
are exactly 200 samples, which reads as "a random batch is almost always uniform" and is not -
with 64 to a batch the chance all of them are 200 is 1.6%, so **98% of random batches carry
padding, averaging 40% of the tensor and 65% at worst**. That padding is masked out of the loss
but not hidden from `BatchNorm`, which normalises over batch and length together and carries its
statistics into inference, where no padding exists.

`drop_last` is off and must stay off. Bucketed, the short batch is not a remainder - it is the
whole of a rare length. Turning it on discards every window of 300 samples and longer: 82 of
them, the slowest and most irregular breathing in the set.

**At inference, run one window at a time - or bucket - but never mix lengths in one batch.**
`evaluate.predict` does the former. In eval mode a batch-mate cannot change a window's answer,
because BatchNorm uses its running statistics and nothing else crosses the batch dimension - but
that holds only while the batch is *unpadded*. Padding a 200-sample window up to 600 puts zeros
inside the receptive field of its own tail, and the last samples then get a different answer:

```
per-position max |difference|, a 200-sample window padded to 600
  samples   0- 24   4.8e-07     far from the pad: identical
  samples 100-124   7.8e-02
  samples 175-199   1.6e+00     at the pad: a different answer
```

Beyond half a receptive field (176 samples) from the join the two are bit-identical. On the
device this is not a constraint at all - a signal is analysed on its own, so there is no batch.

`data.crop_samples` can pin a fixed length under memory pressure, but it is not the default and
it is not free. 6.3% of windows are longer than 200 samples and they are the **hard** ones - the
pipeline grows a window by 5 s and retries exactly when it cannot find three breaths in it, so a
200-sample crop discards the slow and irregular breathing first. Those windows score a median
0.68 against 0.84 for the plain 200-sample ones, so they are the part of the set worth keeping
whole.

Alternative if the device needs **causal streaming**: a dilated TCN (Bai, Kolter, Koltun, above)
reaches the same receptive field with fewer parameters and runs off a ring buffer. It cannot see past a boundary, so a boundary is
only callable after it has gone by. Not built - the device already analyses in windows.

### Polarity is a convention, not noise

`select_waveform` returns whichever of a range bin's real or imaginary part has the larger
standard deviation, and the bin is re-chosen per window, so the stored sign of a window is
arbitrary. The build therefore orients each window to the polarity its reviewer labelled
against - `data.labels.orient_by_reviewer_flip`, applied where `ReviewerFlipped` is set.

**There is no polarity-flip augmentation.** Negating half the windows at random would throw that
convention away, and it is the convention the labels are written in. `amplitude_range` stays
positive for the same reason.

So the model *may* use polarity to decide direction, as long as inference orients its input the
same way the dataset was oriented. That is a constraint on deployment rather than on training,
and it is the trade made when the flip came out.

### What the model is not given

`in_channels` is 1. The obvious second channel is the **phase anchor** in
`inhale-exhale-detection/inhale_exhale/utils/displacement.py` - orientation read off the complex
range profile rather than off breath shape, gated at |corr| >= 0.3. It is a physical cue for
exactly the decision above and it is deliberately left out of the first runs, so that the
shape-alone question gets a clean answer before a second input muddies it.

### Decoding - post-processing, and measured as such

**The decoder is not part of the network.** The loss is weighted cross-entropy plus soft Dice on
raw logits; no gradient reaches the decoder, and training never sees it. It runs at inference and
when scoring, over logits the network has already produced.

Argmax will happily emit inhale, exhale, inhale over three samples. Nothing in a lung does that,
and one stray sample splits a breath into three in every event-level metric. Two cheap passes run
after the network, both configured in `training.decoding` and both on the device's budget:

- **Viterbi** over the four classes - impossible transitions cost infinity, staying is free, an
  allowed change costs `switch_penalty`. **On.**
- **Minimum duration** - a run shorter than its class's floor is absorbed into the longer
  neighbour. **Off** - see below.

#### What each pass is worth

The two passes are independent and have to be measured apart. Scoring one run's saved logits four
ways - 228 test windows, 15 held-out patients, 5-fold ensemble, run `14-36-33`:

| | macro F1 | accuracy | spans / window | inhale MAE | exhale MAE | stop MAE |
| --- | --- | --- | --- | --- | --- | --- |
| argmax alone | 0.8487 | 0.8595 | 15.9 | 0.108 | 0.270 | 0.249 |
| minimum duration only | 0.8484 | 0.8592 | 15.1 | 0.096 | 0.242 | 0.247 |
| **Viterbi only** | **0.8500** | **0.8611** | 13.9 | **0.091** | **0.203** | 0.247 |
| Viterbi + minimum duration | 0.8499 | 0.8610 | 13.8 | 0.091 | 0.203 | 0.250 |

**Viterbi does the work; the floors add nothing on top of it.** Alone the floors recover about half
the duration gain, but Viterbi's transition cost already suppresses the flicker they exist to
catch, so applied after it they move macro F1 by -0.0001 and make the stop duration slightly
worse. They also carry a cost the transition table does not: a floor **deletes** a real short
phase rather than smoothing it, and `stop: 2` forbids reporting any pause under 0.2 s against a
labelled stop mean of 0.82 s.

So `enforce_min` is **off**. The pass is kept rather than deleted - it is the right tool if the
transition table is ever turned off, and it is what a device without the table would need.

Reported the same way in every run: `report_folds.py` scores the same logits twice, once on the
network's argmax and once decoded, and writes `raw/`, `viterbi/` and a `raw_vs_viterbi`
comparison, all of which go to ClearML. With `enforce_min: false`, `viterbi/` is now the
transition model alone, so the comparison isolates one thing.

Turning the transition model off too is `training.decoding.viterbi: false`.

#### The transition table is per label source

The allowed transitions are a claim about *the label set*, not about physiology, so there is one
table per source in `training.decoding.allowed`:

- **`algorithm`** - production emits no phase for the turn, so `inhale -> unknown -> exhale` is
  the normal path and `inhale -> exhale` does not occur.
- **`human`** - people draw the phases touching. Measured over 7,713 span transitions in the
  human set: `inhale -> exhale` 33.3%, `stop -> inhale` 30.0%, `exhale -> stop` 29.9%,
  `exhale -> inhale` 2.2%, everything touching `unknown` under 2%.

Using the algorithm table on human labels forbids the commonest transition there is, and Viterbi
bridges it with a one-sample `unknown` on **every breath** - an artefact of the table, not of the
model. That is what a wrong table looks like, and it is why the source picks the table.

## Looking at the test set

```bash
poetry run python evaluate.py --checkpoint <ckpt> --plot 6
poetry run python evaluate.py --checkpoint <ckpt> --plot 4 --plot-pick worst
```

`train.py` draws one automatically at the end of every run and uploads it to ClearML as a debug
sample, because a macro F1 does not say whether the breaths came out as breaths and nobody goes
back to draw a page for a run that looked fine at the time.

**The model's answer is the shading behind the trace; the reference is the ribbon underneath.**
That is the labelling app's own layout, and reading it the same way in both places is worth more
than any refinement here - somebody who has spent a morning labelling windows should not have to
learn a second visual language to check what the model did with them. Colours are
`plot.phase_colors` in the config, defaulting to the app's own values.

`--plot-pick spread` (the default) takes the worst, the median and the best, so a page shows the
range rather than a flattering sample of it. On a set this size a random draw is mostly median
windows and hides both tails. Also `worst`, `best`, `random`.

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
- Whether breath shape settles direction without leaning on polarity. It no longer has to,
  now that windows are oriented - but a model that leans on polarity needs inference to
  orient its input identically, so it is worth knowing which it is doing.
- Whether the human set is large enough for a fine-tune rather than only an evaluation.
- Nothing here is exported for the device yet - no int8 quantization, no ONNX, no timing on
  target hardware.
