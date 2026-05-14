# Teaching a transformer to read fingerspelling

> How I trained a Connectionist Temporal Classification (CTC) model on
> 67,000 ASL fingerspelling clips and learned that the boring stuff at
> the bottom of the ML stack matters more than the model architecture
> sitting on top.

## The problem with the first model

The original sign recognition model in OpenHand was a small MLP that took
21 hand landmarks from MediaPipe and produced a single letter A–Z. It
was trained on the Kaggle "ASL Alphabet" image dataset — 87,000 still
photos of one signer making each letter.

Three things wrong with this:

1. **One signer.** Generalisation to anyone else's hands was poor.
2. **Still photos.** Letters that are *defined* by motion — J and Z —
   showed up in the training data as arbitrary snapshots from somewhere
   in the middle of the gesture. The model never saw the motion.
3. **One letter at a time.** No way to recognise actual phrases. The
   user had to hold each letter, the frontend debounced, characters
   appeared one by one. Slow, awkward.

The fix was always going to be a sequence model on a real fingerspelling
dataset, but I underestimated by how much.

## The dataset

[Google's ASL Fingerspelling Recognition competition on Kaggle](https://www.kaggle.com/competitions/asl-fingerspelling)
released a dataset of 67,208 video clips of people fingerspelling
phrases — addresses, URLs, names, phone numbers, anything that
real-world signing has to handle. Critically, **100+ different
signers**, captured across diverse lighting and angles.

The data comes pre-processed: for each video frame, MediaPipe Holistic
has already extracted 543 landmark coordinates (468 face + 33 pose + 21
each hand) into Parquet files. Total raw dataset: 160 GB.

The labels are the **entire phrase** per clip, not per-frame. That's the
crux of the whole project. To train on this you need a loss function
that handles unknown alignments between input frames and output
characters. That loss function is **CTC** — Connectionist Temporal
Classification.

## How CTC works (briefly)

CTC sits on top of a sequence model that emits one probability
distribution over the vocabulary per input frame. The vocabulary
includes a special "blank" token. CTC computes the probability of all
possible alignments that, after collapsing repeats and removing blanks,
yield the target phrase. The training objective is to maximise this
sum-over-alignments probability.

At inference, you take the argmax token per frame and apply the same
collapse rule. So if the model emits

```
b _ b _ _ a _ a _ n _ _ _ a _ n _ a _ _ _ _
```

(where `_` is blank), greedy decoding gives `b-a-n-a-n-a` → `banana`.

That's the elegance: the model doesn't need to know which frame
corresponds to which letter. As long as it puts the right characters in
the right *order*, with at least one blank between repeats, CTC will
sort the alignment out.

## My first attempt didn't work at all

Initial plan: 5.5M-param transformer (d_model=256, 6 encoder layers),
train for 30 epochs on the full dataset.

After 30 epochs and 50 minutes of GPU time, the val character error
rate was **0.94**. The model was outputting nothing — empty strings on
every input.

This is a well-known failure mode of CTC training called **blank
collapse**. Predicting "blank" everywhere is a low-loss local minimum
that the model finds early in training, and once it lands there the
gradients get small and it doesn't escape.

Three fixes broke us out of it:

1. **Linear LR warmup over 500–1500 steps.** Stops the optimizer from
   slamming hard into the blank-collapse basin during the first noisy
   gradient updates, when the loss surface is most pathological.
2. **KL-to-uniform label smoothing** (weight 0.1). I added a secondary
   loss term equal to the KL divergence between the model's predicted
   distribution and the uniform distribution. This penalises peaky
   "always blank" outputs and nudges the model toward real character
   emissions.
3. **Cosine LR decay** after warmup, ending near zero. Standard for
   transformers, but particularly important here so the model can
   fine-tune cleanly in the late stages without overshooting.

The smoke test confirmed these worked: the model's first-epoch outputs
went from `""` to `"e"` to `"eee"` to recognisable garbage to actual
words. The collapse was broken.

## My second attempt was working but agonisingly slow

After fixing blank collapse, I had a model that was learning — but
training was running at **57 minutes per epoch**. 20 epochs would have
taken 19 hours. Something was deeply wrong with the data pipeline.

I'd built the Dataset class to load Parquet shards on demand: pull the
sequence from disk, slice the columns we want, normalise, return.
DataLoader with `shuffle=True` was hammering this — each training step
needed sequences from 16-24 different shards, and each shard was 1.4 GB
of Parquet to decode.

Profiling showed >95% of step time was Parquet decoding. The "LRU cache
of size one" I'd written was thrashing constantly because shuffled
sampling guarantees consecutive `__getitem__` calls land in different
shards.

The fix was straightforward but boring: **pre-extract every sequence to
its own tiny `.npz` file once**. For each of the 67K sequences, save:

- `x`: the (T, 390) landmark tensor for that clip
- `missing`: an explicit (T, 130) boolean mask of which landmarks were
  absent on which frames
- `target`: the encoded character IDs

This took 10 minutes one-time and produced 11 GB on disk (~165 KB per
sequence). After the change, **training dropped from 57 min/epoch to
17 seconds/epoch** — about a 200× speedup. The bottleneck moved from
"decode Parquet" to "actual training compute."

This kind of "boring data pipeline" speedup ends up being the most
impactful change in the entire project. It's also the one nobody writes
about.

## A subtle correctness bug nobody would have caught from accuracy alone

While auditing the pipeline I found that `normalize_sequence` was
treating zero values as "missing" landmarks. MediaPipe normalises
landmarks to `[0, 1]`, so `x=0` means "left edge of frame" — a perfectly
valid real coordinate. The function was conflating real near-origin
landmarks with absent ones, and after wrist-centring (which subtracts
the wrist position, writing zeros into the wrist's slot), legitimate
landmarks were getting re-masked as missing.

The fix was to save an explicit boolean mask alongside the features in
each `.npz`, and rewrite `normalize_sequence` to use the mask directly
instead of inferring from zeros. Probably worth a couple of CER points,
though it's hard to isolate.

## What actually moved the needle

I ran three real training experiments after these fixes:

| Run | Architecture | Augmentation | Epochs | Best val CER |
|---|---|---|---|---:|
| 1 | 5.5M params (d=256, 6 layers) | mild | 40 | **0.274** |
| 2 | 27.5M params (d=512, 12 layers) | mild | 100 | **0.248** |
| 3 | 27.5M params (d=512, 12 layers) | strong (6 tricks) | 80 | **0.249** |

For reference, the Kaggle 1st place solution on this dataset reached
about 0.21 on the held-out test set using a similarly-sized model and
beam search decoding. We landed at 0.249 on a different signer-held-out
split.

The instructive part is what didn't help:

- **5.5M → 27.5M params**: CER dropped 0.274 → 0.248. Modest gain, and
  the model started overfitting (train loss 0.62, val loss 1.10).
- **Mild → strong augmentation**: closed the train/val gap (train loss
  went back up to 1.31, val loss down to 0.99 — much healthier) but
  val CER stayed at 0.249. Augmentation was preventing overfit, not
  unlocking new capacity. The data was the constraint.
- **Greedy → beam search at inference**: I implemented standard CTC
  prefix beam search expecting 0.03–0.05 CER reduction. Got 0.002. The
  model's softmax outputs were too peaked (typically >0.95 argmax
  probability per frame) for beam to help — all the beams collapsed to
  the same prefix.

What *did* help, ranked by my honest accounting:

1. Fixing the data pipeline (Parquet → .npz): 200× speed-up, no CER
   change. But this is what made every subsequent experiment possible.
2. Anti-blank-collapse tricks: turned CER 0.94 (unusable) into CER 0.27
   (working model).
3. Fixing the missing-data sentinel: probably 2–3 CER points, hard to
   isolate.
4. More data (16K subset → 67K full): CER 0.52 → 0.27.
5. Bigger model: 0.27 → 0.25.
6. Strong augmentation: closed the train/val gap, no CER change.
7. Beam search: negligible.

The lesson sounds banal but: **pipeline correctness and the loss-function
tricks dominated model architecture by a wide margin.**

## The final architecture

What I ended up with:

```
Input: (B, T, 390) landmark sequences, padded to T_max in batch
   │
   ▼
Conv1d stem (3×5 kernels, BN, GELU) — smooths jittery frames
   │
   ▼
Sinusoidal positional encoding
   │
   ▼
Transformer encoder × 12 layers (d_model=512, nhead=8, FFN=1024, GELU, pre-LN)
   │
   ▼
Linear head → 60 logits per frame (59 chars + 1 blank)
   │
   ▼
log_softmax → (T, B, 60)  ← CTCLoss input shape
```

27.5 million parameters. Trains in ~50 minutes on an RTX 4070 Super at
batch 48. Exports to a 116 MB ONNX file. Inference on CPU: ~16 ms for a
128-frame clip.

## Deploying it

The training repo (`openhand-model/`) produces an ONNX file. The
application repo (`openhand/`) consumes it via `onnxruntime` with no
PyTorch dependency.

The PyTorch → ONNX export had its own series of trouble:

- The legacy `torch.onnx.export` tracer baked the dummy time dimension
  into the multi-head-attention reshape ops, so exported models worked
  at T=64 but crashed at any other T. Switched to the new dynamo-based
  exporter (`torch.onnx.export(..., dynamo=True)`).
- Dynamo couldn't convert `BatchNorm` in eval mode. Worked around it
  by fusing BN into the preceding Conv1d weights before export.
- The dynamo exporter requires batch ≥ 2 to keep the batch axis
  dynamic. The runtime classifier pads with a fully-masked second item
  it ignores.
- The TensorFlow Lite XNNPACK delegate prints a checkmark emoji on
  successful export which crashed Windows' cp1252 console. Had to force
  UTF-8 stdout in the export script.

Inference in the backend builds the 127-landmark feature vector per
frame by running **three separate MediaPipe Tasks detectors** (Hand,
Pose, Face) in series. I initially tried the all-in-one
`HolisticLandmarker`, but on Windows + MediaPipe 0.10.21 it crashed the
entire Python process with a fatal C++ assertion when an internal
sub-task produced an empty packet. Three separate detectors are slower
per frame (~15-25 ms total) but stable.

## What it can do today

| Reference | Hypothesis |
|---|---|
| `www.horseillustrated.com/` | `www.horseilustrated.com/` |
| `tomeka salinas` | `tomeka salinas` |
| `centroeducaljarafe` | `centro educaljarafe` |
| `252-523-1055` | `252-523-1055` |
| `hindilinks4u.dirproxy.org` | `hindilinds4u.dirproxy.org` |

Three of those are exact, the others are one or two characters off.
These are on signers the model was never trained on.

Where it still struggles: very short phrases (`713-809-2808` →
`i.0/270`) and digit-heavy strings in general. The pattern is consistent
across runs and feels like a function of CTC's preference for longer
contexts.

## What I'd do next

In rough order of expected gain-per-effort:

1. **Language-model fusion during beam search**. The current model
   knows letter co-occurrence purely from its 64K training phrases.
   Adding even a simple character n-gram language model scored against
   beam candidates would catch realistic URL/address patterns. This is
   also where beam search would finally start to matter — beam +
   strong LM compounds much harder than greedy + LM.
2. **Mirror augmentation for left-handed signers**. The dataset is
   right-handed. Flipping x-axis + swapping left/right hand landmarks
   doubles training samples for free.
3. **Streaming inference**. Right now the user holds a button to
   record a clip, releases, gets a transcription. A rolling-window
   decode during live signing would feel much better but requires
   solving the "when has the user finished signing" problem.
4. **Use the supplemental data**. There's a `supplemental_metadata.csv`
   with another 53K sequences I haven't trained on.
5. **Ensemble two models** trained with different seeds. Usually 0.02-
   0.03 CER reduction for "free" — at the cost of inference compute.

## Some honest reflections

I went into this expecting model architecture to dominate — bigger
transformer, longer training, more capacity. The actual takeaway was
that **maybe 80% of the value came from boring infrastructure work**:
fixing data loading, fixing the missing-data sentinel, fixing the
blank-collapse problem. The architecture choice was almost a free
variable once those were right.

The other thing I underestimated was how *coupled* training-time and
inference-time decisions are. The 127-landmark selection has to match
exactly between the training preprocessor and the backend inference
service. The normalisation formula has to match exactly. The vocab has
to match. The blank token has to be at the same index. None of those
are interesting individually, but each one is a silent failure mode if
it drifts.

In the end, what I have is a 116 MB file that turns short videos of
people fingerspelling into text strings. It works on signers it's never
seen, in lighting it's never trained for, on a laptop CPU at ~16 ms
per clip. It's not state-of-the-art — it's about 4 CER points behind
the Kaggle winners — but it's a real working model produced in a few
days of part-time effort. That feels like a fair trade.
