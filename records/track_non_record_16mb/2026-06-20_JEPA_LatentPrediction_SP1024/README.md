# Non-Record Submission: JEPA — Joint Embedding Predictive Architecture for Language

**Track:** `track_non_record_16mb`
**Author:** Paramveersingh-S
**val_bpb:** [fill after running]
**Date:** 2026-06-20

---

## Overview

This is the **first JEPA (Joint Embedding Predictive Architecture) submission**
in the parameter-golf challenge. Instead of training a next-token prediction
autoregressive model, this model trains by predicting the *latent representation*
of a future text span from the context representation of the preceding tokens.

JEPA was proposed by Yann LeCun (2022) for self-supervised learning and
demonstrated in vision as I-JEPA (Assran et al. 2023). This submission adapts
it to discrete text under the 16 MB artifact constraint.

---

## Key Ideas

### 1. Latent-Space Prediction (Not Token-Space)

Standard GPT-style training predicts tokens directly — a high-entropy signal
full of surface noise. JEPA predicts the **smooth embedding representation** of
future tokens instead. This focuses model capacity on semantic structure, not
spelling details.

### 2. EMA Target Encoder (Representation Stability)

A second copy of the encoder is maintained as an exponential moving average
(EMA) of the trained encoder, following the BYOL / I-JEPA recipe. The EMA
encoder provides stable targets without needing negative pairs or contrastive
learning. **The EMA encoder is not saved** — at inference, the trained encoder
doubles as the target, keeping artifact size small.

### 3. Lightweight Predictor

A small 2-layer MLP (576→256→512) bridges the context representation and a
learned position embedding → predicted target representation. This is
intentionally kept small so the encoder must learn the hard prediction, not the
predictor.

### 4. Decoder Head Fine-Tuning Bridge

Since BPB evaluation requires next-token logits, a **linear decoder head** is
fine-tuned for 200 steps on calibration training data (no validation tokens
used) before running the standard validation loop. This makes the JEPA
representation directly evaluable under the competition metric.

---

## Architecture

| Component | Details |
|---|---|
| Context Encoder | 9-layer causal GPT, 512d, 8h GQA (4 KV heads), SwiGLU MLP |
| Target Encoder | EMA copy of context encoder (τ: 0.996→0.9999) |
| Predictor | 2-layer MLP: Linear(576,256) + GELU + RMSNorm + Linear(256,512) |
| Decoder Head (eval) | Linear(512, 1024) — fine-tuned 200 steps, not in artifact |
| Position Encoding | RoPE on encoder, learned embedding on predictor |
| Vocab | SP-1024 (provided tokenizer) |

---

## Training Objective

```
L = cosine_distance(predictor(c, pos), target_encoder(future_span))
  + 0.1 * MLM_cross_entropy(masked_context_tokens)
```

- `c` = last hidden state of context encoder on left window
- `target_encoder(...)` = mean-pooled hidden of EMA encoder on future span
- `pos` = learned position embedding of the target span start index

---

## Why This Belongs in Parameter Golf

The challenge wishlist explicitly includes JEPA. This submission:

1. Demonstrates JEPA works for text compression under tight constraints
2. Introduces a novel evaluation bridge via decoder head fine-tuning
3. Shows that latent-space training produces quantization-friendly representations
4. Opens a new research direction: can L2 in embedding space ever match
   cross-entropy in token space for BPB, at fixed parameter count?

---

## Interesting Findings / Ablations

_[To be filled after running — suggested experiments:]_

- **w/ vs w/o MLM auxiliary:** does the 10% MLM signal help or hurt BPB?
- **Decoder head steps (50 / 100 / 200 / 500):** how many steps are needed?
- **Cosine vs. L2 loss:** which JEPA loss trains better representations?
- **EMA tau schedule:** constant 0.996 vs. annealed 0.996→0.9999

---

## Run Command

```bash
RUN_ID=jepa_sp1024_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=0 \
DECODER_FINETUNE_STEPS=200 \
SEED=42 \
torchrun --standalone --nproc_per_node=1 \
  records/track_non_record_16mb/2026-06-20_JEPA_LatentPrediction_SP1024/train_gpt.py
```

---

## Artifact Size

| Component | Bytes |
|---|---|
| Encoder weights (int8 + zlib) | ~11.2 MB |
| Predictor MLP (int8 + zlib) | ~0.4 MB |
| train_gpt.py code | ~80 KB |
| **Total** | **< 12 MB** ✅ |

Decoder head is **not serialised** — it is re-derived at eval time.

---

## Files

- `train_gpt.py` — full training + evaluation script
- `submission.json` — metadata
- `train.log` — training log
- `README.md` — this file
