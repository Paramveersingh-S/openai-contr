# Non-Record Submission: JEPA — Joint Embedding Predictive Architecture for Language

**Track:** `track_non_record_16mb`
**Author:** [your-github-username]
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

