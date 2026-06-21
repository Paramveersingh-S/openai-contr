# Parameter Golf: JEPA Latent Prediction Submission

This repository contains a specialized submission for the [openai/parameter-golf](https://github.com/openai/parameter-golf) challenge, targeting the `track_non_record_16mb` track.

Unlike traditional autoregressive next-token prediction models, this implementation introduces a **Joint Embedding Predictive Architecture (JEPA)** adapted for discrete text. 

## 🧠 Technical Overview

### Latent-Space Prediction
Standard GPT-style training predicts tokens directly, which is a high-entropy signal full of surface noise. JEPA, originally proposed by Yann LeCun (2022), instead predicts the **smooth embedding representation** of future tokens. This focuses model capacity on semantic structure, not spelling details.

### Architecture Details
- **Context Encoder**: A 9-layer causal GPT with 512 embedding dimension and 8-head Grouped Query Attention (4 KV heads). It processes the context window.
- **Target Encoder**: An Exponential Moving Average (EMA) copy of the context encoder. Its weights are updated smoothly ($\tau: 0.996 \rightarrow 0.9999$). This provides stable targets without needing contrastive negative pairs. *Crucially, this is not saved in the artifact, saving ~11MB.*
- **Predictor**: A lightweight 2-layer MLP (`576 -> 256 -> 512`) that bridges the context representation and a learned position embedding to predict the target representation.
- **Decoder Head**: Since BPB evaluation requires next-token logits, a linear decoder head is fine-tuned for 200 steps on calibration data (no validation tokens used) before running standard validation.

### Training Objective
The model minimizes the cosine distance between the predicted latent representation and the target encoder's representation of a future span, plus a small Masked Language Modeling (MLM) auxiliary loss:
$$ L = 1 - \text{cosine\_similarity}(Predictor(c, pos), TargetEncoder(future\_span)) + 0.1 \times MLM\_Loss $$

## 🚀 Running on Google Colab

This repository is optimized to be cloned and run directly in Google Colab.

### Step 1: Open Colab and Clone
Create a new notebook in Google Colab. Go to `Runtime` -> `Change runtime type` and select either **T4 GPU** (Standard) or **TPU v2** (Faster).

Run the following in a cell to clone this repository:
```bash
!git clone https://github.com/Paramveersingh-S/openai-contr.git
%cd openai-contr
```

### Step 2: Install Dependencies
Ensure you have the required tokenizers installed:
```bash
!pip install sentencepiece
```

### Step 3: Download the Dataset
The script relies on the preprocessed FineWeb dataset provided by the parameter-golf challenge:
```bash
!python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

### Step 4: Train the Model

#### Option A: GPU (Standard)
Launch the training run using `torchrun`.

```bash
!RUN_ID=jepa_sp1024_run1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=0 DECODER_FINETUNE_STEPS=200 \
SEED=42 \
torchrun --standalone --nproc_per_node=1 \
  records/track_non_record_16mb/2026-06-20_JEPA_LatentPrediction_SP1024/train_gpt.py \
  2>&1 | tee records/track_non_record_16mb/2026-06-20_JEPA_LatentPrediction_SP1024/train.log
```

#### Option B: TPU (Faster, ~8x Speedup)
If you selected the **TPU v2** runtime, install PyTorch XLA first:
```bash
!pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

Then run the specialized TPU script using standard `python`. XLA will automatically distribute the work across all 8 TPU cores:
```bash
!XLA_USE_BF16=1 \
RUN_ID=jepa_sp1024_run1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=0 DECODER_FINETUNE_STEPS=200 \
SEED=42 \
python records/track_non_record_16mb/2026-06-20_JEPA_LatentPrediction_SP1024/train_gpt_tpu.py \
  2>&1 | tee records/track_non_record_16mb/2026-06-20_JEPA_LatentPrediction_SP1024/train_tpu.log
```

### Step 5: Extract Metrics
Once the training completes, open `records/track_non_record_16mb/2026-06-20_JEPA_LatentPrediction_SP1024/train.log`.
Find the final `val_bpb` and the `total: XX,XXX,XXX bytes` size metrics to update the `submission.json` file for your PR.

## 📁 Repository Structure
The core contribution is located in:
`records/track_non_record_16mb/2026-06-20_JEPA_LatentPrediction_SP1024/`
- `train_gpt.py`: Complete self-contained model, training, and evaluation script.
- `submission.json`: Metadata for the PR.
- `README.md`: Specific technical breakdown of the track submission.
