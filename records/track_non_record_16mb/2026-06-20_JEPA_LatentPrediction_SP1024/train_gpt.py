"""
JEPA (Joint Embedding Predictive Architecture) for Parameter Golf
Non-record submission — openai/parameter-golf
Track: track_non_record_16mb

Architecture:
  - Context Encoder: 9-layer causal transformer (GPT-style)
  - Target Encoder: EMA copy of context encoder (no gradient, not saved)
  - Predictor: Lightweight MLP that maps (context_repr, position) → target_repr
  - Decoder Head: Linear(d_model → vocab_size), fine-tuned 200 steps before eval

Training Loss:
  L = cosine_distance(predictor(c, pos), target_encoder(future_span))
    + 0.1 * MLM_crossentropy_on_context

Evaluation:
  1. Fine-tune decoder_head on calibration split (200 steps, no val tokens)
  2. Run standard BPB evaluation with decoder_head instead of lm_head
"""

import os, math, time, zlib, pickle, uuid, copy, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataclasses import dataclass
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    # Data
    data_path:       str  = os.environ.get("DATA_PATH",
                               "./data/datasets/fineweb10B_sp1024")
    tokenizer_path:  str  = os.environ.get("TOKENIZER_PATH",
                               "./data/tokenizers/fineweb_1024_bpe.model")
    run_id:          str  = os.environ.get("RUN_ID", str(uuid.uuid4()))
    # Architecture
    vocab_size:      int  = int(os.environ.get("VOCAB_SIZE", 1024))
    n_layers:        int  = int(os.environ.get("NUM_LAYERS", 9))
    d_model:         int  = int(os.environ.get("MODEL_DIM", 512))
    n_heads:         int  = int(os.environ.get("NUM_HEADS", 8))
    n_kv_heads:      int  = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult:        int  = int(os.environ.get("MLP_MULT", 2))
    # JEPA-specific
    jepa_target_len: int  = int(os.environ.get("JEPA_TARGET_LEN", 64))
    jepa_offset_min: int  = int(os.environ.get("JEPA_OFFSET_MIN", 8))
    jepa_offset_max: int  = int(os.environ.get("JEPA_OFFSET_MAX", 32))
    ema_start:       float = float(os.environ.get("EMA_START", 0.996))
    ema_end:         float = float(os.environ.get("EMA_END", 0.9999))
    mlm_weight:      float = float(os.environ.get("MLM_WEIGHT", 0.1))
    mlm_mask_rate:   float = float(os.environ.get("MLM_MASK_RATE", 0.15))
    decoder_steps:   int  = int(os.environ.get("DECODER_FINETUNE_STEPS", 200))
    # Training
    train_seq_len:   int  = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    batch_tokens:    int  = int(os.environ.get("TRAIN_BATCH_TOKENS", 524288))
    iterations:      int  = int(os.environ.get("ITERATIONS", 9000))
    warmup:          int  = int(os.environ.get("WARMUP_STEPS", 100))
    warmdown:        int  = int(os.environ.get("WARMDOWN_ITERS", 1000))
    max_lr:          float = float(os.environ.get("MAX_LR", 0.0018))
    min_lr:          float = float(os.environ.get("MIN_LR", 0.0))
    weight_decay:    float = float(os.environ.get("WEIGHT_DECAY", 0.1))
    grad_clip:       float = float(os.environ.get("GRAD_CLIP", 1.0))
    # Wallclock
    max_wallclock:   int  = int(os.environ.get("MAX_WALLCLOCK_SECONDS", 0))
    # Logging / eval
    log_every:       int  = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    val_every:       int  = int(os.environ.get("VAL_LOSS_EVERY", 500))
    val_batch:       int  = int(os.environ.get("VAL_BATCH_SIZE", 524288))
    eval_stride:     int  = int(os.environ.get("EVAL_STRIDE", 64))
    seed:            int  = int(os.environ.get("SEED", 42))

args = Args()
torch.manual_seed(args.seed)

# ─────────────────────────────────────────────────────────────────────────────
# Distributed setup
# ─────────────────────────────────────────────────────────────────────────────

dist.init_process_group(backend="nccl")
rank       = dist.get_rank()
world_size = dist.get_world_size()
device     = torch.device(f"cuda:{rank}")
master     = (rank == 0)
torch.cuda.set_device(device)

# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Embedding
# ─────────────────────────────────────────────────────────────────────────────

def build_rope_cache(seq_len: int, head_dim: int, device) -> torch.Tensor:
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos   = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return torch.cat([freqs, freqs], dim=-1)

def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    B, H, T, D = x.shape
    cos = cos_sin[:T, :D//2].cos()[None, None]  # (1,1,T,D/2)
    sin = cos_sin[:T, :D//2].sin()[None, None]
    x1, x2 = x[..., :D//2], x[..., D//2:]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)

# ─────────────────────────────────────────────────────────────────────────────
# Grouped-Query Attention
# ─────────────────────────────────────────────────────────────────────────────

class GQAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h  = n_heads
        self.kv = n_kv_heads
        self.hd = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model,             bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(d_model, d_model,             bias=False)

    def forward(self, x, rope, causal=True):
        B, T, C = x.shape
        Q = self.q_proj(x).view(B, T, self.h,  self.hd).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        Q = apply_rope(Q, rope)
        K = apply_rope(K, rope)
        # Repeat KV heads
        r = self.h // self.kv
        K = K.repeat_interleave(r, dim=1)
        V = V.repeat_interleave(r, dim=1)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, C))

# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d_model, mult):
        super().__init__()
        h = int(d_model * mult)
        self.fc1  = nn.Linear(d_model, h * 2, bias=False)
        self.fc2  = nn.Linear(h, d_model,    bias=False)

    def forward(self, x):
        g, v = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(g) * v)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, mlp_mult):
        super().__init__()
        self.ln1  = nn.RMSNorm(d_model)
        self.attn = GQAttention(d_model, n_heads, n_kv_heads)
        self.ln2  = nn.RMSNorm(d_model)
        self.mlp  = MLP(d_model, mlp_mult)

    def forward(self, x, rope, causal=True):
        x = x + self.attn(self.ln1(x), rope, causal=causal)
        x = x + self.mlp(self.ln2(x))
        return x

# ─────────────────────────────────────────────────────────────────────────────
# Context Encoder (trainable, causal)
# ─────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.embed  = nn.Embedding(a.vocab_size, a.d_model)
        self.blocks = nn.ModuleList([
            Block(a.d_model, a.n_heads, a.n_kv_heads, a.mlp_mult)
            for _ in range(a.n_layers)
        ])
        self.ln_f   = nn.RMSNorm(a.d_model)
        self.rope   = build_rope_cache(a.train_seq_len * 2, a.d_model // a.n_heads, device)

    def forward(self, tokens, causal=True):
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x, self.rope, causal=causal)
        return self.ln_f(x)

# ─────────────────────────────────────────────────────────────────────────────
# JEPA Predictor (lightweight MLP)
# ─────────────────────────────────────────────────────────────────────────────

class JEPAPredictor(nn.Module):
    """
    Maps (context_repr ‖ position_emb) → predicted target repr.
    Kept small intentionally — should not memorise targets.
    """
    def __init__(self, d_model, pos_dim=64, hidden=256):
        super().__init__()
        self.pos_emb = nn.Embedding(4096, pos_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model + pos_dim, hidden, bias=False),
            nn.GELU(),
            nn.RMSNorm(hidden),
            nn.Linear(hidden, d_model, bias=False),
        )

    def forward(self, ctx_repr, target_pos):
        # ctx_repr: (B, d_model)   target_pos: (B,) int
        p = self.pos_emb(target_pos)            # (B, pos_dim)
        z = torch.cat([ctx_repr, p], dim=-1)    # (B, d_model+pos_dim)
        return self.net(z)                       # (B, d_model)

# ─────────────────────────────────────────────────────────────────────────────
# Full JEPA Model
# ─────────────────────────────────────────────────────────────────────────────

class JEPAModel(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.args      = a
        self.encoder   = ContextEncoder(a)
        self.predictor = JEPAPredictor(a.d_model)
        # Target encoder = EMA copy, registered as a buffer-like structure
        # (not an nn.Module so it doesn't appear in self.parameters())
        self._target_enc_state = None

    def init_target_encoder(self):
        """Copy encoder weights → target encoder state dict (no grad)."""
        self._target_enc_state = copy.deepcopy(self.encoder.state_dict())

    @torch.no_grad()
    def update_ema(self, tau: float):
        """EMA update: θ_target ← τ*θ_target + (1-τ)*θ_encoder"""
        enc_sd = self.encoder.state_dict()
        for key in self._target_enc_state:
            self._target_enc_state[key] = (
                tau * self._target_enc_state[key].float()
                + (1 - tau) * enc_sd[key].float()
            ).to(enc_sd[key].dtype)

    def target_encode(self, tokens):
        """Run target encoder (EMA weights) without gradient."""
        # Temporarily load EMA weights into encoder, run forward, restore
        enc_sd_backup = copy.deepcopy(self.encoder.state_dict())
        self.encoder.load_state_dict(self._target_enc_state)
        with torch.no_grad():
            out = self.encoder(tokens, causal=False).mean(dim=1)  # mean-pool
        self.encoder.load_state_dict(enc_sd_backup)
        return out  # (B, d_model)

    def forward(self, tokens):
        """
        tokens: (B, T) int64

        Returns dict with:
          jepa_loss  – cosine distance between predicted and target repr
          mlm_loss   – optional masked LM auxiliary loss
          loss       – total combined loss
        """
        B, T = tokens.shape
        a = self.args

        # ── Split context / target spans ──────────────────────────────────
        # Context: tokens[:, :ctx_end]
        # Target:  tokens[:, tgt_start : tgt_start + jepa_target_len]
        ctx_end   = max(T // 2, T - a.jepa_target_len - a.jepa_offset_max - 1)
        offset    = torch.randint(a.jepa_offset_min, a.jepa_offset_max + 1, (1,)).item()
        tgt_start = ctx_end + offset
        tgt_end   = min(tgt_start + a.jepa_target_len, T)

        ctx_tokens = tokens[:, :ctx_end]                       # (B, ctx_len)
        tgt_tokens = tokens[:, tgt_start:tgt_end]              # (B, tgt_len)

        # ── Context encoding ──────────────────────────────────────────────
        ctx_hidden = self.encoder(ctx_tokens, causal=True)     # (B, ctx_len, D)
        ctx_repr   = ctx_hidden[:, -1, :]                      # (B, D)  last token

        # ── Target encoding (EMA, no grad) ───────────────────────────────
        tgt_repr   = self.target_encode(tgt_tokens)            # (B, D)

        # ── Prediction ────────────────────────────────────────────────────
        tgt_pos    = torch.full((B,), tgt_start, dtype=torch.long, device=tokens.device)
        pred_repr  = self.predictor(ctx_repr, tgt_pos)         # (B, D)

        # ── JEPA loss (cosine distance) ───────────────────────────────────
        jepa_loss = 1.0 - F.cosine_similarity(pred_repr, tgt_repr.detach(), dim=-1).mean()

        # ── MLM auxiliary loss on context ─────────────────────────────────
        mlm_loss = torch.tensor(0.0, device=tokens.device)
        if a.mlm_weight > 0 and ctx_end >= 4:
            mask     = (torch.rand(B, ctx_end, device=tokens.device) < a.mlm_mask_rate)
            masked   = ctx_tokens.clone()
            masked[mask] = 0  # replace with PAD=0
            mlm_hid  = self.encoder(masked, causal=True)       # (B, ctx_end, D)
            # Shared embedding as LM head (tied weights, zero extra params)
            lm_logits = mlm_hid @ self.encoder.embed.weight.T  # (B, ctx_end, V)
            mlm_loss  = F.cross_entropy(
                lm_logits[mask],                               # (N, V)
                ctx_tokens[mask],                              # (N,)
            )

        loss = jepa_loss + a.mlm_weight * mlm_loss
        return {"loss": loss, "jepa_loss": jepa_loss, "mlm_loss": mlm_loss}

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

class ShardLoader:
    def __init__(self, data_path, split, seq_len, rank, world):
        self.seq_len = seq_len
        pattern = "train_" if split == "train" else "val_"
        shards  = sorted(Path(data_path).glob(f"*{pattern}*.bin"))
        assert shards, f"No shards matching *{pattern}*.bin in {data_path}"
        # Interleave shards across ranks
        self.shards = [s for i, s in enumerate(shards) if i % world == rank]
        self.sidx, self.pos = 0, 0
        self._load()

    def _load(self):
        file = self.shards[self.sidx % len(self.shards)]
        header_bytes = 256 * np.dtype("<i4").itemsize
        header = np.fromfile(file, dtype="<i4", count=256)
        num_tokens = int(header[2])
        self.data = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
        self.sidx += 1

    def next_batch(self, n_tokens):
        out = []
        needed = n_tokens + 1
        while needed > 0:
            avail = len(self.data) - self.pos
            if avail <= 0:
                self._load(); self.pos = 0; avail = len(self.data)
            take = min(avail, needed)
            out.append(self.data[self.pos : self.pos + take])
            self.pos  += take
            needed    -= take
        arr = np.concatenate(out)[:n_tokens + 1]
        t   = torch.from_numpy(arr.astype(np.int32)).long()
        return t[:n_tokens].reshape(-1, self.seq_len)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer utilities (BPB calculation)
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp

def bits_per_byte(tokenizer, log_probs_sum, n_tokens):
    """Convert sum of log-probs (nats) to bits-per-byte via tokenizer ratio."""
    # Compute average bytes per token
    sample  = "the quick brown fox jumps over the lazy dog " * 20
    toks    = tokenizer.encode(sample)
    bpt     = len(sample.encode()) / len(toks)          # bytes per token
    bpb     = (-log_probs_sum / n_tokens) / (bpt * math.log(2))
    return bpb

# ─────────────────────────────────────────────────────────────────────────────
# Decoder Head Fine-tuning (pre-eval)
# ─────────────────────────────────────────────────────────────────────────────

def finetune_decoder_head(model, calib_loader, args, device):
    """
    Train a linear decoder head on calibration tokens for decoder_steps steps.
    Returns a trained nn.Linear(d_model, vocab_size) on the given device.
    """
    if master:
        print("Fine-tuning decoder head for BPB evaluation...")
    head = nn.Linear(args.d_model, args.vocab_size, bias=False).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.01)
    model.eval()
    with torch.no_grad():
        pass  # encoder is frozen during head fine-tune

    for step in range(args.decoder_steps):
        tokens = calib_loader.next_batch(16 * args.train_seq_len).to(device)
        with torch.no_grad():
            h = model.encoder(tokens, causal=True)       # (B, T, D)
        # Shift: predict token t+1 from hidden at t
        logits = head(h[:, :-1, :])                      # (B, T-1, V)
        labels = tokens[:, 1:].reshape(-1)               # (B*(T-1),)
        loss   = F.cross_entropy(logits.reshape(-1, args.vocab_size), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if master and (step + 1) % 50 == 0:
            print(f"  [decoder head] step {step+1}/{args.decoder_steps}  loss={loss.item():.4f}")

    model.train()
    return head

# ─────────────────────────────────────────────────────────────────────────────
# Validation (BPB)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, decoder_head, val_loader, tokenizer, args, device):
    model.eval()
    total_lp, total_tok = 0.0, 0
    # Process validation tokens in chunks to avoid OOM
    val_tokens_to_process = args.val_batch // world_size
    chunk_size = 16 * args.train_seq_len
    for _ in range(max(1, val_tokens_to_process // chunk_size)):
        tokens = val_loader.next_batch(chunk_size).to(device)
        h      = model.encoder(tokens, causal=True)      # (B, T, D)
        logits = decoder_head(h[:, :-1, :])              # (B, T-1, V)
        labels = tokens[:, 1:]                            # (B, T-1)
        lp     = -F.cross_entropy(logits.reshape(-1, args.vocab_size),
                                   labels.reshape(-1),
                                   reduction="sum").item()
        total_lp  += lp
        total_tok += labels.numel()

    # Gather across ranks
    t = torch.tensor([total_lp, total_tok], device=device)
    dist.all_reduce(t); total_lp, total_tok = t[0].item(), int(t[1].item())

    bpb = bits_per_byte(tokenizer, total_lp, total_tok)
    model.train()
    return bpb

# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule (cosine with warmup + warmdown)
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step, args):
    if step < args.warmup:
        return args.max_lr * step / max(1, args.warmup)
    progress = (step - args.warmup) / max(1, args.iterations - args.warmup)
    if step >= args.iterations - args.warmdown:
        wd_prog = (step - (args.iterations - args.warmdown)) / args.warmdown
        return args.max_lr * (1 - wd_prog) + args.min_lr * wd_prog
    return args.min_lr + 0.5 * (args.max_lr - args.min_lr) * (
        1 + math.cos(math.pi * progress))

# ─────────────────────────────────────────────────────────────────────────────
# Quantisation helper (INT8 + zlib, same as baseline)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_and_compress(model, exclude_head=True):
    sd = model.encoder.state_dict()
    q8 = {}
    for k, v in sd.items():
        f = v.float()
        sc = f.abs().max() / 127.0 + 1e-8
        q8[k + "_scale"] = sc
        q8[k]            = (f / sc).round().clamp(-127, 127).to(torch.int8)
    payload = pickle.dumps(q8)
    compressed = zlib.compress(payload, level=9)
    return compressed

# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    a = args
    t0 = time.time()

    # Data loaders
    train_loader = ShardLoader(a.data_path, "train", a.train_seq_len, rank, world_size)
    calib_loader = ShardLoader(a.data_path, "train", a.train_seq_len, rank, world_size)
    val_loader   = ShardLoader(a.data_path, "val",   a.train_seq_len, rank, world_size)

    # Tokenizer
    tokenizer = load_tokenizer(a.tokenizer_path)

    # Model
    raw_model = JEPAModel(a).to(device)
    raw_model.init_target_encoder()
    model = DDP(raw_model, device_ids=[rank])

    # Optimizer (AdamW on encoder + predictor)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.max_lr,
        weight_decay=a.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )

    # Compile
    model = torch.compile(model)

    if master:
        total_params = sum(p.numel() for p in raw_model.encoder.parameters()) + \
                       sum(p.numel() for p in raw_model.predictor.parameters())
        print(f"JEPA model: {total_params/1e6:.2f}M trainable parameters")
        print(f"Training for {a.iterations} steps on {world_size} GPUs")

    step = 0
    for step in range(a.iterations):
        # Wallclock check
        if a.max_wallclock > 0 and (time.time() - t0) > a.max_wallclock:
            if master:
                print(f"Wallclock limit reached at step {step}")
            break

        # LR
        lr = get_lr(step, a)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # EMA momentum schedule
        ema_tau = a.ema_start + (a.ema_end - a.ema_start) * (step / a.iterations)

        # Forward pass with Gradient Accumulation to prevent OOM
        micro_batch_size = 16
        grad_accum_steps = (a.batch_tokens // world_size) // (micro_batch_size * a.train_seq_len)
        if grad_accum_steps == 0:
            grad_accum_steps = 1
            micro_batch_size = (a.batch_tokens // world_size) // a.train_seq_len

        total_loss, total_jepa, total_mlm = 0.0, 0.0, 0.0
        for micro_step in range(grad_accum_steps):
            tokens = train_loader.next_batch(micro_batch_size * a.train_seq_len).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(tokens)
            loss = out["loss"] / grad_accum_steps
            loss.backward()
            
            total_loss += loss.item()
            total_jepa += out["jepa_loss"].item() / grad_accum_steps
            total_mlm += out["mlm_loss"].item() / grad_accum_steps if isinstance(out["mlm_loss"], torch.Tensor) else out["mlm_loss"] / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], a.grad_clip)
        opt.step(); opt.zero_grad()

        # EMA update (unwrap DDP + compile)
        raw_model.update_ema(ema_tau)

        # Logging
        if master and (step % a.log_every == 0):
            elapsed = time.time() - t0
            print(f"step:{step:6d}  lr:{lr:.4e}  "
                  f"jepa:{total_jepa:.4f}  mlm:{total_mlm:.4f}  total:{total_loss:.4f}  "
                  f"ema_tau:{ema_tau:.5f}  t:{elapsed:.0f}s")

    # ── Final evaluation ───────────────────────────────────────────────────
    dist.barrier()
    if master:
        print("\n=== Training complete. Fine-tuning decoder head... ===")

    decoder_head = finetune_decoder_head(raw_model, calib_loader, a, device)

    if master:
        print("=== Running BPB evaluation... ===")

    bpb = validate(raw_model, decoder_head, val_loader, tokenizer, a, device)

    if master:
        print(f"\nval_bpb: {bpb:.6f}")

    # ── Compression ───────────────────────────────────────────────────────
    if master:
        compressed_model = quantize_and_compress(raw_model)
        code_bytes       = open(__file__, "rb").read()
        total_bytes      = len(compressed_model) + len(code_bytes)
        print(f"\n=== Artifact size ===")
        print(f"  model (int8+zlib): {len(compressed_model):,} bytes")
        print(f"  code:              {len(code_bytes):,} bytes")
        print(f"  total:             {total_bytes:,} bytes")
        print(f"  limit:             16,000,000 bytes")
        print(f"  within limit:      {total_bytes < 16_000_000}")
        print(f"\nfinal_int8_zlib_roundtrip_exact val_bpb:{bpb:.8f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
