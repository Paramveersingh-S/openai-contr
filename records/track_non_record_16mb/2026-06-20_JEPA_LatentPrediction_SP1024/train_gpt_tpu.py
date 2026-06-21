"""
JEPA (Joint Embedding Predictive Architecture) for Parameter Golf
Non-record submission — openai/parameter-golf
Track: track_non_record_16mb

TPU VERSION (PyTorch XLA)
"""

import os, math, time, zlib, pickle, uuid, copy, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

# --- PyTorch XLA imports ---
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    # Data
    data_path:       str  = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path:  str  = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
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

# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Embedding
# ─────────────────────────────────────────────────────────────────────────────

def build_rope_cache(seq_len: int, head_dim: int, device) -> torch.Tensor:
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos   = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return torch.cat([freqs, freqs], dim=-1)

def apply_rope(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    B, H, T, D = x.shape
    cos = cos_sin[:T, :D//2].cos()[None, None]  
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
        r = self.h // self.kv
        K = K.repeat_interleave(r, dim=1)
        V = V.repeat_interleave(r, dim=1)
        
        # In XLA, scaled_dot_product_attention works, but we should ensure it's XLA-friendly.
        # F.scaled_dot_product_attention is well supported in PyTorch 2.0+ XLA.
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
# Context Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    def __init__(self, a: Args, device):
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
# JEPA Predictor
# ─────────────────────────────────────────────────────────────────────────────

class JEPAPredictor(nn.Module):
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
        p = self.pos_emb(target_pos)           
        z = torch.cat([ctx_repr, p], dim=-1)   
        return self.net(z)                     

# ─────────────────────────────────────────────────────────────────────────────
# Full JEPA Model
# ─────────────────────────────────────────────────────────────────────────────

class JEPAModel(nn.Module):
    def __init__(self, a: Args, device):
        super().__init__()
        self.args      = a
        self.encoder   = ContextEncoder(a, device)
        self.predictor = JEPAPredictor(a.d_model)
        self.target_encoder = None

    def init_target_encoder(self):
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self, tau: float):
        for tp, ep in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.copy_(tau * tp.data + (1.0 - tau) * ep.data)

    def target_encode(self, tokens):
        with torch.no_grad():
            out = self.target_encoder(tokens, causal=False).mean(dim=1) 
        return out  

    def forward(self, tokens):
        B, T = tokens.shape
        a = self.args

        # PyTorch XLA prefers static shapes when possible.
        # But this dynamic offset should compile down to XLA fine.
        import random
        ctx_end   = max(T // 2, T - a.jepa_target_len - a.jepa_offset_max - 1)
        offset    = random.randint(a.jepa_offset_min, a.jepa_offset_max)
        tgt_start = ctx_end + offset
        tgt_end   = min(tgt_start + a.jepa_target_len, T)

        ctx_tokens = tokens[:, :ctx_end]                       
        tgt_tokens = tokens[:, tgt_start:tgt_end]              

        ctx_hidden = self.encoder(ctx_tokens, causal=True)     
        ctx_repr   = ctx_hidden[:, -1, :]                      

        tgt_repr   = self.target_encode(tgt_tokens)            

        tgt_pos    = torch.full((B,), tgt_start, dtype=torch.long, device=tokens.device)
        pred_repr  = self.predictor(ctx_repr, tgt_pos)         

        jepa_loss = 1.0 - F.cosine_similarity(pred_repr, tgt_repr.detach(), dim=-1).mean()

        mlm_loss = torch.tensor(0.0, device=tokens.device)
        if a.mlm_weight > 0 and ctx_end >= 4:
            mask     = (torch.rand(B, ctx_end, device=tokens.device) < a.mlm_mask_rate)
            masked   = ctx_tokens.clone()
            masked[mask] = 0 
            mlm_hid  = self.encoder(masked, causal=True)       
            lm_logits = mlm_hid @ self.encoder.embed.weight.T  
            mlm_loss  = F.cross_entropy(
                lm_logits[mask],                               
                ctx_tokens[mask],                              
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
    sample  = "the quick brown fox jumps over the lazy dog " * 20
    toks    = tokenizer.encode(sample)
    bpt     = len(sample.encode()) / len(toks)          
    bpb     = (-log_probs_sum / n_tokens) / (bpt * math.log(2))
    return bpb

# ─────────────────────────────────────────────────────────────────────────────
# Decoder Head Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def finetune_decoder_head(model, calib_loader, args, device, master):
    if master:
        print("Fine-tuning decoder head for BPB evaluation...")
    head = nn.Linear(args.d_model, args.vocab_size, bias=False).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.01)
    model.eval()
    
    for step in range(args.decoder_steps):
        tokens = calib_loader.next_batch(4 * args.train_seq_len).to(device)
        with torch.no_grad():
            h = model.encoder(tokens, causal=True)       
        logits = head(h[:, :-1, :])                      
        labels = tokens[:, 1:].reshape(-1)               
        loss   = F.cross_entropy(logits.reshape(-1, args.vocab_size), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        xm.optimizer_step(opt)
        opt.zero_grad()
        if master and (step + 1) % 50 == 0:
            xm.master_print(f"  [decoder head] step {step+1}/{args.decoder_steps}  loss={loss.item():.4f}")

    model.train()
    return head

# ─────────────────────────────────────────────────────────────────────────────
# Validation (BPB)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, decoder_head, val_loader, tokenizer, args, device, world_size):
    model.eval()
    total_lp, total_tok = 0.0, 0
    val_tokens_to_process = args.val_batch // world_size
    chunk_size = 4 * args.train_seq_len
    for _ in range(max(1, val_tokens_to_process // chunk_size)):
        tokens = val_loader.next_batch(chunk_size).to(device)
        h      = model.encoder(tokens, causal=True)      
        logits = decoder_head(h[:, :-1, :])              
        labels = tokens[:, 1:]                           
        lp     = -F.cross_entropy(logits.reshape(-1, args.vocab_size),
                                   labels.reshape(-1),
                                   reduction="sum").item()
        total_lp  += lp
        total_tok += labels.numel()

    t = torch.tensor([total_lp, float(total_tok)], device=device)
    t = xm.all_reduce(xm.REDUCE_SUM, t)
    total_lp, total_tok = t[0].item(), int(t[1].item())

    bpb = bits_per_byte(tokenizer, total_lp, total_tok)
    model.train()
    return bpb

# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule
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
# XLA Multiprocessing Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def _mp_fn(index):
    a = args
    t0 = time.time()
    
    torch.manual_seed(a.seed + index)
    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    master = xm.is_master_ordinal(local=False)

    train_loader = ShardLoader(a.data_path, "train", a.train_seq_len, rank, world_size)
    calib_loader = ShardLoader(a.data_path, "train", a.train_seq_len, rank, world_size)
    val_loader   = ShardLoader(a.data_path, "val",   a.train_seq_len, rank, world_size)

    tokenizer = load_tokenizer(a.tokenizer_path)

    model = JEPAModel(a, device).to(device)
    model.init_target_encoder()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=a.max_lr,
        weight_decay=a.weight_decay,
        betas=(0.9, 0.95),
    )

    if master:
        total_params = sum(p.numel() for p in model.encoder.parameters()) + \
                       sum(p.numel() for p in model.predictor.parameters())
        xm.master_print(f"JEPA model: {total_params/1e6:.2f}M trainable parameters")
        xm.master_print(f"Training for {a.iterations} steps on {world_size} TPU Cores")

    for step in range(a.iterations):
        if a.max_wallclock > 0 and (time.time() - t0) > a.max_wallclock:
            if master: xm.master_print(f"Wallclock limit reached at step {step}")
            break

        lr = get_lr(step, a)
        for pg in opt.param_groups:
            pg["lr"] = lr

        ema_tau = a.ema_start + (a.ema_end - a.ema_start) * (step / a.iterations)

        micro_batch_size = 4
        grad_accum_steps = (a.batch_tokens // world_size) // (micro_batch_size * a.train_seq_len)
        if grad_accum_steps == 0:
            grad_accum_steps = 1
            micro_batch_size = (a.batch_tokens // world_size) // a.train_seq_len

        total_loss, total_jepa, total_mlm = 0.0, 0.0, 0.0
        for micro_step in range(grad_accum_steps):
            tokens = train_loader.next_batch(micro_batch_size * a.train_seq_len).to(device)
            # Mixed precision happens implicitly via XLA_USE_BF16=1 env var
            out = model(tokens)
            loss = out["loss"] / grad_accum_steps
            loss.backward()
            
            total_loss += loss.item()
            total_jepa += out["jepa_loss"].item() / grad_accum_steps
            total_mlm += out["mlm_loss"].item() / grad_accum_steps if isinstance(out["mlm_loss"], torch.Tensor) else out["mlm_loss"] / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], a.grad_clip)
        
        xm.optimizer_step(opt)
        opt.zero_grad()

        model.update_ema(ema_tau)

        if master and (step % a.log_every == 0):
            elapsed = time.time() - t0
            xm.master_print(f"step:{step:6d}  lr:{lr:.4e}  "
                  f"jepa:{total_jepa:.4f}  mlm:{total_mlm:.4f}  total:{total_loss:.4f}  "
                  f"ema_tau:{ema_tau:.5f}  t:{elapsed:.0f}s")

    xm.rendezvous("training_complete")
    if master:
        xm.master_print("\n=== Training complete. Fine-tuning decoder head... ===")

    decoder_head = finetune_decoder_head(model, calib_loader, a, device, master)

    if master:
        xm.master_print("=== Running BPB evaluation... ===")

    bpb = validate(model, decoder_head, val_loader, tokenizer, a, device, world_size)

    if master:
        xm.master_print(f"\nval_bpb: {bpb:.6f}")

    # No compression roundtrip in XLA version as it's meant just for speed. 
    # Use GPU run for final artifact generation.

if __name__ == "__main__":
    # Launch on 8 TPU cores
    xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')
