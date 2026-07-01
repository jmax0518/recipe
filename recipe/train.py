"""
Canonical training loop for the Ralph launch track.

This file is part of the recipe — miners may patch it (subject to the
restricted-files contract). The proof-test runner invokes this script with a
fixed config; the training is deterministic given (config, seed, manifest).

Outputs written to `--out-dir`:
  checkpoint.pt         the final model state_dict
  training_log.jsonl    one JSON line per step (loss, lr, throughput, gradnorm)
  final_state.json      run summary (steps, final loss, wall-clock, total tokens)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import TokenShardDataset
from model import RalphBase, RalphConfig


@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 50257
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    ffn_mult: float = 8 / 3
    max_seq_len: int = 1024

    # Training
    seq_len: int = 256
    batch_size: int = 16
    micro_batch_size: int = 16  # gradient accumulation = batch_size / micro_batch_size
    total_steps: int = 200
    warmup_steps: int = 20
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule shape. "cosine" = warmup + cosine decay to min_lr (canonical).
    # "wsd" = warmup, hold max_lr for stable_frac of the post-warmup span, then
    # cosine-decay to min_lr over the remaining (1 - stable_frac). Default stays
    # "cosine" so existing configs are unchanged; stable_frac is honoured from
    # config (re-added cleanly after the forged #618 king, which hardcoded 0.8,
    # was reverted — this carries no data/manifest bypass, only the LR shape).
    schedule: str = "cosine"
    stable_frac: float = 0.8

    # Optimizer. "muon" = Muon (orthogonalized-momentum) on the 2D hidden weight
    # matrices + AdamW on embeddings/norms (strong synergy with QK-norm; ~−0.13
    # val_bpb vs AdamW at the h100_proxy scale). "adamw" = AdamW on everything.
    optimizer: str = "muon"
    muon_lr: float = 0.04
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # Separate AdamW base LR for the embedding/head matrix (tok_embed/lm_head).
    # None => fall back to max_lr (preserves prior behaviour). The Muon configs
    # ship embed_lr but the canonical loop ignored it; this honours it. Norms and
    # the readout-calibration params stay on max_lr (see build_optimizer).
    embed_lr: float | None = None

    # Data + reproducibility
    manifest_path: str = "data/data_manifest.json"
    data_base_dir: str = "data"
    data_seed: int = 1337
    init_seed: int = 1337

    # Optional warm start from a previously saved checkpoint (checkpoint.pt or a
    # checkpoints/step_*.pt). Loads ONLY the model weights — the saved
    # checkpoints carry no optimizer state, so Muon/AdamW momentum is re-zeroed
    # and the LR schedule restarts from step 0. None => fresh random init
    # (canonical proof-test behaviour is unchanged).
    init_from: str | None = None

    # Precision
    use_bf16: bool = True  # bf16 autocast on CUDA; ignored on CPU

    # Optional periodic validation eval (research/diagnostic tool — OFF by
    # default so the canonical, deterministic proof-test run is unchanged).
    # When val_every > 0 the loop evaluates held-out val_bpb every val_every
    # steps and logs it next to the training metrics, so you can SEE whether/
    # when a config starts to overfit. This is diagnostic only: it does NOT
    # early-stop and does NOT affect the scored number (the protocol's hidden
    # eval remains the source of truth).
    val_every: int = 0
    # Held-out eval shard (uint16 tokens). Default resolves to the recipe's
    # eval/private/active_tokens.bin produced by `data.prepare --eval-tokens`.
    val_eval_path: str | None = None
    val_max_tokens: int = 1_000_000  # cap eval cost (tokens read from the shard)
    val_batch_size: int = 16

    # Logging
    log_every: int = 10

    @property
    def grad_accum_steps(self) -> int:
        assert self.batch_size % self.micro_batch_size == 0
        return self.batch_size // self.micro_batch_size


def set_determinism(seed: int) -> None:
    """Set all the knobs we can to get deterministic training. Not bit-perfect
    on GPU — see whitepaper §5.2 note on cuBLAS/atomic-reduction non-determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup + (cosine | WSD) LR envelope. Returns an absolute LR; the training
    loop converts it to a per-group fraction via lr_frac = lr / max_lr."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if getattr(cfg, "schedule", "cosine") == "wsd":
        # Hold the peak for stable_frac, then cosine-cooldown over the tail.
        if progress <= cfg.stable_frac:
            return cfg.max_lr
        decay = (progress - cfg.stable_frac) / max(1e-9, 1.0 - cfg.stable_frac)
        return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * decay))
    return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def build_model(cfg: TrainConfig) -> RalphBase:
    return RalphBase(RalphConfig(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        ffn_mult=cfg.ffn_mult,
        max_seq_len=cfg.max_seq_len,
    ))


def load_checkpoint(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict:
    """Warm-start `model` from a saved checkpoint written by this script.

    Accepts either the final `checkpoint.pt` or a periodic
    `checkpoints/step_*.pt`. Both store `{"model": state_dict, "config": ...}`
    (periodic ones also carry "step"). Only the model weights are restored —
    optimizer state is NOT saved by the training loop, so momentum buffers stay
    zeroed and the LR schedule restarts from step 0 (this is a warm start, not a
    bit-exact resume). Returns the checkpoint dict (sans weights) for logging.

    Loads with weights_only=True (safe; the payload is plain tensors/JSON-able
    config) and strict key matching so a geometry mismatch fails loudly rather
    than silently dropping layers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"init_from checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Older torch (<2.0) has no weights_only kwarg; fall back.
        ckpt = torch.load(path, map_location=device)
    if "model" not in ckpt:
        raise KeyError(f"checkpoint {path} has no 'model' state_dict (keys: {list(ckpt)})")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    meta = {k: v for k, v in ckpt.items() if k != "model"}
    print(f"[train] warm-start from {path} (step={meta.get('step', 'final')}); "
          f"optimizer state NOT restored — momentum re-zeroed, schedule from step 0")
    if missing or unexpected:
        print(f"[train]   missing keys={list(missing)} unexpected keys={list(unexpected)}")
    return meta


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to orthogonalize the update matrix (Muon).
    Computes G (G^T G)^(-1/2) approximately via a quintic iteration in bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum orthogonalized by Newton-Schulz, for 2D hidden weight matrices.
    See Keller Jordan's modded-nanogpt. Embeddings/heads/norms use AdamW instead."""

    def __init__(self, params, lr=0.04, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(p.grad)
                upd = p.grad.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = _zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                # Scale so the RMS update magnitude is ~LR-invariant to matrix shape.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(upd, alpha=-lr * scale)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """Returns a LIST of optimizers stepped together. Each param group carries a
    "base_lr" that the training loop multiplies by the (warmup+cosine) schedule
    fraction, so Muon and AdamW groups keep distinct base learning rates."""
    if cfg.optimizer == "muon":
        muon_params, embed_params, norm_params = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "tok_embed" in n or "lm_head" in n:
                embed_params.append(p)
            elif p.dim() >= 2:
                muon_params.append(p)
            else:
                norm_params.append(p)
        muon = Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps)
        # Embeddings/head get their own base_lr (cfg.embed_lr) when set; norms and
        # the readout-calibration params stay on max_lr so wiring embed_lr does not
        # silently re-tune them. base_lr is the value the training loop scales by
        # the schedule fraction (lr_frac), so each group keeps a distinct peak LR.
        embed_base = cfg.embed_lr if cfg.embed_lr else cfg.max_lr
        adamw = torch.optim.AdamW(
            [
                {"params": embed_params, "weight_decay": cfg.weight_decay, "base_lr": embed_base},
                {"params": norm_params, "weight_decay": 0.0, "base_lr": cfg.max_lr},
            ],
            lr=cfg.max_lr,
            betas=(cfg.beta1, cfg.beta2),
        )
        for grp in muon.param_groups:
            grp["base_lr"] = cfg.muon_lr
        return [muon, adamw]

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    adamw = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
    )
    for grp in adamw.param_groups:
        grp["base_lr"] = cfg.max_lr
    return [adamw]


def _init_wandb(cfg: TrainConfig, out_dir: Path, use_wandb: bool) -> object | None:
    if not use_wandb:
        return None
    try:
        import wandb
        miner_gh = os.environ.get("RALPH_MINER_GH", "")
        miner_wallet = os.environ.get("BT_WALLET", "")
        run_config = {k: v for k, v in asdict(cfg).items()}
        if miner_gh:
            run_config["miner_github"] = miner_gh
        if miner_wallet:
            run_config["miner_wallet"] = miner_wallet
        tags = ["proof-test", f"{cfg.dim}d", f"{cfg.n_layers}L"]
        if miner_gh:
            tags.append(f"gh:{miner_gh}")
        if miner_wallet:
            tags.append(f"wallet:{miner_wallet}")
        name_prefix = f"{miner_gh}-" if miner_gh else ""
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "ralphlabs-hub"),
            project=os.environ.get("WANDB_PROJECT", "ralph"),
            name=f"{name_prefix}train-{cfg.dim}d-{cfg.n_layers}L-{cfg.total_steps}s",
            config=run_config,
            dir=str(out_dir),
            tags=tags,
        )
        return run
    except Exception as e:
        print(f"[train] wandb init failed ({e}), continuing without it")
        return None


# Pinned validation window, matching the protocol's hidden eval (eval/val_bpb.py
# EVAL_SEQ_LEN). Using the same window keeps the in-loop val_bpb comparable to
# the number the validator will score.
VAL_EVAL_SEQ_LEN = 512
# Token-to-byte ratio used to convert nats -> bits-per-byte. Matches the
# protocol's DEFAULT_BYTES_PER_TOKEN (GPT-2 BPE on English ~4.0 bytes/token).
VAL_BYTES_PER_TOKEN = 4.0


def _resolve_val_path(cfg: TrainConfig) -> Path | None:
    """Locate the held-out eval shard. Honours cfg.val_eval_path, else falls
    back to the recipe's eval/private/active_tokens.bin. Returns None if the
    file is absent so the caller can disable val eval with a warning."""
    if cfg.val_eval_path:
        p = Path(cfg.val_eval_path)
    else:
        p = Path(__file__).resolve().parent.parent / "eval" / "private" / "active_tokens.bin"
    return p if p.exists() else None


@torch.no_grad()
def evaluate_val_bpb(
    model: torch.nn.Module,
    eval_tokens: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict:
    """Held-out val_bpb over non-overlapping windows.

    Mirrors eval/val_bpb.compute_val_bpb: summed cross-entropy (nats) over
    packed (seq_len+1) windows, normalized by log(2) * tokens * bytes_per_token.
    Self-contained (no protocol import) so the recipe stays standalone.
    """
    was_training = model.training
    model.eval()
    seq_len = max(2, min(VAL_EVAL_SEQ_LEN, cfg.max_seq_len))
    n = len(eval_tokens)
    n_windows = max(1, (n - 1) // seq_len)
    total_nats = 0.0
    total_tokens = 0
    batch_inp: list[torch.Tensor] = []
    batch_tgt: list[torch.Tensor] = []
    for w in range(n_windows):
        start = w * seq_len
        ids = eval_tokens[start : start + seq_len + 1]
        if len(ids) < seq_len + 1:
            break
        batch_inp.append(torch.from_numpy(ids[:-1].astype(np.int64)))
        batch_tgt.append(torch.from_numpy(ids[1:].astype(np.int64)))
        if len(batch_inp) == cfg.val_batch_size or w == n_windows - 1:
            inp = torch.stack(batch_inp).to(device)
            tgt = torch.stack(batch_tgt).to(device)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(inp)
            loss_sum = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                tgt.reshape(-1),
                reduction="sum",
            )
            total_nats += loss_sum.item()
            total_tokens += tgt.numel()
            batch_inp.clear()
            batch_tgt.clear()
    if was_training:
        model.train()
    total_bytes = total_tokens * VAL_BYTES_PER_TOKEN
    val_bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float("inf")
    nll_per_token = total_nats / max(total_tokens, 1)
    return {
        "val_bpb": val_bpb,
        "val_loss": nll_per_token,
        "val_tokens": total_tokens,
    }


def train(cfg: TrainConfig, out_dir: Path, use_wandb: bool = False) -> dict:
    set_determinism(cfg.init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    if cfg.init_from:
        load_checkpoint(cfg.init_from, model, device)
    optimizers = build_optimizer(model, cfg)
    ds = TokenShardDataset(cfg.manifest_path, cfg.data_base_dir, cfg.seq_len, cfg.data_seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    log_f = log_path.open("w")

    wb_run = _init_wandb(cfg, out_dir, use_wandb)

    use_amp = cfg.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    # bf16 has enough dynamic range that no GradScaler is needed (Muon orthogonalizes
    # in bf16 internally; AdamW groups are range-safe), so we step optimizers directly.

    n_params = model.num_parameters()
    n_params_no_embed = model.num_parameters(exclude_embeddings=True)
    print(f"[train] device={device} params={n_params:,} (no embeddings: {n_params_no_embed:,})")
    print(f"[train] precision={'bf16' if use_amp else 'fp32'}")
    print(f"[train] manifest tokens={ds.total_tokens:,} hash={ds.manifest.manifest_hash()[:16]}…")
    print(f"[train] steps={cfg.total_steps} batch={cfg.batch_size} micro={cfg.micro_batch_size} seq={cfg.seq_len}")
    if wb_run:
        print(f"[train] wandb: {wb_run.url}")

    # Optional held-out validation eval (diagnostic; see TrainConfig.val_every).
    val_tokens: np.ndarray | None = None
    if cfg.val_every > 0:
        val_path = _resolve_val_path(cfg)
        if val_path is None:
            print("[train] val_every>0 but no eval shard found "
                  "(set val_eval_path or run data.prepare --eval-tokens); "
                  "validation eval DISABLED.")
        else:
            raw = np.memmap(val_path, dtype=np.uint16, mode="r")
            if cfg.val_max_tokens and len(raw) > cfg.val_max_tokens:
                raw = raw[: cfg.val_max_tokens]
            val_tokens = np.asarray(raw)
            print(f"[train] val eval ON: every {cfg.val_every} steps over "
                  f"{len(val_tokens):,} tokens from {val_path}")

    start = time.time()
    tokens_seen = 0
    last_loss = float("nan")
    last_val_metrics: dict | None = None
    for step in range(cfg.total_steps):
        lr = cosine_lr(step, cfg)
        # Scale each optimizer's per-group base_lr by the schedule fraction so
        # the Muon and AdamW groups keep distinct learning rates.
        lr_frac = lr / cfg.max_lr
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_frac
            opt.zero_grad(set_to_none=True)

        step_loss = 0.0
        for accum in range(cfg.grad_accum_steps):
            sub_step = step * cfg.grad_accum_steps + accum
            inp, tgt = ds.get_batch(sub_step, cfg.micro_batch_size)
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(inp, targets=tgt)
            scaled_loss = loss / cfg.grad_accum_steps
            scaled_loss.backward()
            step_loss += loss.item() / cfg.grad_accum_steps
            tokens_seen += cfg.micro_batch_size * cfg.seq_len

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        for opt in optimizers:
            opt.step()

        last_loss = step_loss
        elapsed = time.time() - start
        tok_per_s = tokens_seen / max(elapsed, 1e-6)

        entry = {
            "step": step,
            "loss": step_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_seen": tokens_seen,
            "tokens_per_sec": tok_per_s,
            "elapsed_s": elapsed,
        }
        # Diagnostic held-out eval at the configured cadence (and on the final
        # step). Folded into the same log entry so the JSONL/wandb carry a
        # train-loss vs val_bpb pair you can watch for overfitting.
        val_metrics: dict | None = None
        if val_tokens is not None and (
            step % cfg.val_every == 0 or step == cfg.total_steps - 1
        ):
            val_metrics = evaluate_val_bpb(
                model, val_tokens, cfg, device, use_amp, amp_dtype
            )
            entry.update(val_metrics)
            last_val_metrics = val_metrics
        # recipe-v4: gate the JSONL write under log_every so long runs don't make
        # one line per step (the proof-test turns each ~10 lines into a per-epoch
        # NRAS attestation -> thousands of calls -> NRAS rate-limit/timeout).
        if step % cfg.log_every == 0 or step == cfg.total_steps - 1 or val_metrics:
            log_f.write(json.dumps(entry) + "\n")
        log_f.flush()
        if wb_run:
            wb_run.log(entry, step=step)
        if step % cfg.log_every == 0 or step == cfg.total_steps - 1 or val_metrics:
            val_str = (
                f" val_bpb={val_metrics['val_bpb']:.4f} val_loss={val_metrics['val_loss']:.4f}"
                if val_metrics else ""
            )
            print(
                f"[step {step:4d}/{cfg.total_steps}] loss={step_loss:.4f} lr={lr:.2e} "
                f"|g|={grad_norm:.2f} tok/s={tok_per_s:,.0f}{val_str}",
                flush=True,
            )
        if (step % 2000 == 0 and step > 0) or step == cfg.total_steps - 1:
            _ckpt_dir = out_dir / "checkpoints"
            _ckpt_dir.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, _ckpt_dir / f"step_{step:06d}.pt")
            with (out_dir / "progress.tsv").open("a") as _pf:
                _pf.write(f"{step}\t{step_loss:.6f}\n")
                _pf.flush()
    log_f.close()
    wb_url = None
    if wb_run:
        wb_url = wb_run.url
        try:
            history = wb_run.history(pandas=False)
            (out_dir / "wandb_metrics.json").write_text(json.dumps(history, indent=2))
            (out_dir / "wandb_run_url.txt").write_text(wb_url + "\n")
            print(f"[train] wandb metrics exported ({len(history)} steps)")
        except Exception as e:
            print(f"[train] wandb export failed ({e}), continuing")
        wb_run.finish()

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, ckpt_path)

    summary = {
        "steps": cfg.total_steps,
        "final_loss": last_loss,
        "tokens_seen": tokens_seen,
        "wall_clock_s": time.time() - start,
        "n_params": n_params,
        "n_params_no_embed": n_params_no_embed,
        "manifest_hash": ds.manifest.manifest_hash(),
        "device": str(device),
        "precision": "bf16" if use_amp else "fp32",
        "wandb_url": wb_url,
        "init_from": cfg.init_from,
        "config": asdict(cfg),
    }
    if last_val_metrics is not None:
        summary["final_val_bpb"] = last_val_metrics["val_bpb"]
        summary["final_val_loss"] = last_val_metrics["val_loss"]
    (out_dir / "final_state.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done. final loss={last_loss:.4f} wall={summary['wall_clock_s']:.1f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None, help="Optional JSON config override.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (requires `pip install wandb`)")
    p.add_argument("--val-every", type=int, default=None,
                   help="Diagnostic: evaluate held-out val_bpb every N steps (0=off). "
                        "Does not affect the scored run; for watching overfitting.")
    p.add_argument("--val-eval-path", type=Path, default=None,
                   help="Held-out eval shard (uint16). Default: eval/private/active_tokens.bin")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Warm-start model weights from a checkpoint.pt or "
                        "checkpoints/step_*.pt (optimizer state not restored).")
    args = p.parse_args()

    cfg = TrainConfig()
    if args.config and args.config.exists():
        overrides = json.loads(args.config.read_text())
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.manifest is not None:
        cfg.manifest_path = str(args.manifest)
    if args.seed is not None:
        cfg.init_seed = args.seed
        cfg.data_seed = args.seed
    if args.val_every is not None:
        cfg.val_every = args.val_every
    if args.val_eval_path is not None:
        cfg.val_eval_path = str(args.val_eval_path)
    if args.init_from is not None:
        cfg.init_from = str(args.init_from)

    train(cfg, args.out_dir, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
