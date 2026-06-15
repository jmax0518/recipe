# Ralph — canonical training recipe

The current best-known open recipe for the **small-LLM pretraining track** of the [Ralph](https://github.com/RalphLabsAI/ralph) Bittensor subnet. Each accepted miner patch becomes a new commit on `main`, tagged as `recipe-vX.Y.Z`, with the resulting metrics in the release notes.

This repo is **the artifact**. The protocol that scores patches and crowns kings lives at [RalphLabsAI/ralph](https://github.com/RalphLabsAI/ralph); this repo is what miners patch.

## Layout

| Path | What |
|---|---|
| `model/` | Ralph-base — Llama-style transformer (RMSNorm, RoPE, SwiGLU, MHA) |
| `recipe/` | Canonical training loop, optimizer, LR schedule |
| `configs/` | `proxy_cpu_smoke.json`, `h100_proxy.json`, `h100_default.json`, `h100_scale.json` |
| `data/` | Tokenizer, data manifest schema, FineWeb-Edu prepare script |

## Working on a patch (miner workflow)

1. Fork this repo to your own GitHub account.
2. Clone your fork **and** the protocol repo side-by-side:
   ```
   git clone git@github.com:<your-gh>/recipe.git
   git clone git@github.com:RalphLabsAI/ralph.git
   ```
3. Edit files in your `recipe/` clone — change a learning-rate schedule, tweak the warmup, propose a new initialization, etc.
4. Run the proof test from the protocol repo:
   ```
   cd ralph
   RALPH_RECIPE_DIR=../recipe python scripts/miner_run.py \
     --patch <(cd ../recipe && git diff main) \
     --label round1 \
     --config configs/h100_proxy.json \
     --tier unverified
   ```
5. If the proof test produces a bundle and the validator scores it favourably, your patch gets merged here as a tagged release.

For full miner setup on a rented H100, see [docs/h100_miner_setup.md](https://github.com/RalphLabsAI/ralph/blob/main/docs/h100_miner_setup.md) in the protocol repo.

## Releases

Each accepted king change publishes a release here named `recipe-vX.Y.Z`. The release body contains:
- `val_bpb` improvement vs previous king
- compute cost (H100-normalized)
- miner hotkey + GitHub attribution
- proof bundle URL (HuggingFace)
- wandb run URL

This is the lineage of the canonical recipe.

## License

Apache-2.0.
