"""karpa -> ralph rebrand compatibility (2026-06).

Guards the back-compat contract added when KarpaBase/KarpaConfig were
renamed to RalphBase/RalphConfig:

  1. `from model import KarpaBase, KarpaConfig` still resolves and the
     aliases are the *same objects* as the canonical Ralph* classes.
  2. A checkpoint saved in the canonical `{"model": state_dict,
     "config": asdict(cfg)}` form round-trips — proving the rename does
     not touch the serialized format (config is keyed by field name, not
     class name; state_dict keys are module-attribute paths).
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import KarpaBase, KarpaConfig, RalphBase, RalphConfig  # noqa: E402


def test_aliases_are_canonical_classes():
    assert KarpaConfig is RalphConfig
    assert KarpaBase is RalphBase


def test_checkpoint_roundtrips_under_renamed_classes(tmp_path):
    # Tiny model so the test is fast + CPU-only.
    cfg = RalphConfig(vocab_size=256, dim=32, n_layers=2, n_heads=2,
                      head_dim=16, max_seq_len=32, tie_embeddings=True)
    model = RalphBase(cfg)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, ckpt)

    # No class-name string is serialized — neither old nor new brand.
    raw = ckpt.read_bytes()
    for needle in (b"KarpaBase", b"KarpaConfig", b"RalphBase", b"RalphConfig"):
        assert needle not in raw

    # Reload: config is a plain field->value dict; rebuild via the canonical
    # class AND via the back-compat alias. Both must reconstruct identically.
    blob = torch.load(ckpt, weights_only=True, map_location="cpu")
    assert set(blob.keys()) == {"model", "config"}
    cfg_kwargs = {k: v for k, v in blob["config"].items()
                  if k in RalphConfig.__dataclass_fields__}

    for klass in (RalphBase, KarpaBase):
        rebuilt = klass(RalphConfig(**cfg_kwargs))
        rebuilt.load_state_dict(blob["model"])  # strict=True by default
        assert rebuilt.num_parameters() == model.num_parameters()
