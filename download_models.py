#!/usr/bin/env python3
"""
Downloads recommended GGUF models for CommandClaw.
Run this after setup.sh has installed dependencies.

Usage:
    python download_models.py          # auto-selects tier based on VRAM
    python download_models.py --tier 1 # CPU / low VRAM  (~2.5 GB)
    python download_models.py --tier 2 # 6-8 GB VRAM    (~5 GB)   [default for GPU]
    python download_models.py --tier 3 # 16+ GB VRAM    (~8 GB)
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path.home() / "models"

TIERS = {
    1: {
        "label": "Tier 1 — CPU / low VRAM",
        "models": [
            {
                "repo":     "bartowski/Qwen3-4B-GGUF",
                "filename": "Qwen3-4B-Q4_K_M.gguf",
                "note":     "Interview + synthesis model (~2.5 GB)",
            },
        ],
    },
    2: {
        "label": "Tier 2 — 6-8 GB VRAM (recommended)",
        "models": [
            {
                "repo":     "bartowski/Qwen3-8B-GGUF",
                "filename": "Qwen3-8B-Q4_K_M.gguf",
                "note":     "Interview + synthesis model (~5 GB)",
            },
        ],
    },
    3: {
        "label": "Tier 3 — 16+ GB VRAM",
        "models": [
            {
                "repo":     "bartowski/Qwen3-8B-GGUF",
                "filename": "Qwen3-8B-Q4_K_M.gguf",
                "note":     "Interview model (~5 GB)",
            },
            {
                "repo":     "bartowski/Qwen3-14B-GGUF",
                "filename": "Qwen3-14B-Q4_K_M.gguf",
                "note":     "Synthesis model (~8 GB)",
            },
        ],
    },
}


def detect_vram_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split("\n")[0]
        return float(out) / 1024
    except Exception:
        pass
    # Apple Silicon — shared memory, use 8 GB as conservative estimate
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return 8.0
    return 0.0


def pick_tier(vram: float) -> int:
    if vram >= 16:
        return 3
    if vram >= 6:
        return 2
    return 1


def download(repo: str, filename: str, dest_dir: Path):
    from huggingface_hub import hf_hub_download
    print(f"  Downloading {filename} from {repo}…")
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=None)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.tier:
        tier = args.tier
    else:
        vram = detect_vram_gb()
        tier = pick_tier(vram)
        print(f"Detected VRAM: {vram:.1f} GB → {TIERS[tier]['label']}")

    tier_info = TIERS[tier]
    print(f"\nDownloading {tier_info['label']}:")
    for m in tier_info["models"]:
        print(f"\n• {m['filename']}  ({m['note']})")
        target = MODELS_DIR / m["filename"]
        if target.exists():
            print(f"  Already exists, skipping.")
        else:
            download(m["repo"], m["filename"], MODELS_DIR)

    print(f"\nDone. Models in: {MODELS_DIR}")
    print("Run commandclaw.sh (Linux/Mac) to start.")


if __name__ == "__main__":
    main()
