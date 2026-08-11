#!/usr/bin/env bash
#SBATCH --job-name=grpo-gsm8k
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --mem=32G
#SBATCH --output=logs/slurm_%j.log
#SBATCH --time=04:00:00

# Qwen2.5-0.5B (bf16) + LoRA is tiny (~1-2GB of weights) -- the 18GB MIG partition is
# sized for GRPO's real memory cost here: holding num_generations=4 full rollouts
# (up to max_completion_length=512 tokens each) in VRAM simultaneously per prompt,
# not the base model itself. --mem=32G covers host-side dataset/tokenizer overhead;
# see ../heretic/ideogram/submit.sbatch's history for why this is set explicitly
# rather than left to Slurm's default (8GB, too small).

export TMPDIR="$HOME/.tmp"
mkdir -p "$TMPDIR" logs

# --project is required: the repo root's own pyproject.toml is a separate, lighter
# project (dev tooling only) -- torch/transformers/etc. live in src/grpo's project.
PYTHONUNBUFFERED=1 uv run --project src/grpo python src/grpo/train.py
