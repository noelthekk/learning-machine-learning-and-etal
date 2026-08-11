"""One-off diagnostic: sample raw completions from a GRPO checkpoint to see what the
model is actually generating, rather than only looking at the aggregate reward metrics
in trainer_state.json."""

from __future__ import annotations

import sys

import torch
from peft import PeftModel
from train import FORMAT_PATTERN, MODEL_ID, build_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/qwen2.5-0.5b-grpo-gsm8k/checkpoint-200"
NUM_SAMPLES = 5


def main() -> None:
    assert torch.cuda.is_available(), "No CUDA device visible -- check --gres was actually granted."
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {MODEL_ID}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")

    print(f"Loading LoRA adapter from {CHECKPOINT_DIR}...", flush=True)
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR).to("cuda")
    model.eval()

    print("Building dataset...", flush=True)
    dataset = build_dataset()

    print(f"Generating {NUM_SAMPLES} samples...\n", flush=True)
    for i in range(NUM_SAMPLES):
        prompt = dataset[i]["prompt"]
        label = dataset[i]["answer"]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(output_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)

        print(f"{'=' * 60}\nSAMPLE {i + 1} | ground-truth answer: {label}\n{'=' * 60}", flush=True)
        print(f"RAW COMPLETION:\n{completion!r}\n", flush=True)
        print(f"matches strict format regex: {FORMAT_PATTERN.match(completion.strip()) is not None}\n", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
