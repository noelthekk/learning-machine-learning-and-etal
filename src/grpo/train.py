"""GRPO training of Qwen2.5-0.5B (base) on GSM8K math reasoning.

Sized for an 18GB H200 MIG partition: LoRA adapters instead of full fine-tuning,
bf16 throughout, group size 4, and a 512-token completion cap. Reward is the sum of
three deterministic, rule-based signals -- no reward model, no human feedback:

  - format_reward:   1.0 if the completion is exactly
                     `<think>...</think>\\s*Answer:\\s*<int>`, else 0.0.
  - accuracy_reward: 2.0 if the integer after 'Answer: ' matches GSM8K's
                     '#### <int>' label exactly, else 0.0.
  - brevity_penalty: -0.5 if the completion exceeds 800 characters, else 0.0 --
                     guards against runaway/looping generations, not a reward for
                     being terse per se.

GRPOTrainer's own divisibility rule: generation_batch_size (= per_device_train_batch_size
* num_processes * gradient_accumulation_steps, when steps_per_generation is unset) must
be divisible by num_generations. Here: 2 * 1 * 4 = 8, and 8 % 4 == 0.
"""

from __future__ import annotations

import re

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

MODEL_ID: str = "Qwen/Qwen2.5-0.5B"
DATASET_ID: str = "openai/gsm8k"
DATASET_CONFIG: str = "main"
NUM_TRAIN_EXAMPLES: int = 500
MAX_COMPLETION_LENGTH: int = 512
BREVITY_CHAR_LIMIT: int = 800
OUTPUT_DIR: str = "outputs/qwen2.5-0.5b-grpo-gsm8k"

PROMPT_TEMPLATE: str = (
    "Solve the following grade-school math problem.\n"
    "Show your step-by-step reasoning inside <think></think> tags, then give the "
    "final answer as an integer immediately after 'Answer: ' with nothing after the "
    "number.\n\n"
    "Question: {question}\n"
)

FORMAT_PATTERN: re.Pattern[str] = re.compile(r"^<think>.*?</think>\s*Answer:\s*\d+$", re.DOTALL)
FINAL_ANSWER_PATTERN: re.Pattern[str] = re.compile(r"Answer:\s*(-?\d+)")
GSM8K_LABEL_PATTERN: re.Pattern[str] = re.compile(r"####\s*(-?\d+)")


def build_dataset() -> Dataset:
    """Load the first NUM_TRAIN_EXAMPLES GSM8K training rows and reformat them into
    GRPOTrainer's expected {"prompt": str} shape"""
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=f"train[:{NUM_TRAIN_EXAMPLES}]")

    def format_example(example: dict) -> dict:
        label_match = GSM8K_LABEL_PATTERN.search(example["answer"])
        if label_match is None:
            raise ValueError(f"Could not find a '#### <number>' label in: {example['answer']!r}")
        return {
            "prompt": PROMPT_TEMPLATE.format(question=example["question"]),
            "answer": label_match.group(1),
        }

    return dataset.map(format_example, remove_columns=dataset.column_names)


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """1.0 if the completion exact, else 0.0."""
    return [1.0 if FORMAT_PATTERN.match(completion.strip()) else 0.0 for completion in completions]


def accuracy_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """2.0 if the completion exactly matches the GSM8K label, else 0.0."""
    rewards = []
    for completion, label in zip(completions, answer, strict=True):
        match = FINAL_ANSWER_PATTERN.search(completion)
        rewards.append(2.0 if match is not None and match.group(1) == label else 0.0)
    return rewards


def brevity_penalty(completions: list[str], **kwargs) -> list[float]:
    """-0.5 if the completion goes beyond BREVITY_CHAR_LIMIT, else 0.0."""
    return [-0.5 if len(completion) > BREVITY_CHAR_LIMIT else 0.0 for completion in completions]


def load_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    return model, tokenizer


def main() -> None:
    assert torch.cuda.is_available(), "No CUDA device visible -- check --gres was actually granted."
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    free_vram, total_vram = torch.cuda.mem_get_info()
    print(f"VRAM: {free_vram / 1e9:.2f} GB free / {total_vram / 1e9:.2f} GB total", flush=True)

    model, tokenizer = load_model_and_tokenizer()
    train_dataset = build_dataset()
    print(f"Loaded {len(train_dataset)} training examples from {DATASET_ID}/{DATASET_CONFIG}", flush=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        bf16=True,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward, brevity_penalty],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. LoRA adapter saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
