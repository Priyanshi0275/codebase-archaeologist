"""
Run this in Google Colab (free GPU: Runtime > Change runtime type > T4 GPU).
It fine-tunes a small LoRA adapter on a repo's own commit messages, so
the adapter learns that team's specific tone/style for writing suggestions.

This is intentionally NOT part of the deployed FastAPI app — training
needs a GPU and extra heavy libraries (torch/transformers/peft) that
would slow down or break a free-tier web host. Train once here, then
(optionally) load the saved adapter locally via style_rewriter.py.

Setup in Colab:
    !pip install -r requirements-train.txt

Usage:
    python train_lora.py --repo_path /path/to/cloned/repo --output_dir ./lora-adapter
"""
import argparse
import json
import os

import git
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small enough for free Colab GPU + later CPU inference


def collect_commit_messages(repo_path: str, max_commits: int = 500) -> list[str]:
    repo = git.Repo(repo_path)
    messages = []
    for commit in repo.iter_commits(max_count=max_commits):
        msg = commit.message.strip()
        if len(msg) > 10:  # skip near-empty messages
            messages.append(msg)
    return messages


def build_dataset(messages: list[str], tokenizer):
    # Simple instruction-style framing so the adapter learns to WRITE
    # in this repo's commit-message voice, not just autocomplete text.
    examples = [
        f"Write a commit message in this project's style.\nCommit message: {m}"
        for m in messages
    ]

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_dict({"text": examples})
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", required=True, help="Path to a locally cloned git repo")
    parser.add_argument("--output_dir", default="./lora-adapter")
    parser.add_argument("--max_commits", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print(f"Collecting commit messages from {args.repo_path} ...")
    messages = collect_commit_messages(args.repo_path, args.max_commits)
    print(f"Collected {len(messages)} commit messages.")
    if len(messages) < 20:
        print("[WARN] Very few commit messages found — adapter quality will be limited. "
              "Try a repo with a longer, richer commit history.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(messages, tokenizer)

    training_args = TrainingArguments(
        output_dir="./lora-train-tmp",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],  # set to ["wandb"] if you want W&B tracking, needs WANDB_API_KEY
        fp16=True,
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save a small manifest for the resume/demo write-up
    with open(os.path.join(args.output_dir, "training_manifest.json"), "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "n_commit_messages": len(messages),
            "epochs": args.epochs,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
        }, f, indent=2)

    print(f"Done. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
