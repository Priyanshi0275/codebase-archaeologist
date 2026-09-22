"""
Run this in Google Colab (free GPU: Runtime > Change runtime type > T4 GPU).
Fine-tunes a small LoRA adapter on git commit messages, so the adapter
learns to write refactor suggestions in a terse, engineer-style voice.

Two modes:
  - Single repo (--repo_path)  -> adapter learns THAT team's specific voice.
    Best for a focused demo where you ingest the same repo you trained on.
  - Multiple repos (--repo_paths, several)  -> adapter learns a general,
    repo-agnostic "good commit style" instead of one team's idiosyncrasies.
    Use this if you want the deployed app to sound reasonable on ANY repo
    someone ingests, not just one you trained on.

Setup in Colab:
    !pip install -r requirements-train.txt

Usage (single repo):
    python train_lora.py --repo_path /path/to/cloned/repo --output_dir ./lora-adapter

Usage (multiple repos, for a general style):
    python train_lora.py --repo_paths /path/repo1 /path/repo2 /path/repo3 --output_dir ./lora-adapter
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


def collect_from_multiple(repo_paths: list[str], max_commits_per_repo: int = 300) -> list[str]:
    """Pool commit messages across several repos, so the adapter learns
    a general style rather than one specific team's voice."""
    all_messages = []
    for path in repo_paths:
        try:
            msgs = collect_commit_messages(path, max_commits_per_repo)
            print(f"  {path}: {len(msgs)} messages")
            all_messages.extend(msgs)
        except Exception as e:
            print(f"  [WARN] Skipping {path}: {e}")
    return all_messages


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
    parser.add_argument("--repo_path", help="Single repo mode: path to one locally cloned git repo")
    parser.add_argument("--repo_paths", nargs="+", help="Multi-repo mode: several repo paths, for a general style")
    parser.add_argument("--output_dir", default="./lora-adapter")
    parser.add_argument("--max_commits", type=int, default=500, help="Used in single-repo mode")
    parser.add_argument("--max_commits_per_repo", type=int, default=300, help="Used in multi-repo mode")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if not args.repo_path and not args.repo_paths:
        parser.error("Pass either --repo_path (single repo) or --repo_paths (multiple repos)")

    if args.repo_paths:
        print(f"Collecting commit messages from {len(args.repo_paths)} repos ...")
        messages = collect_from_multiple(args.repo_paths, args.max_commits_per_repo)
        trained_on = args.repo_paths
    else:
        print(f"Collecting commit messages from {args.repo_path} ...")
        messages = collect_commit_messages(args.repo_path, args.max_commits)
        trained_on = [args.repo_path]

    print(f"Collected {len(messages)} commit messages total.")
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
            "trained_on_repos": trained_on,
            "style_mode": "general (multi-repo)" if args.repo_paths else "team-specific (single repo)",
            "n_commit_messages": len(messages),
            "epochs": args.epochs,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
        }, f, indent=2)

    print(f"Done. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
