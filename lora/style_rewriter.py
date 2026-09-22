"""
Loads the LoRA adapter trained by train_lora.py and uses it to rewrite
a refactor suggestion in the repo's own commit-message voice.

Runs on CPU (the base model is only 0.5B params, so this is feasible
without a GPU — a few seconds per call). Kept separate from the main
FastAPI app: install requirements-train.txt only if you want to use this.

Usage:
    python style_rewriter.py --adapter_dir ./lora-adapter --text "Refactor the auth check"
"""
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load(adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def rewrite_in_style(model, tokenizer, text: str, max_new_tokens: int = 60) -> str:
    prompt = f"Write a commit message in this project's style.\nCommit message: {text}\n->"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default="./lora-adapter")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    model, tokenizer = load(args.adapter_dir)
    print(rewrite_in_style(model, tokenizer, args.text))
