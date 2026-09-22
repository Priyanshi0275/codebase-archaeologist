"""
Loads the trained LoRA adapter (base model: Qwen2.5-0.5B-Instruct) and
uses it to rewrite the Refactor Advisor's suggestion in the target
repo's own commit-message voice — live, as part of a real /query call.

This runs IN the deployed app now (not just offline), so it needs a
host with enough RAM to hold the base model + adapter — Hugging Face
Spaces' free CPU tier (16GB RAM) works; Render's free tier (512MB)
does not. See README for the HF Spaces deploy steps.

The adapter is loaded lazily (only on first use) and cached, so app
startup stays fast even before anyone calls /query. If no adapter is
configured, style rewriting is silently skipped and the raw refactor
suggestion is returned as-is — the app still works end to end without
it, it's just not styled.
"""
import os
import threading

from app.config import LORA_ADAPTER_SOURCE, LORA_BASE_MODEL, ENABLE_LORA_STYLE

_model = None
_tokenizer = None
_load_lock = threading.Lock()
_load_attempted = False
_load_failed_reason = None


def _try_load():
    global _model, _tokenizer, _load_attempted, _load_failed_reason

    if _load_attempted:
        return
    with _load_lock:
        if _load_attempted:
            return
        _load_attempted = True

        if not ENABLE_LORA_STYLE or not LORA_ADAPTER_SOURCE:
            _load_failed_reason = "LoRA styling disabled or no adapter configured"
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_SOURCE)
            base_model = AutoModelForCausalLM.from_pretrained(LORA_BASE_MODEL)
            model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_SOURCE)
            model.eval()

            _tokenizer = tokenizer
            _model = model
        except Exception as e:
            _load_failed_reason = f"Failed to load LoRA adapter: {e}"
            _model = None
            _tokenizer = None


def is_available() -> bool:
    _try_load()
    return _model is not None


def status() -> dict:
    _try_load()
    return {
        "enabled": ENABLE_LORA_STYLE,
        "adapter_source": LORA_ADAPTER_SOURCE or None,
        "loaded": _model is not None,
        "reason": _load_failed_reason,
    }


def rewrite_in_style(text: str, max_new_tokens: int = 60) -> str:
    """Rewrite text in the fine-tuned repo voice. Falls back to the
    original text untouched if the adapter isn't available."""
    _try_load()
    if _model is None or _tokenizer is None:
        return text

    prompt = (
        "Rewrite the following suggestion in this project's own commit-message "
        f"voice, keep it short.\nSuggestion: {text}\n->"
    )
    inputs = _tokenizer(prompt, return_tensors="pt")
    output = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=_tokenizer.eos_token_id,
    )
    decoded = _tokenizer.decode(output[0], skip_special_tokens=True)
    rewritten = decoded[len(prompt):].strip()
    return rewritten if rewritten else text
