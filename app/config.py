import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# HF Spaces expects 7860; Render/local default to 8000. Override via env either way.
PORT = int(os.getenv("PORT", "8000"))

# LoRA style rewriting — used LIVE by /query when enabled.
# LORA_ADAPTER_SOURCE can be either:
#   - a local folder path (e.g. "./lora-adapter", if you baked the adapter into the image), or
#   - a Hugging Face Hub repo id (e.g. "your-username/codebase-archaeologist-lora"),
#     in which case it's downloaded automatically on first use.
ENABLE_LORA_STYLE = os.getenv("ENABLE_LORA_STYLE", "false").lower() == "true"
LORA_ADAPTER_SOURCE = os.getenv("LORA_ADAPTER_SOURCE", "")
LORA_BASE_MODEL = os.getenv("LORA_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

if not GROQ_API_KEY:
    print("[WARN] GROQ_API_KEY is not set. /query will fail until you add it to .env")
