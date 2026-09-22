FROM python:3.11-slim

WORKDIR /code

# git is needed for GitPython to clone/read repos
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# HF Spaces containers don't run as root — point HF's model cache at a
# writable folder inside /code so downloading the base model + adapter
# on first request doesn't fail on permissions.
ENV HF_HOME=/code/.cache/huggingface
RUN mkdir -p /code/.cache/huggingface /code/chroma_db /code/repo_cache \
    && chmod -R 777 /code

# Default to 7860 (Hugging Face Spaces' expected port). Override PORT
# via env if deploying elsewhere.
ENV PORT=7860
EXPOSE 7860

# Shell form so $PORT expands (Render/HF Spaces inject this at runtime)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
