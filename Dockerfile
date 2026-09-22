FROM python:3.11-slim

WORKDIR /code

# git is needed for GitPython to clone/read repos
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT expands (Render injects this at runtime)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
