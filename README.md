# Legacy Codebase Archaeologist

**A multi-agent system that investigates why legacy code exists and how risky it is to change — grounded in a repo's own commit history, not generic guessing.**

---

## The idea

Every codebase has functions nobody wants to touch because nobody remembers why they're there. This project points three AI agents at a repo's `git log` and asks them to actually dig it up — with receipts.

Ask something like *"Why does the retry logic in payment.py exist?"* and get back:

- **Historian** — explains why the code likely exists, citing specific commit SHAs as evidence
- **Dependency Risk Analyst** — flags what tends to break alongside it, rates the risk of changing it (low / medium / high)
- **Refactor Advisor** — proposes one concrete, low-risk next step

Each agent retrieves real evidence from a vector store before answering — no agent is allowed to just make something up.

## How it works

```
GitHub repo URL
      │
      ▼
GitPython clones it, walks commit history
      │
      ▼
Commit messages + diffs → chunked (LangChain) → embedded → stored in ChromaDB
      │
      ▼
Question comes in ──► CrewAI runs 3 agents in sequence, each searching
                       Chroma for evidence before answering (Groq LLM)
      │
      ▼
Historian → Dependency Analyst → Refactor Advisor
   (each agent's output feeds the next as context)
```

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| Agent orchestration | [CrewAI](https://github.com/crewAIInc/crewAI) | pinned `<1.0.0` — newer versions have an unresolved bug sending unsupported params to non-native LLM providers |
| LLM | [Groq](https://groq.com) (`llama-3.3-70b-versatile`) | free tier, fast inference |
| Vector store | [ChromaDB](https://www.trychroma.com/) | local, embedded, no hosted DB needed |
| Embeddings | Chroma's built-in ONNX MiniLM | deliberately avoids sentence-transformers/torch to keep memory low on a free host |
| Chunking | LangChain text splitters | |
| Git parsing | [GitPython](https://gitpython.readthedocs.io/) | reads `.git` directly, no GitHub API rate limits |
| Backend | FastAPI + Docker | |
| Hosting | Render (free tier) | |
| Frontend | Streamlit | optional, or just use `/docs` |

## Try it

Open the [live Swagger docs](https://codebase-archaeologist-wnhf.onrender.com/docs):

**1. Ingest a repo**
```json
POST /ingest
{
  "repo_path_or_url": "https://github.com/<owner>/<repo>",
  "max_commits": 150
}
```

**2. Ask about it**
```json
POST /query
{
  "question": "Why does this feature exist?"
}
```

## Run it locally

```bash
git clone https://github.com/priyanshi0275/codebase-archaeologist.git
cd codebase-archaeologist
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add your free Groq API key: https://console.groq.com/keys
uvicorn app.main:app --reload
```

Optional Streamlit UI:
```bash
streamlit run frontend/streamlit_app.py
```

## LoRA fine-tuning (separate artifact)

I also trained a [LoRA adapter](https://huggingface.co/priyanshi0275/codebase-archaeologist-lora) (`Qwen2.5-0.5B-Instruct` base) on commit messages pooled across six of my repos, so it learns a general, terse, engineer-style tone rather than one team's specific voice. Trained in Google Colab (free GPU), published to the Hugging Face Hub.

This is intentionally **not** wired into the live API — actually serving it live needs more RAM than Render's free tier provides. It's a standalone artifact demonstrating the fine-tuning pipeline: `lora/train_lora.py` (single or multi-repo), `lora/style_rewriter.py` (local inference). See `requirements-train.txt`.

## Known limitations

- Diff context is truncated per commit to keep the index small and fast on a free host — very large repos lose some detail.
- "Dependency risk" is inferred from commit-history patterns, not real static/AST analysis — a natural v2 extension.
- Free-tier hosting cold-starts after ~15 min idle.
- LoRA styling exists as a trained, published artifact but isn't called by the live API (see above).

## Project structure

```
codebase-archaeologist/
├── app/
│   ├── main.py           # FastAPI routes
│   ├── agents.py         # CrewAI agents + tasks
│   ├── ingest.py         # git history → chunks → Chroma
│   ├── vectorstore.py    # Chroma client wrapper
│   ├── style_service.py  # optional live LoRA hook (off by default)
│   └── config.py
├── lora/
│   ├── train_lora.py     # Colab training script
│   └── style_rewriter.py # local LoRA inference
├── frontend/
│   └── streamlit_app.py
├── Dockerfile
├── requirements.txt
└── requirements-train.txt
```

---

Built by [Priyanshi Mishra](https://github.com/priyanshi0275)
