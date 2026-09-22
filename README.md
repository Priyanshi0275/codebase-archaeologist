# Legacy Codebase Archaeologist

A multi-agent system that investigates **why old code exists** and **what's risky about changing it**, grounded entirely in a repo's own commit history — not generic LLM guessing.

Three CrewAI agents work in sequence:
1. **Historian** — searches commit history (via Chroma) and explains why the code likely exists, citing commit SHAs.
2. **Dependency Risk Analyst** — flags what tends to break alongside it, rates risk low/medium/high.
3. **Refactor Advisor** — proposes a concrete next step, then a **LoRA adapter fine-tuned on that repo's own commit messages rewrites it live**, in the repo's own voice.

The LoRA rewrite happens in the deployed app itself, not as a side notebook — `/query` calls it as the last step of every request when an adapter is configured.

## Stack (100% free tier)

| Piece | Tool | Why free |
|---|---|---|
| Agent orchestration | CrewAI | open source |
| Chunking/splitting | LangChain text splitters | open source |
| Vector store | ChromaDB (local, embedded) | open source, no hosted DB needed |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | runs locally on CPU, no API cost |
| LLM for agent reasoning | Groq API (`llama-3.1-8b-instant`) | generous free tier, very fast |
| Git history parsing | GitPython | open source, reads local `.git` — no GitHub API rate limits |
| LoRA fine-tuning | HF `transformers` + `peft`, trained in Google Colab | Colab's free GPU tier |
| LoRA live inference | same `transformers`/`peft`, loaded in the running app (CPU, base model is 0.5B) | HF Spaces free CPU tier has enough RAM |
| Backend | FastAPI + Docker | — |
| Frontend | Streamlit | free to run/host |
| Hosting (live link) | **Hugging Face Spaces** (needs its 16GB free RAM for the LoRA model — Render's 512MB free tier is not enough) | free |

## 1. Local setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Get a free Groq API key at https://console.groq.com/keys and put it in `.env`.

Run the backend:
```bash
uvicorn app.main:app --reload
```

Run the frontend (separate terminal):
```bash
streamlit run frontend/streamlit_app.py
```

Check `/lora-status` any time to see whether the adapter is configured and loaded — until you do Part 2 below, it'll report `loaded: false` and `/query` will just return the raw (unstyled) refactor suggestion, which is expected.

## 2. Train the LoRA adapter (do this before deploying, if you want it live)

In a new Google Colab notebook (Runtime → Change runtime type → T4 GPU, free):

```bash
!git clone <the repo you want the archaeologist to analyze> /content/target_repo
!pip install -r requirements-train.txt
!python lora/train_lora.py --repo_path /content/target_repo --output_dir ./lora-adapter
```

This trains a small adapter (`Qwen2.5-0.5B-Instruct` base) on that repo's own commit messages.

**Publish the adapter to the Hugging Face Hub** (free, this is how the deployed app will load it):
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-hf-username/codebase-archaeologist-lora", exist_ok=True)
api.upload_folder(folder_path="./lora-adapter", repo_id="your-hf-username/codebase-archaeologist-lora")
```
(You'll need a free HF account + an access token: https://huggingface.co/settings/tokens — run `huggingface-cli login` in Colab first, or pass `token=` to the calls above.)

## 3. Deploy — Hugging Face Spaces (this is the one that makes LoRA actually work live)

1. Push this project to a GitHub repo.
2. Go to https://huggingface.co/new-space → name it → **SDK: Docker** → visibility public → Create.
3. Connect it to your GitHub repo (Space Settings → repository), or upload the folder directly.
4. Space **Settings → Variables and secrets**, add:
   - `GROQ_API_KEY` = your Groq key
   - `ENABLE_LORA_STYLE` = `true`
   - `LORA_ADAPTER_SOURCE` = `your-hf-username/codebase-archaeologist-lora` (the repo id from step 2)
5. The Space builds automatically (Dockerfile already targets port 7860). First build takes longer than usual — it's installing `torch`/`transformers`/`peft` too.
6. Your live link: `https://huggingface.co/spaces/<you>/codebase-archaeologist`

**Verify LoRA is actually live:**
```
https://<your-space-url>/lora-status
```
Should show `"loaded": true`. First call after a cold start downloads the adapter from the Hub, so it may take 10-20s the very first time.

## 4. Test it end to end

Use the Space's `/docs` Swagger page (`https://<your-space-url>/docs`):

- `POST /ingest`:
```json
{ "repo_path_or_url": "https://github.com/priyanshi0275/<your-repo>", "max_commits": 150 }
```
- `POST /query`:
```json
{ "question": "Why does the retry logic in payment.py exist?" }
```
Look at both `refactor_drafter` (raw) and `refactor_drafter_styled` (LoRA-rewritten) in the response — that side-by-side is a genuinely good thing to screenshot for your resume/portfolio.

## 5. (Optional) Frontend

Deploy `frontend/streamlit_app.py` on https://streamlit.io/cloud, with secret:
```toml
API_URL = "https://<your-space-url>"
```
Or just demo through `/docs` — no extra deploy needed.

## What to say about this project in an interview

- "Each agent is grounded by retrieval, not just prompted — the Historian has to cite a commit SHA before making a claim."
- "The refactor suggestion is rewritten live by a LoRA adapter I fine-tuned on that repo's own commit history, so the tone actually matches the team's — that adapter is loaded from the Hugging Face Hub at runtime, not baked into the image."
- "It's applying agentic AI to developer tooling — the direction the industry (Cursor, Devin, Copilot Workspace) is actually moving — rather than another customer-support chatbot."

## Known limitations (be upfront about these)

- Diff-based context is truncated per commit to keep the index small and free-tier friendly — very large repos will lose some detail.
- Dependency risk is inferred from commit-history patterns, not real static analysis (no AST/tree-sitter) — a natural "v2" extension.
- LoRA adapter quality depends heavily on how many commits the target repo has — very small repos (<50 commits) won't give it much to learn from.
- HF Spaces free tier can cold-start after inactivity; hit `/health` before a live demo.
