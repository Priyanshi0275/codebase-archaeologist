from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.ingest import ingest_repo
from app.agents import run_investigation
from app.vectorstore import collection_count
from app.config import PORT
from app import style_service

app = FastAPI(title="Legacy Codebase Archaeologist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    repo_path_or_url: str
    max_commits: int = 150


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": collection_count()}


@app.get("/lora-status")
def lora_status():
    """Check whether the live LoRA adapter is configured and loaded."""
    return style_service.status()


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        return ingest_repo(req.repo_path_or_url, max_commits=req.max_commits)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
def query(req: QueryRequest):
    if collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No repo has been ingested yet. Call /ingest first.",
        )
    try:
        return run_investigation(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False)
