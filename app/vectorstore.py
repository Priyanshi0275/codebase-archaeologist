"""
Thin wrapper around ChromaDB. Stores commit-history chunks so the agents
can retrieve them later.

Uses Chroma's built-in ONNX embedding model (all-MiniLM-L6-v2, ~80MB,
runs via onnxruntime) instead of sentence-transformers' torch-backed
version — same model, but without pulling in a ~2GB torch install and
the RAM that comes with loading it. This matters a lot on a free-tier
host with a hard memory ceiling.

Everything (Chroma client, embedding function) is created lazily on
first use rather than at import time, so the FastAPI server can bind
its port and pass a host's startup health check immediately, instead
of doing heavy work before it's even listening.
"""
import chromadb
from chromadb.utils import embedding_functions
from app.config import CHROMA_DB_PATH

COLLECTION_NAME = "commit_history"

_client = None
_embedding_fn = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        # Chroma's default: ONNX MiniLM-L6-v2, no torch required.
        _embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_fn


def get_collection(reset: bool = False):
    client = _get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_fn(),
    )


def add_chunks(ids: list[str], texts: list[str], metadatas: list[dict]):
    collection = get_collection()
    # Chroma has a per-call batch limit on some backends; chunk defensively.
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )


def query(text: str, n_results: int = 5) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    n_results = min(n_results, collection.count())
    results = collection.query(query_texts=[text], n_results=n_results)
    out = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        out.append({"text": doc, "metadata": meta, "distance": dist})
    return out


def collection_count() -> int:
    return get_collection().count()
