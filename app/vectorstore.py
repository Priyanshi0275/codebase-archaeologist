"""
Thin wrapper around ChromaDB. Stores commit-history chunks so the agents
can retrieve them later. Uses a local sentence-transformers model for
embeddings (all-MiniLM-L6-v2) — free, runs on CPU, no API calls needed.
"""
import chromadb
from chromadb.utils import embedding_functions
from app.config import CHROMA_DB_PATH

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

COLLECTION_NAME = "commit_history"


def get_collection(reset: bool = False):
    if reset:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
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
