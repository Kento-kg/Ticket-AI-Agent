import json
import logging
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from agent.config import cfg

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    """Get or create the persistent Chroma collection. Cached for process lifetime."""
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=cfg.embedding_model)
    return client.get_or_create_collection(
        name=cfg.collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(batch_size: int = 1000) -> None:
    """Index dataset.json into Chroma. Idempotent: skips if collection already populated."""
    coll = _get_collection()
    if coll.count() > 0:
        logger.info(
            f"Collection '{cfg.collection_name}' already has {coll.count()} docs. Skipping."
        )
        return

    logger.info(f"Loading dataset from {cfg.dataset_path}")
    with cfg.dataset_path.open() as f:
        data = json.load(f)

    logger.info(f"Indexing {len(data)} tickets into Chroma...")
    for i in range(0, len(data), batch_size):
        chunk = data[i : i + batch_size]
        coll.add(
            ids=[f"ticket_{j}" for j in range(i, i + len(chunk))],
            documents=[t["text"] for t in chunk],
            metadatas=[
                {"category": t["category"], "urgency": t["urgency"], "team": t["team"]}
                for t in chunk
            ],
        )
        logger.info(f"  Indexed {i + len(chunk)}/{len(data)}")
    logger.info(f"Done. Collection has {coll.count()} docs.")


def search(text: str, k: int | None = None, category: str | None = None) -> list[dict]:
    """Top-k semantic search. Filters by category via Chroma's native metadata filter."""
    coll = _get_collection()
    res = coll.query(
        query_texts=[text],
        n_results=k or cfg.k_retrieval,
        where={"category": category} if category else None,
    )
    return [
        {
            "text": doc,
            "category": meta["category"],
            "urgency": meta["urgency"],
            "team": meta["team"],
            "score": 1.0 - dist,
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        query = sys.argv[1] if len(sys.argv) > 1 else "I cannot log in to my account"
        for r in search(query, k=3):
            print(f"[{r['score']:.3f}] [{r['category']}/{r['urgency']}] {r['text'][:120]}")
