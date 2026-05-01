import json
from functools import lru_cache
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from agents.config import cfg

@lru_cache(maxsize=1)
def get_collection() -> Collection:
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
    """Index dataset.json into Chroma (skips if collection already populated)."""
    coll = get_collection()
    if coll.count() > 0:
        print(
            f"Collection '{cfg.collection_name}' already has {coll.count()} docs. Skipping."
        )
        return

    print(f"Loading dataset from {cfg.dataset_path}")
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
