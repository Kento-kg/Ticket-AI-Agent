from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AgentConfig:
    hf_model_id: str = "kentokamg/ticket-triage-finetuned"
    embedding_model: str = "sentence-transformers/paraphrase-mpnet-base-v2"

    claude_main_model: str = "claude-sonnet-4-6"
    claude_haiku_model: str = "claude-haiku-4-5"

    max_length: int = 256
    k_retrieval: int = 3
    collection_name: str = "ticket_history"

    ood_threshold: float = 0.5

    chroma_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "chroma"
    )
    dataset_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "dataset.json"
    )


cfg = AgentConfig()
