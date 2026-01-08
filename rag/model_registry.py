"""
Model Registry
Centralized registry of supported models with metadata
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Model metadata"""
    name: str  # Full model name
    dimension: int  # Embedding dimension
    max_seq_length: int  # Maximum sequence length
    model_type: str  # "huggingface", "ollama", "openai"
    description: str = ""  # Human-readable description
    size_mb: Optional[int] = None  # Approximate model size


# Embedding Models Registry
EMBEDDING_MODELS: Dict[str, ModelInfo] = {
    # BGE Models (BAAI General Embedding)
    "bge-large": ModelInfo(
        name="BAAI/bge-large-en-v1.5",
        dimension=1024,
        max_seq_length=512,
        model_type="huggingface",
        description="BGE Large - High accuracy, general purpose",
        size_mb=1340,
    ),
    "bge-base": ModelInfo(
        name="BAAI/bge-base-en-v1.5",
        dimension=768,
        max_seq_length=512,
        model_type="huggingface",
        description="BGE Base - Balanced performance",
        size_mb=438,
    ),
    "bge-small": ModelInfo(
        name="BAAI/bge-small-en-v1.5",
        dimension=384,
        max_seq_length=512,
        model_type="huggingface",
        description="BGE Small - Fast, lightweight",
        size_mb=133,
    ),

    # E5 Models (Microsoft)
    "e5-large": ModelInfo(
        name="intfloat/e5-large-v2",
        dimension=1024,
        max_seq_length=512,
        model_type="huggingface",
        description="E5 Large - Excellent for long documents",
        size_mb=1340,
    ),
    "e5-base": ModelInfo(
        name="intfloat/e5-base-v2",
        dimension=768,
        max_seq_length=512,
        model_type="huggingface",
        description="E5 Base - Good general purpose",
        size_mb=438,
    ),
    "e5-small": ModelInfo(
        name="intfloat/e5-small-v2",
        dimension=384,
        max_seq_length=512,
        model_type="huggingface",
        description="E5 Small - Compact and fast",
        size_mb=133,
    ),

    # Nomic Models
    "nomic": ModelInfo(
        name="nomic-ai/nomic-embed-text-v1.5",
        dimension=768,
        max_seq_length=8192,
        model_type="huggingface",
        description="Nomic - Very long context (8K tokens)",
        size_mb=548,
    ),

    # GTE Models (Alibaba)
    "gte-large": ModelInfo(
        name="thenlper/gte-large",
        dimension=1024,
        max_seq_length=512,
        model_type="huggingface",
        description="GTE Large - Strong multilingual support",
        size_mb=670,
    ),
    "gte-base": ModelInfo(
        name="thenlper/gte-base",
        dimension=768,
        max_seq_length=512,
        model_type="huggingface",
        description="GTE Base - Good multilingual",
        size_mb=220,
    ),

    # Instructor Models
    "instructor-large": ModelInfo(
        name="hkunlp/instructor-large",
        dimension=768,
        max_seq_length=512,
        model_type="huggingface",
        description="Instructor Large - Instruction-tuned",
        size_mb=1340,
    ),
    "instructor-base": ModelInfo(
        name="hkunlp/instructor-base",
        dimension=768,
        max_seq_length=512,
        model_type="huggingface",
        description="Instructor Base - Instruction-tuned",
        size_mb=440,
    ),

    # MiniLM Models (Compact)
    "minilm-l6": ModelInfo(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        max_seq_length=256,
        model_type="huggingface",
        description="MiniLM - Very fast, small footprint",
        size_mb=80,
    ),
    "minilm-l12": ModelInfo(
        name="sentence-transformers/all-MiniLM-L12-v2",
        dimension=384,
        max_seq_length=256,
        model_type="huggingface",
        description="MiniLM L12 - Fast and efficient",
        size_mb=120,
    ),

    # Multilingual Models
    "e5-multilingual-large": ModelInfo(
        name="intfloat/multilingual-e5-large",
        dimension=1024,
        max_seq_length=512,
        model_type="huggingface",
        description="E5 Multilingual Large - 100+ languages, excellent cross-lingual",
        size_mb=2200,
    ),
    "e5-multilingual-base": ModelInfo(
        name="intfloat/multilingual-e5-base",
        dimension=768,
        max_seq_length=512,
        model_type="huggingface",
        description="E5 Multilingual Base - 100+ languages, balanced",
        size_mb=1100,
    ),
    "labse": ModelInfo(
        name="sentence-transformers/LaBSE",
        dimension=768,
        max_seq_length=256,
        model_type="huggingface",
        description="LaBSE - 109 languages, Google research",
        size_mb=1900,
    ),
    "multilingual-minilm": ModelInfo(
        name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimension=384,
        max_seq_length=128,
        model_type="huggingface",
        description="Multilingual MiniLM - 50+ languages, fast and lightweight",
        size_mb=470,
    ),
    "xlm-roberta-base": ModelInfo(
        name="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        dimension=768,
        max_seq_length=128,
        model_type="huggingface",
        description="XLM-RoBERTa - 100+ languages, strong multilingual",
        size_mb=1100,
    ),
}


# Reranker Models Registry
RERANKER_MODELS: Dict[str, ModelInfo] = {
    "bge-reranker-large": ModelInfo(
        name="BAAI/bge-reranker-large",
        dimension=1024,  # Not applicable for rerankers, but keeping for consistency
        max_seq_length=512,
        model_type="huggingface",
        description="BGE Reranker Large - Maximum accuracy",
        size_mb=1340,
    ),
    "bge-reranker-base": ModelInfo(
        name="BAAI/bge-reranker-base",
        dimension=768,
        max_seq_length=512,
        model_type="huggingface",
        description="BGE Reranker Base - Balanced performance",
        size_mb=438,
    ),
    "ms-marco-miniLM": ModelInfo(
        name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        dimension=384,
        max_seq_length=512,
        model_type="huggingface",
        description="MS MARCO MiniLM - Fast reranking",
        size_mb=80,
    ),
    "ms-marco-distilbert": ModelInfo(
        name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        dimension=384,
        max_seq_length=512,
        model_type="huggingface",
        description="MS MARCO TinyBERT - Very fast",
        size_mb=60,
    ),
}


# Common Ollama Embedding Models (for reference)
OLLAMA_EMBEDDING_MODELS = {
    "nomic-embed-text": ModelInfo(
        name="nomic-embed-text",
        dimension=768,
        max_seq_length=8192,
        model_type="ollama",
        description="Nomic via Ollama - Long context",
    ),
    "mxbai-embed-large": ModelInfo(
        name="mxbai-embed-large",
        dimension=1024,
        max_seq_length=512,
        model_type="ollama",
        description="MixedBread AI Large - High quality",
    ),
    "all-minilm": ModelInfo(
        name="all-minilm",
        dimension=384,
        max_seq_length=256,
        model_type="ollama",
        description="MiniLM via Ollama - Fast",
    ),
}


def resolve_model_shortcut(shortcut_or_name: str, model_type: str = "embedding") -> str:
    """
    Resolve a shortcut to full model name

    Args:
        shortcut_or_name: Shortcut (e.g., "bge-large") or full name
        model_type: "embedding" or "reranker"

    Returns:
        Full model name
    """
    registry = EMBEDDING_MODELS if model_type == "embedding" else RERANKER_MODELS

    if shortcut_or_name in registry:
        return registry[shortcut_or_name].name

    # Already a full name
    return shortcut_or_name


def get_model_info(shortcut_or_name: str, model_type: str = "embedding") -> Optional[ModelInfo]:
    """
    Get model info from shortcut or full name

    Args:
        shortcut_or_name: Shortcut or full model name
        model_type: "embedding" or "reranker"

    Returns:
        ModelInfo if found, None otherwise
    """
    registry = EMBEDDING_MODELS if model_type == "embedding" else RERANKER_MODELS

    # Check if it's a shortcut
    if shortcut_or_name in registry:
        return registry[shortcut_or_name]

    # Check if it's a full name in the registry
    for info in registry.values():
        if info.name == shortcut_or_name:
            return info

    # Check Ollama registry for embedding models
    if model_type == "embedding":
        if shortcut_or_name in OLLAMA_EMBEDDING_MODELS:
            return OLLAMA_EMBEDDING_MODELS[shortcut_or_name]

        for info in OLLAMA_EMBEDDING_MODELS.values():
            if info.name == shortcut_or_name:
                return info

    return None


def get_model_dimension(shortcut_or_name: str, model_type: str = "embedding") -> Optional[int]:
    """
    Get embedding dimension for a model

    Args:
        shortcut_or_name: Model shortcut or full name
        model_type: "embedding" or "reranker"

    Returns:
        Dimension if known, None otherwise
    """
    info = get_model_info(shortcut_or_name, model_type)
    return info.dimension if info else None


def validate_dimension_compatibility(
    existing_dimension: int,
    new_model: str,
    model_type: str = "embedding"
) -> tuple[bool, Optional[str]]:
    """
    Check if a new model's dimension is compatible with existing data

    Args:
        existing_dimension: Dimension of existing embeddings
        new_model: New model to validate
        model_type: "embedding" or "reranker"

    Returns:
        (is_compatible, error_message)
    """
    info = get_model_info(new_model, model_type)

    if info is None:
        # Unknown model - we can't validate dimension
        return True, None

    if info.dimension != existing_dimension:
        return False, (
            f"Dimension mismatch: Collection uses {existing_dimension}D embeddings, "
            f"but '{new_model}' produces {info.dimension}D embeddings. "
            f"Cannot mix different embedding dimensions in the same collection."
        )

    return True, None


def list_models_by_dimension(dimension: int, model_type: str = "embedding") -> list[str]:
    """
    List all models with a specific dimension

    Args:
        dimension: Target dimension
        model_type: "embedding" or "reranker"

    Returns:
        List of model shortcuts/names
    """
    registry = EMBEDDING_MODELS if model_type == "embedding" else RERANKER_MODELS

    compatible = []
    for shortcut, info in registry.items():
        if info.dimension == dimension:
            compatible.append(shortcut)

    return compatible


def to_dict(model_info: ModelInfo) -> Dict[str, Any]:
    """Convert ModelInfo to dictionary for JSON serialization"""
    return {
        "name": model_info.name,
        "dimension": model_info.dimension,
        "max_seq_length": model_info.max_seq_length,
        "model_type": model_info.model_type,
        "description": model_info.description,
        "size_mb": model_info.size_mb,
    }
