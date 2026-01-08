"""
Embedding Generation Module
Uses SentenceTransformers (BGE/E5) for state-of-the-art embeddings
"""

from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
from config.settings import settings


class EmbeddingModel:
    """
    Wrapper for embedding models with caching and batching
    Supports BGE, E5, and other SentenceTransformer models
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ("cuda", "cpu", or None for auto)
            normalize: Whether to normalize embeddings
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading embedding model: {model_name} on {device}")

        # Load model
        try:
            # Load model with FP16 optimization for RTX GPUs
            model_kwargs = {}
            if device == "cuda":
                # Enable FP16 (half precision) for 2-3x speedup on modern GPUs
                model_kwargs = {
                    "dtype": torch.float16,
                }
                logger.info("Enabling FP16 mixed precision for faster GPU inference")

            # Load model with optimizations
            self.model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=model_kwargs,
            )

            # Get embedding dimension
            self.dimension = self.model.get_sentence_embedding_dimension()

            logger.info(
                f"✓ Loaded model: {model_name} (dim={self.dimension}) on {device}"
            )

        except OSError as e:
            logger.error(f"Model file not found or inaccessible: {model_name}: {e}")
            raise
        except RuntimeError as e:
            logger.error(
                f"Runtime error loading model (CUDA/memory issue?): {model_name}: {e}"
            )
            raise
        except ValueError as e:
            logger.error(f"Invalid model configuration: {model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model {model_name}: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings (shape: [n_texts, dimension])
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        try:
            # Ensure model is on correct device (prevents CPU fallback)
            if self.device == "cuda":
                is_on_cuda = next(self.model.parameters()).is_cuda
                if not is_on_cuda:
                    logger.warning(f"⚠️  Model was on CPU, moving back to {self.device}")
                    self.model = self.model.to(self.device)
                else:
                    logger.debug(f"✓ Model confirmed on {self.device}")

            # Encode with batching and GPU optimization
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                device=self.device,  # Explicitly pass device to prevent CPU fallback
            )

            return embeddings

        except RuntimeError as e:
            logger.error(f"Runtime error during encoding (out of memory?): {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid input for encoding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error encoding texts: {e}")
            raise

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode a single text (convenience method)

        Args:
            text: Text to encode

        Returns:
            1D numpy array of embedding
        """
        embeddings = self.encode([text], is_query=is_query)
        return embeddings[0]

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        # Ensure normalized
        if not self.normalize:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)

        return float(np.dot(embedding1, embedding2))

    def batch_similarity(
        self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple corpus embeddings

        Args:
            query_embedding: Query embedding (1D array)
            corpus_embeddings: Corpus embeddings (2D array)

        Returns:
            Similarity scores (1D array)
        """
        # Ensure normalized
        if not self.normalize:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            corpus_embeddings = corpus_embeddings / np.linalg.norm(
                corpus_embeddings, axis=1, keepdims=True
            )

        # Compute dot product (cosine similarity for normalized vectors)
        similarities = np.dot(corpus_embeddings, query_embedding)
        return similarities


class BGEEmbedding(EmbeddingModel):
    """
    BGE (BAAI General Embedding) model wrapper
    Optimized for retrieval tasks
    """

    def __init__(
        self,
        model_variant: str = "large",  # "base", "large", "small"
        **kwargs,
    ):
        from .model_registry import resolve_model_shortcut

        # Resolve "large" -> "bge-large" -> "BAAI/bge-large-en-v1.5"
        if model_variant in ["base", "large", "small"]:
            shortcut = f"bge-{model_variant}"
            model_name = resolve_model_shortcut(shortcut, "embedding")
        else:
            # Allow direct model name or bge- shortcut
            model_name = resolve_model_shortcut(model_variant, "embedding")

        super().__init__(model_name=model_name, **kwargs)

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode with BGE instruction prefix for queries
        Note: For BGE, queries should be prefixed with "Represent this sentence for searching relevant passages: "
        But we'll keep it simple by default
        """
        return super().encode(texts, **kwargs)


class E5Embedding(EmbeddingModel):
    """
    E5 (EmbEddings from bidirEctional Encoder rEpresentations) model wrapper
    Requires "query: " and "passage: " prefixes for optimal performance
    """

    def __init__(
        self,
        model_variant: str = "large",  # "base", "large", "small"
        **kwargs,
    ):
        from .model_registry import resolve_model_shortcut

        # Resolve "large" -> "e5-large" -> "intfloat/e5-large-v2"
        if model_variant in ["base", "large", "small"]:
            shortcut = f"e5-{model_variant}"
            model_name = resolve_model_shortcut(shortcut, "embedding")
        else:
            # Allow direct model name or e5- shortcut
            model_name = resolve_model_shortcut(model_variant, "embedding")

        super().__init__(model_name=model_name, **kwargs)

    def encode(
        self, texts: Union[str, List[str]], is_query: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Encode with E5 prefixes: "query: " or "passage: "
        """
        prefix = "query: " if is_query else "passage: "

        if isinstance(texts, str):
            texts = [prefix + texts]
        else:
            texts = [prefix + t for t in texts]

        logger.debug(
            f"E5 Encoding ({'query' if is_query else 'doc'}): {texts[0][:50]}..."
        )

        return super().encode(texts, **kwargs)


class MultilingualEmbeddingModel(EmbeddingModel):
    """
    Multilingual Embedding Model
    Uses intfloat/multilingual-e5-large for cross-lingual retrieval

    Supports 100+ languages in a shared embedding space:
    - All Romance languages (Spanish, French, Italian, Portuguese, Romanian, etc.)
    - All Germanic languages (English, German, Dutch, Swedish, etc.)
    - All Slavic languages (Russian, Polish, Czech, Ukrainian, etc.)
    - CJK languages (Chinese, Japanese, Korean)
    - Indic languages (Hindi, Bengali, Tamil, Telugu, etc.)
    - Arabic, Hebrew, Persian, Turkish
    - And many more...

    Key Features:
    - Cross-lingual retrieval (query in one language, find docs in another)
    - Shared embedding space across all languages
    - E5 prefix support ("query: " and "passage: ")
    - 1024-dimensional embeddings
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Initialize multilingual E5 model

        Args:
            model_name: Multilingual model name (default: multilingual-e5-large)
            device: Device to use ("cuda", "cpu", or None for auto)
            normalize: Whether to normalize embeddings
            batch_size: Batch size for encoding
        """
        # Validate model is multilingual
        if "multilingual-e5" not in model_name.lower():
            logger.warning(
                f"Model '{model_name}' does not appear to be a multilingual E5 model. "
                f"For best multilingual performance, use 'intfloat/multilingual-e5-large'"
            )

        super().__init__(
            model_name=model_name,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
            **kwargs,
        )

        logger.info(f"Multilingual embedding model loaded: {model_name}")
        logger.info("Supports 100+ languages in shared embedding space")

    def encode(
        self,
        texts: Union[str, List[str]],
        is_query: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts with E5 prefix for multilingual retrieval

        E5 models require:
        - "query: " prefix for queries
        - "passage: " prefix for documents

        Args:
            texts: Single text or list of texts (in any supported language)
            is_query: Whether texts are queries (vs documents)
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings (shape: [n_texts, 1024])
        """
        # Add E5 prefix
        prefix = "query: " if is_query else "passage: "

        if isinstance(texts, str):
            prefixed_texts = [prefix + texts]
        else:
            prefixed_texts = [prefix + t for t in texts]

        logger.debug(
            f"Multilingual E5 encoding ({'query' if is_query else 'passage'}): "
            f"{prefixed_texts[0][:50]}..."
        )

        # Use parent's encode method
        return super().encode(
            prefixed_texts,
            show_progress=show_progress,
            is_query=False,  # Don't double-prefix
        )

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode a single text (convenience method)

        Args:
            text: Text to encode (in any supported language)
            is_query: Whether text is a query

        Returns:
            1D numpy array of embedding (1024 dimensions)
        """
        embeddings = self.encode([text], is_query=is_query)
        return embeddings[0]

    @property
    def supported_languages(self) -> List[str]:
        """
        Get list of well-supported languages

        Returns:
            List of ISO 639-1 language codes
        """
        return [
            # Major European languages
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "nl",
            "pl",
            "ru",
            "uk",
            "cs",
            "ro",
            "sv",
            "no",
            "da",
            "fi",
            "hu",
            "bg",
            "hr",
            "sr",
            "sk",
            "sl",
            "lt",
            "lv",
            "et",
            # CJK languages
            "zh",
            "ja",
            "ko",
            # Indic languages
            "hi",
            "bn",
            "te",
            "ta",
            "ur",
            "mr",
            "gu",
            "kn",
            "ml",
            "pa",
            # Middle Eastern languages
            "ar",
            "he",
            "fa",
            "tr",
            # Southeast Asian languages
            "th",
            "vi",
            "id",
            "ms",
            # Other languages
            "el",
            "ca",
            "eu",
            "gl",
            "cy",
            "ga",
            "mt",
            "sq",
            "mk",
            "ka",
            "hy",
            "az",
            "kk",
            "uz",
            "ky",
            "tg",
            "mn",
            "ne",
            "si",
            "km",
            "lo",
            "my",
            "am",
            "sw",
            "af",
            "zu",
            "xh",
            "st",
            "tn",
            "sn",
        ]


class HybridEmbeddingModel:
    """
    Hybrid Embedding Model - Fusion Vectorielle Avancée

    Combines multiple embedding approaches:
    1. Linguistic embeddings (BGE/E5) - semantic content
    2. Structural embeddings - document structure (headers, sections)
    3. Adaptive fusion based on query type

    Based on Stanford NLP 2025 research on Adaptive Embedding Fusion
    """

    def __init__(
        self,
        linguistic_model: EmbeddingModel,
        structural_weight: float = 0.3,
        device: Optional[str] = None,
    ):
        """
        Initialize hybrid embedding model

        Args:
            linguistic_model: Primary embedding model (BGE/E5)
            structural_weight: Weight for structural features (0-1)
            device: Device to use
        """
        self.linguistic_model = linguistic_model
        self.structural_weight = structural_weight
        self.linguistic_weight = 1.0 - structural_weight

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Handle lazy dimension detection (e.g., Ollama embeddings)
        self.dimension = getattr(linguistic_model, "dimension", None)

        logger.info(
            f"HybridEmbedding initialized (linguistic={self.linguistic_weight:.2f}, "
            f"structural={self.structural_weight:.2f})"
        )

    def _extract_structural_features(
        self, text: str, metadata: Optional[dict] = None
    ) -> np.ndarray:
        """
        Extract structural features from text

        Features:
        - Header presence/level
        - List/bullet points
        - Code blocks
        - Length
        - Capitalization patterns
        """
        features = []

        # Header features (Markdown)
        header_count = text.count("#")
        features.append(min(header_count / 10.0, 1.0))  # Normalize

        # List features
        list_count = text.count("\n- ") + text.count("\n* ") + text.count("\n1. ")
        features.append(min(list_count / 10.0, 1.0))

        # Code block features
        code_count = text.count("```") + text.count("`")
        features.append(min(code_count / 10.0, 1.0))

        # Length features
        length_score = min(len(text) / 1000.0, 1.0)
        features.append(length_score)

        # Capitalization (title-case indicates formal content)
        words = text.split()
        if words:
            cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
            features.append(cap_ratio)
        else:
            features.append(0.0)

        # Metadata features
        if metadata:
            # Has title
            features.append(1.0 if "title" in metadata else 0.0)
            # Has section
            features.append(1.0 if "section" in metadata else 0.0)
            # Page number (indicates position)
            if "page_number" in metadata:
                features.append(min(metadata["page_number"] / 100.0, 1.0))
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])

        # Pad to match linguistic dimension
        structural_vector = np.array(features, dtype=np.float32)

        # Lazily update dimension if not set (for Ollama embeddings)
        if self.dimension is None:
            self.dimension = getattr(self.linguistic_model, "dimension", None)

        # If dimension is still not available, use a default
        target_dimension = self.dimension if self.dimension else 768

        # Expand to match linguistic dimension
        # Repeat structural features to fill dimension
        repeat_factor = target_dimension // len(features) + 1
        structural_vector = np.tile(structural_vector, repeat_factor)[:target_dimension]

        # Normalize
        structural_vector = structural_vector / (
            np.linalg.norm(structural_vector) + 1e-8
        )

        return structural_vector

    def encode(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[Union[dict, List[dict]]] = None,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Encode with hybrid approach (linguistic + structural)

        Args:
            texts: Text(s) to encode
            metadata: Optional metadata for structural features
            show_progress: Show progress bar

        Returns:
            Hybrid embeddings
        """
        # Handle single text
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
            if metadata is not None and not isinstance(metadata, list):
                metadata = [metadata]

        # Handle empty input
        if not texts:
            return np.array([])

        # Get linguistic embeddings
        linguistic_embs = self.linguistic_model.encode(
            texts, show_progress=show_progress, is_query=is_query
        )

        # Handle empty embeddings (when texts is empty or encoding fails)
        if linguistic_embs.size == 0:
            return np.array([])

        # Update dimension from linguistic model after first encoding (for lazy detection)
        if self.dimension is None:
            self.dimension = getattr(self.linguistic_model, "dimension", None)

        # Get structural embeddings
        structural_embs = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else None
            struct_emb = self._extract_structural_features(text, meta)
            structural_embs.append(struct_emb)

        structural_embs = np.array(structural_embs)

        # Fuse embeddings (weighted combination)
        hybrid_embs = (
            self.linguistic_weight * linguistic_embs
            + self.structural_weight * structural_embs
        )

        # Normalize fused embeddings only if we have data
        if hybrid_embs.ndim == 2 and hybrid_embs.shape[0] > 0:
            hybrid_embs = hybrid_embs / (
                np.linalg.norm(hybrid_embs, axis=1, keepdims=True) + 1e-8
            )
        elif hybrid_embs.ndim == 1 and hybrid_embs.size > 0:
            hybrid_embs = hybrid_embs / (np.linalg.norm(hybrid_embs) + 1e-8)

        if is_single and hybrid_embs.size > 0:
            return hybrid_embs[0]

        return hybrid_embs

    def encode_single(self, text: str, metadata: Optional[dict] = None) -> np.ndarray:
        """Encode single text"""
        return self.encode(text, metadata=metadata)


class AdaptiveEmbeddingFusion:
    """
    Adaptive Embedding Fusion (Stanford NLP 2025)

    Dynamically adjusts fusion weights based on query characteristics:
    - Factual queries -> Higher linguistic weight
    - Navigational queries -> Higher structural weight
    - Exploratory queries -> Balanced fusion
    """

    def __init__(
        self,
        linguistic_model: EmbeddingModel,
        device: Optional[str] = None,
    ):
        """Initialize adaptive fusion"""
        self.linguistic_model = linguistic_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Handle lazy dimension detection (e.g., Ollama embeddings)
        self.dimension = getattr(linguistic_model, "dimension", None)

        # Create hybrid models with different weights
        self.factual_model = HybridEmbeddingModel(
            linguistic_model,
            structural_weight=0.1,  # Low structural weight
            device=device,
        )

        self.navigational_model = HybridEmbeddingModel(
            linguistic_model,
            structural_weight=0.5,  # High structural weight
            device=device,
        )

        self.exploratory_model = HybridEmbeddingModel(
            linguistic_model,
            structural_weight=0.3,  # Balanced
            device=device,
        )

        logger.info("AdaptiveEmbeddingFusion initialized")

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type

        Types:
        - factual: "What is X?", "How does Y work?"
        - navigational: "Find section about...", "Show me documentation on..."
        - exploratory: "Tell me about...", "Explain..."
        """
        query_lower = query.lower()

        # Factual queries
        factual_keywords = [
            "what is",
            "how does",
            "why is",
            "when did",
            "who is",
            "define",
        ]
        if any(kw in query_lower for kw in factual_keywords):
            return "factual"

        # Navigational queries
        nav_keywords = ["find", "show", "section", "chapter", "page", "documentation"]
        if any(kw in query_lower for kw in nav_keywords):
            return "navigational"

        # Exploratory (default)
        return "exploratory"

    def encode(
        self,
        texts: Union[str, List[str]],
        query_context: Optional[str] = None,
        metadata: Optional[Union[dict, List[dict]]] = None,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Adaptive encoding based on query context

        Args:
            texts: Texts to encode
            query_context: Optional query to guide fusion strategy
            metadata: Optional metadata
            show_progress: Show progress

        Returns:
            Adaptively fused embeddings
        """
        # If no query context, use default (exploratory)
        if query_context:
            query_type = self._classify_query_type(query_context)
        else:
            query_type = "exploratory"

        # Select appropriate model
        if query_type == "factual":
            model = self.factual_model
            logger.debug("Using factual fusion (low structural weight)")
        elif query_type == "navigational":
            model = self.navigational_model
            logger.debug("Using navigational fusion (high structural weight)")
        else:
            model = self.exploratory_model
            logger.debug("Using exploratory fusion (balanced)")

        # Encode with selected model
        return model.encode(
            texts, metadata=metadata, show_progress=show_progress, is_query=is_query
        )

    def encode_single(
        self,
        text: str,
        query_context: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> np.ndarray:
        """Encode single text with adaptive fusion"""
        return self.encode(text, query_context=query_context, metadata=metadata)


def create_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,  # "ollama", "huggingface", or None for auto-detect
    use_ollama: bool = False,
    use_hybrid: bool = False,
    use_adaptive: bool = False,
    use_multilingual: bool = False,
    **kwargs,
):
    """
    Factory function to create embedding model with intelligent provider detection

    Args:
        model_name: Model name or None to use settings default
        device: Device to use or None for auto-detect
        provider: Explicit provider ("ollama", "huggingface") or None for auto-detect
        use_ollama: Use Ollama API instead of SentenceTransformers (deprecated, use provider)
        use_hybrid: Use hybrid embedding (linguistic + structural)
        use_adaptive: Use adaptive embedding fusion
        use_multilingual: Use multilingual E5 model for cross-lingual retrieval
        **kwargs: Additional arguments

    Returns:
        EmbeddingModel or enhanced variant

    Provider Detection Rules:
        1. If provider is specified explicitly, use it
        2. If model contains "/" -> HuggingFace (e.g., "BAAI/bge-large")
        3. If model contains ":" but no "/" -> Ollama (e.g., "qwen3:8b")
        4. If model starts with "bge-" or "e5-" -> HuggingFace shortcut
        5. If "multilingual" in name -> MultilingualEmbeddingModel
        6. Otherwise -> Try Ollama first (most flexible for local models)
    """
    model_name = model_name or settings.EMBEDDING_MODEL
    device = device or settings.EMBEDDING_DEVICE

    # Auto-detect if device is auto
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract hybrid/adaptive-specific parameters from kwargs
    # These should not be passed to base embedding models
    structural_weight = kwargs.pop("structural_weight", 0.3)

    # Check for multilingual model (auto-detect or explicit flag)
    is_multilingual = use_multilingual or "multilingual" in model_name.lower()

    # Intelligent provider detection
    # Priority: explicit provider > use_ollama flag > auto-detection
    if provider:
        provider = provider.lower()
        is_ollama_model = provider == "ollama"
        logger.info(f"Using explicit provider: {provider} for model {model_name}")
    elif use_ollama:
        is_ollama_model = True
        logger.info(f"Using Ollama provider (use_ollama=True) for model {model_name}")
    else:
        # Smart auto-detection based on model name format
        has_colon = ":" in model_name and "/" not in model_name  # Ollama: "model:tag"
        has_slash = "/" in model_name  # HuggingFace: "org/model"
        is_bge = model_name.startswith("bge-")  # HuggingFace shortcut
        is_e5 = model_name.startswith("e5-")  # HuggingFace shortcut

        if has_slash:
            # Definitively HuggingFace
            is_ollama_model = False
            logger.info(f"Auto-detected HuggingFace (contains '/'): {model_name}")
        elif has_colon:
            # Definitively Ollama (model:tag format)
            is_ollama_model = True
            logger.info(f"Auto-detected Ollama (model:tag format): {model_name}")
        elif is_bge or is_e5:
            # Known HuggingFace shortcuts
            is_ollama_model = False
            logger.info(f"Auto-detected HuggingFace (known prefix): {model_name}")
        else:
            # Ambiguous - default to Ollama (more flexible for any local model)
            is_ollama_model = True
            logger.info(f"Auto-detected as Ollama (no clear indicator): {model_name}")

    # Create base model based on type
    if is_multilingual:
        # Multilingual E5 model
        logger.info(f"Creating multilingual embedding model: {model_name}")
        base_model = MultilingualEmbeddingModel(
            model_name=model_name,
            device=device,
            normalize=settings.NORMALIZE_EMBEDDINGS,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            **kwargs,
        )
    elif is_ollama_model:
        from .embeddings_ollama import create_ollama_embedding

        base_model = create_ollama_embedding(model_name=model_name, **kwargs)
    # Check for BGE or E5 shortcuts (only for shortcuts, not full paths with "/")
    elif model_name.startswith("bge-") and "/" not in model_name:
        variant = model_name.split("-")[1]  # e.g., "bge-large" -> "large"
        base_model = BGEEmbedding(
            model_variant=variant,
            device=device,
            normalize=settings.NORMALIZE_EMBEDDINGS,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            **kwargs,
        )
    elif model_name.startswith("e5-") or (
        "e5" in model_name.lower() and "/" not in model_name
    ):
        # Handle E5 shortcuts like "e5-large" (but NOT full paths like "intfloat/multilingual-e5-large")
        variant = "large"  # Default
        if "base" in model_name:
            variant = "base"
        elif "small" in model_name:
            variant = "small"

        base_model = E5Embedding(
            model_variant=variant,
            device=device,
            normalize=settings.NORMALIZE_EMBEDDINGS,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            **kwargs,
        )
    else:
        # Generic model
        base_model = EmbeddingModel(
            model_name=model_name,
            device=device,
            normalize=settings.NORMALIZE_EMBEDDINGS,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            **kwargs,
        )

    # Wrap with hybrid or adaptive if requested
    if use_adaptive:
        logger.info("Using Adaptive Embedding Fusion")
        return AdaptiveEmbeddingFusion(base_model, device=device)
    elif use_hybrid:
        logger.info(f"Using Hybrid Embedding (structural_weight={structural_weight})")
        return HybridEmbeddingModel(
            base_model, structural_weight=structural_weight, device=device
        )
    else:
        return base_model
