"""
Cached Embedding Generation
Wrapper around embedding models with persistent caching
"""

from typing import List, Union
import numpy as np
from loguru import logger

from .embeddings import EmbeddingModel
from .cache import get_embedding_cache


class CachedEmbeddingModel:
    """
    Embedding model with automatic caching
    Wraps an EmbeddingModel and caches results to disk
    """

    def __init__(self, embedding_model: EmbeddingModel, enable_cache: bool = True):
        """
        Initialize cached embedding model

        Args:
            embedding_model: Base embedding model to wrap
            enable_cache: Whether to enable caching (default: True)
        """
        self.model = embedding_model
        self.enable_cache = enable_cache
        self.cache = get_embedding_cache() if enable_cache else None
        self.model_name = embedding_model.model_name

        if enable_cache:
            logger.info(f"Embedding cache enabled for model: {self.model_name}")

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts to embeddings with caching

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings (shape: [n_texts, dimension])
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        if not texts:
            return np.array([])

        # If caching is disabled, encode directly
        if not self.enable_cache or not self.cache:
            return self.model.encode(texts, show_progress=show_progress)

        # Check cache for each text
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []

        for idx, text in enumerate(texts):
            cached_embedding = self.cache.get(text, self.model_name)

            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                texts_to_encode.append(text)
                indices_to_encode.append(idx)

        # Encode missing texts
        if texts_to_encode:
            cache_hits = len(texts) - len(texts_to_encode)
            if cache_hits > 0:
                logger.debug(
                    f"Embedding cache: {cache_hits}/{len(texts)} hits, "
                    f"encoding {len(texts_to_encode)} new texts"
                )

            new_embeddings = self.model.encode(texts_to_encode, show_progress=show_progress)

            # Store in cache and update results
            for idx, text, embedding in zip(
                indices_to_encode, texts_to_encode, new_embeddings
            ):
                # Convert numpy array to list for caching
                embedding_list = embedding.tolist()
                self.cache.set(text, self.model_name, embedding_list)
                embeddings[idx] = embedding_list

        # Convert to numpy array
        result = np.array(embeddings)

        return result[0] if single_input else result

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.dimension

    def __getattr__(self, name):
        """Forward other attributes to the underlying model"""
        return getattr(self.model, name)


def create_cached_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5",
    enable_cache: bool = True,
    **kwargs
) -> CachedEmbeddingModel:
    """
    Factory function to create a cached embedding model

    Args:
        model_name: HuggingFace model name or path
        enable_cache: Whether to enable caching (default: True)
        **kwargs: Additional arguments passed to EmbeddingModel

    Returns:
        CachedEmbeddingModel instance
    """
    base_model = EmbeddingModel(model_name=model_name, **kwargs)
    return CachedEmbeddingModel(base_model, enable_cache=enable_cache)
