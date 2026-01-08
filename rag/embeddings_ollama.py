"""
Ollama Embedding Support
Allows using Ollama embedding models as an alternative to SentenceTransformers
"""

from typing import List, Optional
import numpy as np
import httpx
from loguru import logger

from config.settings import settings


class OllamaEmbedding:
    """
    Ollama embedding model wrapper
    Compatible with the EmbeddingModel interface
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize Ollama embedding model

        Args:
            model_name: Ollama model name (e.g., "nomic-embed-text", "mxbai-embed-large")
            base_url: Ollama API base URL
            normalize: Whether to normalize embeddings
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.normalize = normalize
        self.batch_size = batch_size
        self.dimension = None  # Will be detected on first use

        logger.info(f"Initialized Ollama embedding: {model_name}")

    def encode(
        self,
        texts: List[str] | str,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts to embeddings using Ollama

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar (not implemented for Ollama)

        Returns:
            Numpy array of embeddings
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        embeddings_list = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            for text in batch:
                try:
                    embedding = self._embed_single(text)
                    embeddings_list.append(embedding)
                except Exception as e:
                    logger.error(f"Error embedding text: {e}")
                    # Return zero vector as fallback
                    if self.dimension:
                        embeddings_list.append(np.zeros(self.dimension))
                    else:
                        # Use default dimension
                        embeddings_list.append(np.zeros(768))

        embeddings = np.array(embeddings_list)

        # Normalize if requested
        if self.normalize and len(embeddings) > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms

        return embeddings

    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text using Ollama API"""
        url = f"{self.base_url}/api/embed"

        payload = {
            "model": self.model_name,
            "input": text,
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)

            if not response.is_success:
                error_text = response.text
                if "model" in error_text.lower() and "not found" in error_text.lower():
                    raise ValueError(
                        f"Ollama model '{self.model_name}' not found.\n"
                        f"Install it with: ollama pull {self.model_name}"
                    )
                raise RuntimeError(f"Ollama API error ({response.status_code}): {error_text}")

            data = response.json()

            # Extract embedding
            if "embedding" in data:
                embedding = np.array(data["embedding"], dtype=np.float32)
            elif "embeddings" in data and len(data["embeddings"]) > 0:
                embedding = np.array(data["embeddings"][0], dtype=np.float32)
            else:
                raise ValueError("No embedding in Ollama response")

            # Set dimension on first use
            if self.dimension is None:
                self.dimension = len(embedding)
                logger.info(f"Detected Ollama embedding dimension: {self.dimension}")

            return embedding

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text (convenience method)"""
        embeddings = self.encode([text])
        return embeddings[0]

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity"""
        # Normalize if not already
        if not self.normalize:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)

        return float(np.dot(embedding1, embedding2))

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute similarity between query and multiple corpus embeddings"""
        # Normalize if not already
        if not self.normalize:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            corpus_embeddings = corpus_embeddings / np.linalg.norm(
                corpus_embeddings, axis=1, keepdims=True
            )

        return np.dot(corpus_embeddings, query_embedding)


def create_ollama_embedding(
    model_name: str = "nomic-embed-text",
    base_url: Optional[str] = None,
    **kwargs
) -> OllamaEmbedding:
    """
    Factory function to create Ollama embedding model

    Args:
        model_name: Ollama model name
        base_url: Ollama API URL
        **kwargs: Additional arguments

    Returns:
        OllamaEmbedding instance
    """
    base_url = base_url or settings.LLM_BASE_URL

    return OllamaEmbedding(
        model_name=model_name,
        base_url=base_url,
        normalize=settings.NORMALIZE_EMBEDDINGS,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        **kwargs
    )
