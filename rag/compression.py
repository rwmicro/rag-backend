"""
Context Compression Module
Reduces context size while preserving information
"""

from typing import List, Tuple, Optional
import numpy as np
from loguru import logger
import tiktoken

from .chunking import Chunk


class ContextCompressor:
    """
    Compresses retrieved context to fit within token limits
    while preserving the most relevant information
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        tokenizer_name: str = "cl100k_base",
    ):
        """
        Initialize context compressor

        Args:
            max_tokens: Maximum tokens in compressed context
            tokenizer_name: Tokenizer for token counting
        """
        self.max_tokens = max_tokens

        # Load tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            logger.warning("Failed to load tiktoken, using character approximation")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate: ~4 chars per token
            return len(text) // 4

    def compress(
        self,
        chunks_with_scores: List[Tuple[Chunk, float]],
        query: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Compress context to fit within token limit

        Args:
            chunks_with_scores: Retrieved chunks with scores
            query: Optional query for query-aware compression

        Returns:
            Compressed list of chunks
        """
        if not chunks_with_scores:
            return []

        # Calculate current token count
        total_tokens = sum(
            self.count_tokens(chunk.content)
            for chunk, _ in chunks_with_scores
        )

        logger.info(f"Initial context: {total_tokens} tokens from {len(chunks_with_scores)} chunks")

        # If already within limit, return as-is
        if total_tokens <= self.max_tokens:
            logger.info("Context within limits, no compression needed")
            return chunks_with_scores

        # Strategy 1: Remove lowest scoring chunks
        compressed = []
        current_tokens = 0

        # Sort by score (already sorted, but ensure)
        sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)

        for chunk, score in sorted_chunks:
            chunk_tokens = self.count_tokens(chunk.content)

            if current_tokens + chunk_tokens <= self.max_tokens:
                compressed.append((chunk, score))
                current_tokens += chunk_tokens
            else:
                # Try to fit partial chunk if space available
                remaining_tokens = self.max_tokens - current_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    # Truncate chunk
                    truncated_content = self._truncate_to_tokens(
                        chunk.content,
                        remaining_tokens
                    )
                    if truncated_content:
                        truncated_chunk = Chunk(
                            chunk_id=chunk.chunk_id + "_truncated",
                            content=truncated_content,
                            metadata={**chunk.metadata, "truncated": True},
                        )
                        compressed.append((truncated_chunk, score))
                break

        logger.info(
            f"Compressed: {len(chunks_with_scores)} → {len(compressed)} chunks "
            f"({total_tokens} → {current_tokens} tokens)"
        )

        return compressed

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit
        Tries to end at sentence boundary
        """
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text

            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)
        else:
            # Character-based approximation
            max_chars = max_tokens * 4
            truncated_text = text[:max_chars]

        # Try to end at sentence boundary
        last_period = truncated_text.rfind(".")
        if last_period > len(truncated_text) * 0.7:  # Keep at least 70%
            truncated_text = truncated_text[:last_period + 1]

        return truncated_text

    def extract_relevant_sentences(
        self,
        chunks_with_scores: List[Tuple[Chunk, float]],
        query: str,
        max_sentences_per_chunk: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        Extract only the most relevant sentences from each chunk

        Args:
            chunks_with_scores: Retrieved chunks
            query: Search query
            max_sentences_per_chunk: Max sentences to keep per chunk

        Returns:
            Chunks with extracted sentences
        """
        import re

        compressed = []

        for chunk, score in chunks_with_scores:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', chunk.content)

            # Score sentences by keyword overlap with query
            query_words = set(query.lower().split())
            sentence_scores = []

            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words)
                sentence_scores.append((sent, overlap))

            # Sort by score and take top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [
                sent for sent, _ in sentence_scores[:max_sentences_per_chunk]
            ]

            # Reconstruct chunk with only top sentences
            compressed_content = " ".join(top_sentences)

            compressed_chunk = Chunk(
                chunk_id=chunk.chunk_id + "_compressed",
                content=compressed_content,
                metadata={**chunk.metadata, "compressed": True},
            )

            compressed.append((compressed_chunk, score))

        logger.info(f"Extracted relevant sentences from {len(compressed)} chunks")
        return compressed


class LLMCompressor(ContextCompressor):
    """
    LLM-based compression using a language model
    to summarize and extract key information
    """

    def __init__(
        self,
        llm_generator,
        max_tokens: int = 4000,
        **kwargs
    ):
        super().__init__(max_tokens=max_tokens, **kwargs)
        self.llm_generator = llm_generator

    def compress(
        self,
        chunks_with_scores: List[Tuple[Chunk, float]],
        query: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Compress using LLM summarization

        Args:
            chunks_with_scores: Retrieved chunks
            query: Search query for query-focused compression

        Returns:
            Compressed chunks
        """
        if not chunks_with_scores:
            return []

        # First, use token-based compression
        token_compressed = super().compress(chunks_with_scores, query)

        # Check if further LLM compression is needed
        total_tokens = sum(
            self.count_tokens(chunk.content)
            for chunk, _ in token_compressed
        )

        if total_tokens <= self.max_tokens * 0.8:
            return token_compressed

        # Apply LLM compression to each chunk
        llm_compressed = []

        for chunk, score in token_compressed:
            # Create compression prompt
            prompt = f"""Compress the following text while preserving key information relevant to: "{query}"

Text:
{chunk.content}

Compressed version (keep only essential information):"""

            try:
                compressed_content = self.llm_generator.generate(
                    prompt,
                    max_tokens=min(500, self.count_tokens(chunk.content) // 2)
                )

                compressed_chunk = Chunk(
                    chunk_id=chunk.chunk_id + "_llm_compressed",
                    content=compressed_content.strip(),
                    metadata={**chunk.metadata, "llm_compressed": True},
                )

                llm_compressed.append((compressed_chunk, score))

            except Exception as e:
                logger.warning(f"LLM compression failed for chunk: {e}")
                llm_compressed.append((chunk, score))

        logger.info(f"LLM compression: {total_tokens} → {sum(self.count_tokens(c.content) for c, _ in llm_compressed)} tokens")

        return llm_compressed
