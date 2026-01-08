"""
Document Summarization Module
Generates concise summaries for contextual chunk embeddings
"""

from typing import Optional, Dict
from loguru import logger
import hashlib
import os
import json

from config.settings import settings


class DocumentSummarizer:
    """
    Generates and caches document summaries for contextual embeddings
    """

    def __init__(self, llm_generator=None, cache_dir: Optional[str] = None):
        """
        Initialize document summarizer

        Args:
            llm_generator: LLM generator for creating summaries
            cache_dir: Directory to cache summaries (default: settings.CACHE_DIR/summaries)
        """
        self.llm_generator = llm_generator
        self.cache_dir = cache_dir or os.path.join(settings.CACHE_DIR, "summaries")
        os.makedirs(self.cache_dir, exist_ok=True)

        # In-memory cache
        self.cache: Dict[str, str] = {}

    def summarize(
        self,
        content: str,
        title: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a concise summary of the document

        Args:
            content: Document content
            title: Optional document title
            max_tokens: Maximum tokens for summary (uses settings if None)

        Returns:
            Document summary (2-3 sentences)
        """
        if max_tokens is None:
            max_tokens = settings.CONTEXT_SUMMARY_MAX_TOKENS

        # Create cache key from content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{content_hash}_{max_tokens}"

        # Check in-memory cache
        if cache_key in self.cache:
            logger.debug(f"Summary cache hit for {cache_key[:8]}...")
            return self.cache[cache_key]

        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    summary = f.read()
                self.cache[cache_key] = summary
                logger.debug(f"Summary loaded from disk cache for {cache_key[:8]}...")
                return summary
            except Exception as e:
                logger.warning(f"Failed to load summary from cache: {e}")

        # Generate new summary
        if self.llm_generator is None:
            # Fallback: use simple extraction (first N sentences)
            summary = self._extract_summary(content, title)
        else:
            # Use LLM for better summarization
            summary = self._generate_llm_summary(content, title, max_tokens)

        # Cache the summary
        self.cache[cache_key] = summary
        try:
            with open(cache_file, "w") as f:
                f.write(summary)
        except Exception as e:
            logger.warning(f"Failed to write summary to cache: {e}")

        logger.debug(f"Generated summary for document: {summary[:50]}...")
        return summary

    def _extract_summary(self, content: str, title: Optional[str] = None) -> str:
        """
        Simple extraction-based summarization (fallback)

        Args:
            content: Document content
            title: Optional title

        Returns:
            Extracted summary
        """
        # Split into sentences
        sentences = []
        for sent in content.split(". "):
            sent = sent.strip()
            if sent and not sent.endswith("."):
                sent += "."
            if sent:
                sentences.append(sent)

        # Take first 2-3 sentences
        if title:
            summary_parts = [f"{title}."]
        else:
            summary_parts = []

        # Add first few sentences up to max_tokens
        for sent in sentences[:3]:
            summary_parts.append(sent)
            # Simple token estimate (4 chars per token)
            if len(" ".join(summary_parts)) > settings.CONTEXT_SUMMARY_MAX_TOKENS * 4:
                break

        summary = " ".join(summary_parts)
        return summary[:500]  # Hard limit

    def _generate_llm_summary(
        self,
        content: str,
        title: Optional[str] = None,
        max_tokens: int = 100,
    ) -> str:
        """
        Generate summary using LLM

        Args:
            content: Document content
            title: Optional title
            max_tokens: Maximum tokens for summary

        Returns:
            Generated summary
        """
        # Truncate content if too long (to fit in prompt)
        max_content_chars = 4000  # Approx 1000 tokens
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "..."

        title_part = f"\nTitle: {title}" if title else ""

        prompt = f"""Summarize the following document in 2-3 concise sentences. Focus on the main topic and key points.{title_part}

Document:
{content}

Summary (2-3 sentences):"""

        try:
            summary = self.llm_generator.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return summary.strip()
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, falling back to extraction")
            return self._extract_summary(content, title)

    def clear_cache(self):
        """Clear in-memory and disk cache"""
        self.cache.clear()

        # Clear disk cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".txt"):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Summary cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear summary cache: {e}")


# Global summarizer instance
_document_summarizer: Optional[DocumentSummarizer] = None


def get_document_summarizer(llm_generator=None) -> DocumentSummarizer:
    """Get or create the global document summarizer"""
    global _document_summarizer

    if _document_summarizer is None:
        _document_summarizer = DocumentSummarizer(llm_generator=llm_generator)

    return _document_summarizer
