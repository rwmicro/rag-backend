"""
Advanced Chunking Module
Implements semantic and recursive chunking strategies
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from abc import ABC, abstractmethod
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import tiktoken
from loguru import logger


@dataclass
class Chunk:
    """Represents a document chunk"""

    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    embedding: Optional[List[float]] = None
    content_for_embedding: Optional[str] = None  # Contextual version for embedding


class BaseChunker(ABC):
    """Base class for chunking strategies"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: Optional[int] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Set min_chunk_size to 50% of chunk_size if not specified
        self.min_chunk_size = min_chunk_size if min_chunk_size is not None else max(50, chunk_size // 2)

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
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

    @abstractmethod
    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Chunk a document into smaller pieces"""
        pass


class SemanticChunker(BaseChunker):
    """
    Semantic chunking that preserves meaning and structure
    Uses sentence boundaries and paragraph coherence
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sentence boundary regex
        self.sentence_endings = re.compile(r'([.!?])\s+')

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Split on sentence endings but keep the punctuation
        sentences = self.sentence_endings.split(text)

        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])

        # Filter out empty strings
        result = [s.strip() for s in result if s.strip()]

        # If no sentences were found (no punctuation), treat entire text as one sentence
        if not result and text.strip():
            result = [text.strip()]

        return result

    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk document semantically by sentence boundaries

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            List of Chunk objects
        """
        # Check for empty content
        if not content or not content.strip():
            logger.warning("Empty content provided to chunker")
            return []

        # Split into paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        logger.debug(f"Split content into {len(paragraphs)} paragraphs")

        if not paragraphs:
            logger.warning("No paragraphs found after splitting content")
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            sentences = self._split_into_sentences(para)

            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)

                # If adding this sentence exceeds chunk size and we have content
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    if self.count_tokens(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)

                    # Start new chunk with overlap
                    # Keep last few sentences for context
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_chunk):
                        s_tokens = self.count_tokens(s)
                        if overlap_tokens + s_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break

                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens

                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_tokens = self.count_tokens(chunk_text)
            logger.debug(f"Final chunk: {chunk_tokens} tokens, min required: {self.min_chunk_size}")
            # Fix: If we haven't created any chunks yet, accept this one regardless of size
            # Otherwise, enforce min_chunk_size
            if not chunks or chunk_tokens >= self.min_chunk_size:
                chunks.append(chunk_text)
                logger.debug(f"Added final chunk ({chunk_tokens} tokens)")
            else:
                logger.warning(f"Final chunk ({chunk_tokens} tokens) rejected because it's below min_chunk_size ({self.min_chunk_size}) and other chunks exist")
        else:
            logger.warning("No content in current_chunk at end of processing")

        if not chunks:
            logger.warning(f"No chunks created from content. Content length: {len(content)}, paragraphs: {len(paragraphs)}")

        # Create Chunk objects
        result = []
        for idx, chunk_content in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "chunk_tokens": self.count_tokens(chunk_content),
            }

            chunk_id = f"{metadata.get('filename', 'unknown')}-{idx}"

            result.append(Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))

        logger.info(f"Created {len(result)} semantic chunks")
        return result


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking using LangChain's RecursiveCharacterTextSplitter
    Respects document structure (paragraphs, sentences)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,  # Approximate char count from tokens
            chunk_overlap=self.chunk_overlap * 4,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk document recursively

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            List of Chunk objects
        """
        # Use LangChain splitter
        splits = self.splitter.split_text(content)

        # Filter out too-small chunks
        chunks = [s for s in splits if self.count_tokens(s) >= self.min_chunk_size]

        # Create Chunk objects
        result = []
        for idx, chunk_content in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "chunk_tokens": self.count_tokens(chunk_content),
            }

            chunk_id = f"{metadata.get('filename', 'unknown')}-{idx}"

            result.append(Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))

        logger.info(f"Created {len(result)} recursive chunks")
        return result


class MarkdownChunker(BaseChunker):
    """
    Markdown-aware chunking that preserves headers and structure
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        # Secondary splitter for large sections
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,
            chunk_overlap=self.chunk_overlap * 4,
            length_function=self.count_tokens,
        )

    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk Markdown document by headers

        Args:
            content: Markdown content
            metadata: Document metadata

        Returns:
            List of Chunk objects
        """
        # Split by headers first
        md_chunks = self.md_splitter.split_text(content)

        # Further split large sections
        all_chunks = []
        for md_chunk in md_chunks:
            chunk_text = md_chunk.page_content
            chunk_metadata = md_chunk.metadata

            # Check if chunk is too large
            if self.count_tokens(chunk_text) > self.chunk_size:
                # Split further
                sub_chunks = self.text_splitter.split_text(chunk_text)
                for sub_chunk in sub_chunks:
                    all_chunks.append({
                        "content": sub_chunk,
                        "header_metadata": chunk_metadata
                    })
            else:
                all_chunks.append({
                    "content": chunk_text,
                    "header_metadata": chunk_metadata
                })

        # Filter and create Chunk objects
        result = []
        for idx, chunk_data in enumerate(all_chunks):
            chunk_content = chunk_data["content"]

            if self.count_tokens(chunk_content) < self.min_chunk_size:
                continue

            chunk_metadata = {
                **metadata,
                **chunk_data.get("header_metadata", {}),
                "chunk_index": idx,
                "total_chunks": len(all_chunks),
                "chunk_tokens": self.count_tokens(chunk_content),
            }

            chunk_id = f"{metadata.get('filename', 'unknown')}-{idx}"

            result.append(Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))

        logger.info(f"Created {len(result)} markdown chunks")
        return result


def create_chunker(
    strategy: str = "semantic",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: Optional[int] = None,
) -> BaseChunker:
    """
    Factory function to create a chunker

    Args:
        strategy: Chunking strategy ("semantic", "recursive", "markdown")
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_chunk_size: Minimum chunk size in tokens

    Returns:
        Chunker instance
    """
    chunkers = {
        "semantic": SemanticChunker,
        "recursive": RecursiveChunker,
        "markdown": MarkdownChunker,
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(chunkers.keys())}")

    return chunkers[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )


def apply_contextual_embeddings(
    chunks: List[Chunk],
    document_summary: str,
    document_title: Optional[str] = None,
) -> List[Chunk]:
    """
    Apply contextual embedding enrichment to chunks

    Prepends document context to each chunk for better embedding quality:
    "Document: {title}. Context: {summary}. Section: {headers}.\n\n{content}"

    Args:
        chunks: List of chunks to enrich
        document_summary: Document summary (2-3 sentences)
        document_title: Optional document title

    Returns:
        Chunks with content_for_embedding populated
    """
    for chunk in chunks:
        # Build contextual prefix
        context_parts = []

        # Add document title
        if document_title:
            context_parts.append(f"Document: {document_title}")

        # Add document summary
        if document_summary:
            context_parts.append(f"Context: {document_summary}")

        # Add section/header information if available
        headers = []
        for key in ["Header 1", "Header 2", "Header 3"]:
            if key in chunk.metadata:
                headers.append(chunk.metadata[key])

        if headers:
            header_hierarchy = " > ".join(headers)
            context_parts.append(f"Section: {header_hierarchy}")

        # Combine context with content
        if context_parts:
            context_prefix = ". ".join(context_parts) + ".\n\n"
            chunk.content_for_embedding = context_prefix + chunk.content
        else:
            chunk.content_for_embedding = chunk.content

        logger.debug(f"Applied contextual embedding to chunk {chunk.chunk_id}")

    logger.info(f"Applied contextual embeddings to {len(chunks)} chunks")
    return chunks


def create_parent_child_chunks(
    chunks: List[Chunk],
    parent_chunk_size: int = 2048,
    parent_chunk_overlap: int = 512,
) -> Tuple[List[Chunk], List[Chunk]]:
    """
    Create parent-child chunk relationships for better retrieval

    Strategy:
    - Child chunks (small, ~512 tokens): Used for precise retrieval
    - Parent chunks (large, ~2048 tokens): Returned for richer context

    Args:
        chunks: Original child chunks
        parent_chunk_size: Target size for parent chunks in tokens
        parent_chunk_overlap: Overlap between parent chunks

    Returns:
        Tuple of (child_chunks_with_parent_refs, parent_chunks)
    """
    if not chunks:
        return [], []

    # Sort chunks by index to ensure correct ordering
    sorted_chunks = sorted(chunks, key=lambda c: c.metadata.get("chunk_index", 0))

    # Group chunks into parents based on combined token count
    parent_chunks = []
    current_parent_content = []
    current_parent_token_count = 0
    current_parent_children = []
    parent_index = 0

    # Get tokenizer from first chunk's chunker
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        count_tokens = lambda text: len(tokenizer.encode(text))
    except Exception:
        # Fallback: approximate
        count_tokens = lambda text: len(text) // 4

    for child_chunk in sorted_chunks:
        chunk_tokens = count_tokens(child_chunk.content)

        # If adding this chunk exceeds parent size, create parent
        if current_parent_token_count + chunk_tokens > parent_chunk_size and current_parent_children:
            # Create parent chunk
            parent_content = "\n\n".join(current_parent_content)
            parent_chunk_id = f"{chunks[0].metadata.get('filename', 'unknown')}-parent-{parent_index}"

            parent_chunk = Chunk(
                content=parent_content,
                metadata={
                    **chunks[0].metadata,
                    "chunk_index": parent_index,
                    "chunk_type": "parent",
                    "child_chunk_ids": [c.chunk_id for c in current_parent_children],
                    "chunk_tokens": current_parent_token_count,
                },
                chunk_id=parent_chunk_id
            )
            parent_chunks.append(parent_chunk)

            # Link children to parent
            for child in current_parent_children:
                child.metadata["parent_chunk_id"] = parent_chunk_id

            # Start new parent with overlap
            # Keep last few chunks for overlap
            overlap_chunks = []
            overlap_tokens = 0
            for c in reversed(current_parent_children):
                c_tokens = count_tokens(c.content)
                if overlap_tokens + c_tokens <= parent_chunk_overlap:
                    overlap_chunks.insert(0, c)
                    overlap_tokens += c_tokens
                else:
                    break

            current_parent_content = [c.content for c in overlap_chunks]
            current_parent_token_count = overlap_tokens
            current_parent_children = overlap_chunks.copy()
            parent_index += 1

        # Add current child to parent
        current_parent_content.append(child_chunk.content)
        current_parent_token_count += chunk_tokens
        current_parent_children.append(child_chunk)

    # Create final parent chunk
    if current_parent_children:
        parent_content = "\n\n".join(current_parent_content)
        parent_chunk_id = f"{chunks[0].metadata.get('filename', 'unknown')}-parent-{parent_index}"

        parent_chunk = Chunk(
            content=parent_content,
            metadata={
                **chunks[0].metadata,
                "chunk_index": parent_index,
                "chunk_type": "parent",
                "child_chunk_ids": [c.chunk_id for c in current_parent_children],
                "chunk_tokens": current_parent_token_count,
            },
            chunk_id=parent_chunk_id
        )
        parent_chunks.append(parent_chunk)

        # Link children to parent
        for child in current_parent_children:
            child.metadata["parent_chunk_id"] = parent_chunk_id

    logger.info(f"Created {len(parent_chunks)} parent chunks from {len(chunks)} child chunks")

    return chunks, parent_chunks


# =============================================================================
# TABLE-AWARE CHUNKING
# =============================================================================

@dataclass
class TableRegion:
    """Represents a table detected in content"""
    start_idx: int
    end_idx: int
    content: str
    table_type: str  # "markdown", "html", "plain_text", "pdf_extracted"
    confidence: float = 1.0
    headers: Optional[List[str]] = None


class TableAwareChunker:
    """
    Detects and preserves tables as atomic units during chunking.
    Tables are never split mid-row - they remain as single chunks or
    are split by rows with headers attached.
    """

    def __init__(
        self,
        base_chunker: BaseChunker,
        max_table_chunk_size: int = 2000,
        preserve_headers: bool = True
    ):
        self.base_chunker = base_chunker
        self.max_table_chunk_size = max_table_chunk_size
        self.preserve_headers = preserve_headers

    def detect_markdown_tables(self, content: str) -> List[TableRegion]:
        """
        Detect markdown tables (pipe-delimited).

        Markdown table format:
        | Column 1 | Column 2 |
        |----------|----------|
        | Value 1  | Value 2  |
        """
        tables = []

        # Pattern for markdown tables
        # Look for lines with pipes, including separator row
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check if line looks like a table row (has pipes)
            if '|' in line and line.count('|') >= 2:
                # Found potential table start
                table_start_line = i
                table_start_idx = sum(len(lines[j]) + 1 for j in range(i))

                # Collect all consecutive table rows
                table_lines = []
                headers = None

                while i < len(lines) and '|' in lines[i]:
                    current_line = lines[i].strip()

                    # Check for separator row (|---|---|)
                    if re.match(r'^\|[\s\-:]+\|[\s\-:|]+$', current_line):
                        # Previous line was headers
                        if table_lines:
                            headers = [
                                cell.strip()
                                for cell in table_lines[-1].split('|')[1:-1]
                            ]
                    table_lines.append(current_line)
                    i += 1

                # Minimum 2 rows for a table (header + separator, or 2 data rows)
                if len(table_lines) >= 2:
                    table_content = '\n'.join(table_lines)
                    table_end_idx = table_start_idx + len(table_content)

                    tables.append(TableRegion(
                        start_idx=table_start_idx,
                        end_idx=table_end_idx,
                        content=table_content,
                        table_type="markdown",
                        headers=headers
                    ))

                    logger.debug(f"Detected markdown table: {len(table_lines)} rows, "
                               f"headers: {headers[:3] if headers else None}")
            else:
                i += 1

        return tables

    def detect_html_tables(self, content: str) -> List[TableRegion]:
        """Detect HTML <table> blocks."""
        tables = []

        # Find all <table>...</table> blocks
        pattern = r'<table[^>]*>(.*?)</table>'
        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
            table_content = match.group(0)
            start_idx = match.start()
            end_idx = match.end()

            # Extract headers from <th> tags
            headers = re.findall(r'<th[^>]*>(.*?)</th>', table_content, re.IGNORECASE)
            headers = [re.sub(r'<[^>]+>', '', h).strip() for h in headers]

            tables.append(TableRegion(
                start_idx=start_idx,
                end_idx=end_idx,
                content=table_content,
                table_type="html",
                headers=headers if headers else None
            ))

            logger.debug(f"Detected HTML table at {start_idx}-{end_idx}, "
                       f"headers: {headers[:3] if headers else None}")

        return tables

    def detect_plain_text_tables(self, content: str) -> List[TableRegion]:
        """
        Detect aligned columnar data in plain text.
        Uses whitespace patterns to identify columns.
        """
        tables = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for multiple spaces or tabs (column separators)
            if re.search(r'(\s{2,}|\t)', line) and len(line.strip()) > 20:
                # Potential table row
                table_start_line = i
                table_start_idx = sum(len(lines[j]) + 1 for j in range(i))

                table_lines = []

                # Collect consecutive lines with similar spacing patterns
                while i < len(lines):
                    current_line = lines[i]

                    if re.search(r'(\s{2,}|\t)', current_line) or current_line.strip() == '':
                        table_lines.append(current_line)
                        i += 1
                    else:
                        break

                # Minimum 3 rows for plain text table
                if len([l for l in table_lines if l.strip()]) >= 3:
                    table_content = '\n'.join(table_lines)
                    table_end_idx = table_start_idx + len(table_content)

                    tables.append(TableRegion(
                        start_idx=table_start_idx,
                        end_idx=table_end_idx,
                        content=table_content,
                        table_type="plain_text",
                        confidence=0.7  # Lower confidence for plain text
                    ))

                    logger.debug(f"Detected plain text table: {len(table_lines)} rows")
            else:
                i += 1

        return tables

    def detect_tables(self, content: str, content_type: str = "auto") -> List[TableRegion]:
        """
        Detect table regions in content.

        Args:
            content: Text content
            content_type: "markdown", "html", "plain_text", or "auto"

        Returns:
            List of TableRegion objects
        """
        all_tables = []

        if content_type in ["auto", "markdown"]:
            all_tables.extend(self.detect_markdown_tables(content))

        if content_type in ["auto", "html"]:
            all_tables.extend(self.detect_html_tables(content))

        if content_type in ["auto", "plain_text"]:
            all_tables.extend(self.detect_plain_text_tables(content))

        # Sort by start index
        all_tables.sort(key=lambda t: t.start_idx)

        # Remove overlapping tables (keep highest confidence)
        filtered_tables = []
        for table in all_tables:
            # Check for overlap with existing tables
            overlaps = False
            for existing in filtered_tables:
                if (table.start_idx < existing.end_idx and
                    table.end_idx > existing.start_idx):
                    overlaps = True
                    # Keep the one with higher confidence
                    if table.confidence > existing.confidence:
                        filtered_tables.remove(existing)
                        filtered_tables.append(table)
                    break

            if not overlaps:
                filtered_tables.append(table)

        logger.info(f"Detected {len(filtered_tables)} tables in content")
        return filtered_tables

    def split_large_table(self, table: TableRegion) -> List[TableRegion]:
        """
        Split large tables by rows, keeping headers attached to each part.

        Args:
            table: TableRegion to split

        Returns:
            List of smaller TableRegion objects
        """
        tokens = self.base_chunker.count_tokens(table.content)

        if tokens <= self.max_table_chunk_size:
            # Table is small enough, return as-is
            return [table]

        logger.info(f"Splitting large table ({tokens} tokens) into smaller parts")

        # Split by rows
        rows = table.content.split('\n')

        # Identify header rows
        header_rows = []
        if table.table_type == "markdown":
            # First row is usually header, second is separator
            if len(rows) >= 2:
                header_rows = rows[:2]
                data_rows = rows[2:]
            else:
                data_rows = rows
        elif table.table_type == "html":
            # Extract <thead> if present
            thead_match = re.search(r'<thead[^>]*>(.*?)</thead>', table.content, re.DOTALL | re.IGNORECASE)
            if thead_match:
                header_rows = [thead_match.group(0)]
                # Get tbody content
                tbody_match = re.search(r'<tbody[^>]*>(.*?)</tbody>', table.content, re.DOTALL | re.IGNORECASE)
                if tbody_match:
                    data_rows = tbody_match.group(1).split('\n')
                else:
                    data_rows = rows
            else:
                data_rows = rows
        else:
            # Plain text: first row might be header
            if rows:
                header_rows = [rows[0]]
                data_rows = rows[1:]
            else:
                data_rows = []

        # Split data rows into chunks
        table_parts = []
        current_rows = []
        current_tokens = 0

        if self.preserve_headers and header_rows:
            header_text = '\n'.join(header_rows)
            header_tokens = self.base_chunker.count_tokens(header_text)
        else:
            header_text = ""
            header_tokens = 0

        for row in data_rows:
            row_tokens = self.base_chunker.count_tokens(row)

            if (current_tokens + row_tokens + header_tokens) > self.max_table_chunk_size and current_rows:
                # Create chunk with current rows
                if header_text:
                    chunk_content = header_text + '\n' + '\n'.join(current_rows)
                else:
                    chunk_content = '\n'.join(current_rows)

                table_parts.append(TableRegion(
                    start_idx=0,  # Will be adjusted later
                    end_idx=0,
                    content=chunk_content,
                    table_type=table.table_type,
                    headers=table.headers
                ))

                current_rows = []
                current_tokens = 0

            current_rows.append(row)
            current_tokens += row_tokens

        # Add remaining rows
        if current_rows:
            if header_text:
                chunk_content = header_text + '\n' + '\n'.join(current_rows)
            else:
                chunk_content = '\n'.join(current_rows)

            table_parts.append(TableRegion(
                start_idx=0,
                end_idx=0,
                content=chunk_content,
                table_type=table.table_type,
                headers=table.headers
            ))

        logger.info(f"Split table into {len(table_parts)} parts")
        return table_parts

    def chunk_with_tables(
        self,
        content: str,
        metadata: Dict[str, Any],
        content_type: str = "auto"
    ) -> List[Chunk]:
        """
        Chunk content while preserving table integrity.

        Algorithm:
        1. Detect all table regions
        2. Split content into: [text_before_table1, table1, text_between, table2, ...]
        3. Chunk non-table regions with base_chunker
        4. Keep tables as atomic chunks (or split by rows if too large)
        5. Mark table chunks with metadata

        Args:
            content: Document content
            metadata: Document metadata
            content_type: Content type hint

        Returns:
            List of Chunk objects
        """
        # Detect tables
        tables = self.detect_tables(content, content_type)

        if not tables:
            # No tables found, use base chunker
            return self.base_chunker.chunk_document(content, metadata)

        # Split content by tables
        chunks = []
        last_idx = 0
        chunk_counter = 0

        for table_idx, table in enumerate(tables):
            # Chunk text before this table
            if table.start_idx > last_idx:
                text_before = content[last_idx:table.start_idx].strip()
                if text_before:
                    text_chunks = self.base_chunker.chunk_document(text_before, metadata)
                    chunks.extend(text_chunks)
                    chunk_counter += len(text_chunks)

            # Handle table
            table_parts = self.split_large_table(table)

            for part_idx, table_part in enumerate(table_parts):
                chunk_id = f"{metadata.get('filename', 'doc')}_{chunk_counter:04d}"
                chunk_counter += 1

                table_metadata = {
                    **metadata,
                    "is_table": True,
                    "table_type": table_part.table_type,
                    "table_headers": table_part.headers
                }

                if len(table_parts) > 1:
                    table_metadata["table_part"] = f"{part_idx + 1}/{len(table_parts)}"
                    table_metadata["original_table_id"] = f"table_{table_idx}"

                chunk = Chunk(
                    chunk_id=chunk_id,
                    content=table_part.content,
                    metadata=table_metadata
                )
                chunks.append(chunk)

            last_idx = table.end_idx

        # Chunk remaining text after last table
        if last_idx < len(content):
            text_after = content[last_idx:].strip()
            if text_after:
                text_chunks = self.base_chunker.chunk_document(text_after, metadata)
                chunks.extend(text_chunks)

        logger.info(f"Created {len(chunks)} chunks ({len(tables)} tables, "
                   f"{len(chunks) - len(tables)} text chunks)")

        return chunks

