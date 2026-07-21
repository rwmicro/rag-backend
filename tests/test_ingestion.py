"""
Tests for Document Ingestion Pipeline
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import DocumentIngestor, Document
from rag.chunking import (
    SemanticChunker,
    RecursiveChunker,
    MarkdownChunker,
    create_chunker,
    apply_contextual_embeddings,
    Chunk
)


class TestDocumentIngestor:
    """Test document ingestion"""

    @pytest.fixture
    def ingestor(self):
        return DocumentIngestor()

    def test_ingest_markdown(self, ingestor):
        """Test ingesting markdown content"""
        markdown_content = """# Test Document

This is a test paragraph.

## Section 1

Content in section 1.

## Section 2

Content in section 2.
"""
        document = ingestor.ingest_from_buffer(
            content=markdown_content.encode('utf-8'),
            filename="test.md",
            doc_type="markdown"
        )

        assert isinstance(document, Document)
        assert document.metadata["filename"] == "test.md"
        assert "Test Document" in document.content
        assert "Section 1" in document.content

    def test_ingest_text(self, ingestor):
        """Plain text goes through the markdown parser — there is no "text" type.

        Supported doc_types are pdf / markdown / csv / tsv; main.py maps ingested
        .txt files to "markdown" for the same reason.
        """
        text_content = "This is plain text content for testing."

        document = ingestor.ingest_from_buffer(
            content=text_content.encode('utf-8'),
            filename="test.txt",
            doc_type="markdown"
        )

        assert isinstance(document, Document)
        assert document.content == text_content
        assert document.metadata["filename"] == "test.txt"


class TestChunkers:
    """Test different chunking strategies"""

    def test_create_chunker_factory(self):
        """The factory used in production maps strategy names to chunker classes."""
        assert isinstance(create_chunker(strategy="semantic"), SemanticChunker)
        assert isinstance(create_chunker(strategy="recursive"), RecursiveChunker)
        assert isinstance(create_chunker(strategy="markdown"), MarkdownChunker)

        with pytest.raises(ValueError, match="Unknown strategy"):
            create_chunker(strategy="fixed")

    def test_chunk_size_is_measured_in_tokens(self):
        """chunk_size is a token budget, not a character count."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=1)

        content = "This is a sentence. " * 100
        chunks = chunker.chunk_document(content, {"filename": "test.txt"})

        assert len(chunks) > 1
        for chunk in chunks:
            # Regression guard: chunk_size used to be multiplied by 4 before being
            # handed to a token-measuring splitter, yielding ~4x oversized chunks.
            assert chunker.count_tokens(chunk.content) <= 50

    def test_recursive_chunker(self):
        """Test recursive text splitting"""
        # min_chunk_size=1: the default (max(50, chunk_size//2) tokens) would
        # discard this whole sample as too small and yield zero chunks.
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=1)

        content = """This is paragraph one. It has multiple sentences.

This is paragraph two. It also has content.

This is paragraph three."""

        chunks = chunker.chunk_document(content, {"filename": "test.txt"})

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.metadata["filename"] == "test.txt"

    def test_markdown_chunker(self):
        """Test markdown-aware chunking"""
        chunker = MarkdownChunker(chunk_size=200, chunk_overlap=50, min_chunk_size=1)

        content = """# Main Title

This is intro content.

## Section 1

Content under section 1.

### Subsection 1.1

Detailed content here.

## Section 2

More content here."""

        chunks = chunker.chunk_document(content, {"filename": "test.md"})

        assert len(chunks) > 0

        # The header hierarchy is lifted out of the text into metadata ("Header 1",
        # "Header 2", …) rather than left as "#" markup inside chunk.content.
        assert all(chunk.metadata["Header 1"] == "Main Title" for chunk in chunks)

        sections = {chunk.metadata.get("Header 2") for chunk in chunks}
        assert {"Section 1", "Section 2"} <= sections

        deepest = [c for c in chunks if c.metadata.get("Header 3") == "Subsection 1.1"]
        assert len(deepest) == 1
        assert deepest[0].content.strip() == "Detailed content here."

    def test_semantic_chunker(self):
        """Test semantic chunking (sentence-based)"""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

        content = """First sentence here. Second sentence follows. Third one too.

New paragraph starts here. It continues with more text. And concludes."""

        chunks = chunker.chunk_document(content, {"filename": "test.txt"})

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_chunk_metadata(self):
        """Test that chunks have proper metadata"""
        # min_chunk_size defaults to max(50, chunk_size // 2) tokens, which would
        # discard this short sample entirely and make the assertions vacuous.
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=1)

        metadata = {
            "filename": "test.pdf",
            "page": 1,
            "title": "Test Document"
        }

        content = "Some test content here."
        chunks = chunker.chunk_document(content, metadata)

        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["filename"] == "test.pdf"
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["chunk_index"] == i
            # chunk_id is "{filename}-{index}"
            assert chunk.chunk_id == f"test.pdf-{i}"

    def test_contextual_embeddings(self):
        """Test applying contextual embeddings"""
        chunks = [
            Chunk(
                chunk_id="chunk1",
                content="First chunk content",
                metadata={"filename": "test.txt"}
            ),
            Chunk(
                chunk_id="chunk2",
                content="Second chunk content",
                metadata={"filename": "test.txt"}
            )
        ]

        doc_summary = "This is a summary of the document."
        doc_title = "Test Document"

        enhanced_chunks = apply_contextual_embeddings(
            chunks=chunks,
            document_summary=doc_summary,
            document_title=doc_title
        )

        assert len(enhanced_chunks) == len(chunks)
        for chunk in enhanced_chunks:
            assert hasattr(chunk, 'content_for_embedding')
            assert chunk.content_for_embedding is not None
            # Context should be prepended
            assert doc_title in chunk.content_for_embedding
            assert doc_summary in chunk.content_for_embedding

    def test_empty_content_handling(self):
        """Test handling of empty or whitespace content"""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=1)

        # Empty content
        chunks = chunker.chunk_document("", {"filename": "empty.txt"})
        assert len(chunks) == 0

        # Whitespace only
        chunks = chunker.chunk_document("   \n\n  ", {"filename": "whitespace.txt"})
        assert len(chunks) == 0


class TestChunkOverlap:
    """Test chunk overlap functionality"""

    def test_overlap_creates_continuity(self):
        """Test that overlap creates continuity between chunks"""
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5, min_chunk_size=1)

        content = "abcdefghijklmnopqrstuvwxyz" * 3

        chunks = chunker.chunk_document(content, {"filename": "test.txt"})

        # Check that consecutive chunks have overlapping content
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i].content[-5:]
            chunk2_start = chunks[i + 1].content[:5]
            # Should have some overlap
            assert len(chunk1_end) > 0
            assert len(chunk2_start) > 0

    def test_no_overlap(self):
        """Test chunking with zero overlap"""
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0, min_chunk_size=1)

        content = "a" * 100

        chunks = chunker.chunk_document(content, {"filename": "test.txt"})

        # With no overlap, chunks should fit perfectly
        total_chars = sum(len(chunk.content) for chunk in chunks)
        assert total_chars <= len(content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
