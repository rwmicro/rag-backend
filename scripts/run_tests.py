#!/usr/bin/env python3
"""
Comprehensive Test Suite for RAG Pipeline
Tests all components before deployment
"""

import sys
import os
from pathlib import Path
import tempfile
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def test_imports():
    """Test that all modules can be imported"""
    console.print("\n[bold]Test 1: Module Imports[/bold]")

    modules = [
        "config.settings",
        "rag.ingest",
        "rag.chunking",
        "rag.embeddings",
        "rag.vectordb",
        "rag.retrieval",
        "rag.compression",
        "rag.generation",
        "rag.cache",
    ]

    results = []
    for module in modules:
        try:
            __import__(module)
            results.append((module, "✓", "green"))
        except Exception as e:
            results.append((module, f"✗ {str(e)[:50]}", "red"))

    table = Table()
    table.add_column("Module", style="cyan")
    table.add_column("Status", style="white")

    for module, status, color in results:
        table.add_row(module, f"[{color}]{status}[/{color}]")

    console.print(table)

    failures = [r for r in results if r[2] == "red"]
    if failures:
        console.print(f"\n[red]✗ {len(failures)} import failures[/red]")
        return False
    else:
        console.print("\n[green]✓ All imports successful[/green]")
        return True


def test_document_ingestion():
    """Test document parsing"""
    console.print("\n[bold]Test 2: Document Ingestion[/bold]")

    from rag.ingest import DocumentIngestor, parse_markdown

    # Create temporary markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test Document\n\n## Section 1\n\nContent here.\n")
        temp_file = f.name

    try:
        # Test markdown parsing
        doc = parse_markdown(temp_file)

        assert doc.content, "Content should not be empty"
        assert doc.metadata['file_type'] == 'markdown'
        assert 'title' in doc.metadata

        console.print("[green]✓ Markdown parsing works[/green]")

        # Test ingestor
        ingestor = DocumentIngestor()
        doc2 = ingestor.ingest_file(temp_file)

        assert doc2.content == doc.content

        console.print("[green]✓ Document ingestor works[/green]")
        return True

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return False

    finally:
        os.unlink(temp_file)


def test_chunking():
    """Test chunking strategies"""
    console.print("\n[bold]Test 3: Chunking[/bold]")

    from rag.chunking import create_chunker

    text = """
    This is a test document. It has multiple sentences.

    This is a second paragraph. It also has multiple sentences.
    Each sentence contains some words.

    This is a third paragraph for testing purposes.
    """ * 10  # Make it longer

    metadata = {
        "filename": "test.txt",
        "source": "test",
        "file_type": "md",
    }

    strategies = ["semantic", "recursive"]
    results = []

    for strategy in strategies:
        try:
            chunker = create_chunker(
                strategy=strategy,
                chunk_size=100,
                chunk_overlap=20,
            )

            chunks = chunker.chunk_document(text, metadata)

            assert len(chunks) > 0, f"Should create chunks for {strategy}"
            assert all(c.content for c in chunks), "Chunks should have content"
            assert all(c.chunk_id for c in chunks), "Chunks should have IDs"

            results.append((strategy, len(chunks), "✓", "green"))

        except Exception as e:
            results.append((strategy, 0, f"✗ {str(e)[:50]}", "red"))

    table = Table()
    table.add_column("Strategy", style="cyan")
    table.add_column("Chunks", style="white")
    table.add_column("Status", style="white")

    for strategy, count, status, color in results:
        table.add_row(strategy, str(count), f"[{color}]{status}[/{color}]")

    console.print(table)

    failures = [r for r in results if r[3] == "red"]
    if failures:
        console.print(f"\n[red]✗ {len(failures)} chunking failures[/red]")
        return False
    else:
        console.print("\n[green]✓ All chunking strategies work[/green]")
        return True


def test_embeddings():
    """Test embedding generation"""
    console.print("\n[bold]Test 4: Embeddings[/bold]")

    try:
        from rag.embeddings import create_embedding_model

        console.print("Loading embedding model (this may take a minute)...")

        model = create_embedding_model(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller for testing
            device="cpu",
        )

        # Test single encoding
        text = "This is a test sentence."
        embedding = model.encode_single(text)

        assert len(embedding) > 0, "Embedding should have dimension"
        assert embedding.shape[0] == model.dimension

        console.print(f"[green]✓ Single embedding: dim={len(embedding)}[/green]")

        # Test batch encoding
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = model.encode(texts)

        assert len(embeddings) == 3
        assert all(len(e) == model.dimension for e in embeddings)

        console.print(f"[green]✓ Batch embeddings: {len(embeddings)} texts[/green]")

        # Test similarity
        sim = model.similarity(embeddings[0], embeddings[1])
        assert 0 <= sim <= 1

        console.print(f"[green]✓ Similarity computation: {sim:.3f}[/green]")

        return True

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return False


def test_vector_store():
    """Test vector store operations"""
    console.print("\n[bold]Test 5: Vector Store[/bold]")

    try:
        from rag.vectordb import create_vector_store
        from rag.chunking import Chunk
        import numpy as np

        # Create temporary vector store
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use LanceDB for testing
            vector_store = create_vector_store(
                store_type="lancedb",
                uri=f"{tmpdir}/test_lancedb",
                table_name="test",
                dimension=384,
            )

            # Create test chunks
            chunks = [
                Chunk(
                    chunk_id=f"test-{i}",
                    content=f"Test content {i}",
                    metadata={"filename": "test.txt", "chunk_index": i},
                    embedding=np.random.rand(384).tolist(),
                )
                for i in range(5)
            ]

            # Test add
            vector_store.add_chunks(chunks)
            console.print("[green]✓ Add chunks works[/green]")

            # Test search
            query_embedding = np.random.rand(384)
            results = vector_store.search(query_embedding, top_k=3)

            assert len(results) <= 3
            assert all(isinstance(r[0], Chunk) for r in results)
            assert all(isinstance(r[1], float) for r in results)

            console.print(f"[green]✓ Search works: {len(results)} results[/green]")

            # Test stats
            stats = vector_store.get_stats()
            assert "total_chunks" in stats

            console.print(f"[green]✓ Stats: {stats['total_chunks']} chunks[/green]")

            return True

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def test_cache():
    """Test caching functionality"""
    console.print("\n[bold]Test 6: Caching[/bold]")

    try:
        from rag.cache import DiskCache, EmbeddingCache

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test disk cache
            cache = DiskCache(cache_dir=tmpdir)

            cache.set("test_key", {"value": 123})
            result = cache.get("test_key")

            assert result == {"value": 123}

            console.print("[green]✓ Disk cache works[/green]")

            # Test embedding cache
            emb_cache = EmbeddingCache(cache)

            emb_cache.set("test text", "test_model", [1.0, 2.0, 3.0])
            cached_emb = emb_cache.get("test text", "test_model")

            assert cached_emb == [1.0, 2.0, 3.0]

            console.print("[green]✓ Embedding cache works[/green]")

            return True

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return False


def main():
    """Run all tests"""
    console.print(Panel.fit(
        "[bold cyan]RAG Pipeline Test Suite[/bold cyan]",
        border_style="cyan"
    ))

    tests = [
        ("Module Imports", test_imports),
        ("Document Ingestion", test_document_ingestion),
        ("Chunking", test_chunking),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("Caching", test_cache),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "✓ PASS" if result else "✗ FAIL", "green" if result else "red"))
        except Exception as e:
            console.print(f"\n[red]Exception in {name}: {e}[/red]")
            results.append((name, f"✗ ERROR", "red"))

    # Summary
    console.print("\n[bold]Test Summary[/bold]\n")

    summary_table = Table()
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Result", style="white")

    for name, result, color in results:
        summary_table.add_row(name, f"[{color}]{result}[/{color}]")

    console.print(summary_table)

    # Overall result
    passed = sum(1 for r in results if r[2] == "green")
    total = len(results)

    console.print()
    if passed == total:
        console.print(f"[bold green]✅ All tests passed! ({passed}/{total})[/bold green]")
        sys.exit(0)
    else:
        console.print(f"[bold red]❌ Some tests failed ({passed}/{total} passed)[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
