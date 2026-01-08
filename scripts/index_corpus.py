#!/usr/bin/env python3
"""
Corpus Indexing Script
Ingests all documents from corpus directory and builds vector index
"""

import sys
import os
from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings, ensure_directories
from rag import (
    DocumentIngestor,
    create_chunker,
    create_embedding_model,
    create_vector_store,
)

console = Console()


@click.command()
@click.option(
    "--corpus-dir",
    default=None,
    help=f"Corpus directory (default: {settings.CORPUS_DIR})",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Process subdirectories recursively",
)
@click.option(
    "--chunk-size",
    default=1000,
    help="Target chunk size in tokens",
)
@click.option(
    "--chunk-overlap",
    default=200,
    help="Overlap between chunks in tokens",
)
@click.option(
    "--chunking-strategy",
    type=click.Choice(["semantic", "recursive", "markdown"]),
    default="semantic",
    help="Chunking strategy",
)
@click.option(
    "--embedding-model",
    default=None,
    help=f"Embedding model (default: {settings.EMBEDDING_MODEL})",
)
@click.option(
    "--clear-existing/--keep-existing",
    default=False,
    help="Clear existing index before ingestion",
)
def main(
    corpus_dir: str,
    recursive: bool,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str,
    embedding_model: str,
    clear_existing: bool,
):
    """
    Index corpus documents and build vector database
    """
    console.print("\n[bold cyan]📚 RAG Corpus Indexer[/bold cyan]\n")

    # Use default corpus dir if not specified
    corpus_dir = corpus_dir or settings.CORPUS_DIR

    # Ensure directories exist
    ensure_directories()

    # Check if corpus directory exists
    if not Path(corpus_dir).exists():
        console.print(f"[bold red]❌ Error:[/bold red] Corpus directory not found: {corpus_dir}")
        console.print(f"\nCreate it with: [cyan]mkdir -p {corpus_dir}[/cyan]")
        console.print(f"Then add your PDF and Markdown files to this directory.")
        return

    # Display configuration
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Corpus Directory", corpus_dir)
    table.add_row("Recursive", str(recursive))
    table.add_row("Chunking Strategy", chunking_strategy)
    table.add_row("Chunk Size", f"{chunk_size} tokens")
    table.add_row("Chunk Overlap", f"{chunk_overlap} tokens")
    table.add_row("Embedding Model", embedding_model or settings.EMBEDDING_MODEL)
    table.add_row("Vector Store", settings.VECTOR_STORE_TYPE)
    table.add_row("Clear Existing", str(clear_existing))

    console.print(table)
    console.print()

    # Initialize components
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load embedding model
        task = progress.add_task("Loading embedding model...", total=None)
        emb_model = create_embedding_model(
            model_name=embedding_model,
        )
        progress.update(task, completed=True)

        # Initialize vector store
        task = progress.add_task("Initializing vector store...", total=None)
        vector_store = create_vector_store()
        progress.update(task, completed=True)

        # Clear existing if requested
        if clear_existing:
            task = progress.add_task("Clearing existing index...", total=None)
            # Note: Implementation depends on vector store type
            console.print("[yellow]⚠️  Clear existing not fully implemented for all stores[/yellow]")
            progress.update(task, completed=True)

    # Initialize ingestor
    ingestor = DocumentIngestor()

    # Ingest documents
    console.print("\n[bold]Step 1: Document Ingestion[/bold]")
    with console.status("[bold green]Reading documents...") as status:
        documents = ingestor.ingest_directory(
            directory=corpus_dir,
            recursive=recursive,
        )

    if not documents:
        console.print(f"[bold yellow]⚠️  No documents found in {corpus_dir}[/bold yellow]")
        console.print("\nSupported formats: .pdf, .md, .markdown")
        return

    console.print(f"[green]✓[/green] Loaded {len(documents)} documents")

    # Create chunker
    console.print("\n[bold]Step 2: Document Chunking[/bold]")
    chunker = create_chunker(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        chunk_task = progress.add_task("Chunking documents...", total=len(documents))

        for doc in documents:
            chunks = chunker.chunk_document(doc.content, doc.metadata)
            all_chunks.extend(chunks)
            progress.update(chunk_task, advance=1)

    console.print(f"[green]✓[/green] Created {len(all_chunks)} chunks from {len(documents)} documents")
    console.print(f"  Average: {len(all_chunks) / len(documents):.1f} chunks per document")

    # Generate embeddings
    console.print("\n[bold]Step 3: Embedding Generation[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        embed_task = progress.add_task("Generating embeddings...", total=len(all_chunks))

        # Process in batches
        batch_size = settings.EMBEDDING_BATCH_SIZE
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]

            # Generate embeddings
            embeddings = emb_model.encode(texts)

            # Assign to chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding.tolist()

            progress.update(embed_task, advance=len(batch))

    console.print(f"[green]✓[/green] Generated embeddings for {len(all_chunks)} chunks")

    # Add to vector store
    console.print("\n[bold]Step 4: Vector Store Indexing[/bold]")

    with console.status("[bold green]Adding chunks to vector store...") as status:
        vector_store.add_chunks(all_chunks)

    console.print(f"[green]✓[/green] Indexed {len(all_chunks)} chunks")

    # Display statistics
    console.print("\n[bold cyan]📊 Indexing Complete[/bold cyan]\n")

    stats_table = Table(title="Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green", justify="right")

    stats_table.add_row("Total Documents", str(len(documents)))
    stats_table.add_row("Total Chunks", str(len(all_chunks)))
    stats_table.add_row("Avg Chunks/Doc", f"{len(all_chunks) / len(documents):.1f}")

    # File type breakdown
    pdf_count = sum(1 for d in documents if d.doc_type == "pdf")
    md_count = len(documents) - pdf_count
    stats_table.add_row("PDF Files", str(pdf_count))
    stats_table.add_row("Markdown Files", str(md_count))

    console.print(stats_table)

    # Get vector store stats
    vs_stats = vector_store.get_stats()
    console.print(f"\n[bold]Vector Store:[/bold] {vs_stats.get('total_chunks', 0)} total chunks indexed")

    console.print("\n[bold green]✅ Indexing completed successfully![/bold green]")
    console.print("\nYou can now start the API server:")
    console.print("  [cyan]cd backend && python main.py[/cyan]\n")


if __name__ == "__main__":
    main()
