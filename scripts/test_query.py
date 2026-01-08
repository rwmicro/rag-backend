#!/usr/bin/env python3
"""
Query Testing Script
Test the RAG pipeline with sample queries
"""

import sys
import asyncio
from pathlib import Path
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from rag import (
    create_embedding_model,
    create_vector_store,
    HybridRetriever,
    Reranker,
    ContextCompressor,
    LLMGenerator,
)

console = Console()


@click.command()
@click.option(
    "--query",
    "-q",
    prompt="Enter your query",
    help="Search query",
)
@click.option(
    "--top-k",
    default=5,
    help="Number of results to retrieve",
)
@click.option(
    "--use-reranking/--no-reranking",
    default=True,
    help="Use reranking",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Stream the response",
)
def main(query: str, top_k: int, use_reranking: bool, stream: bool):
    """
    Test RAG query pipeline
    """
    console.print("\n[bold cyan]🔍 RAG Query Tester[/bold cyan]\n")

    # Initialize components
    with console.status("[bold green]Initializing components...") as status:
        # Embedding model
        console.print("Loading embedding model...")
        embedding_model = create_embedding_model()

        # Vector store
        console.print("Loading vector store...")
        vector_store = create_vector_store()

        # Retriever
        console.print("Initializing retriever...")
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
        )

        # Reranker (optional)
        reranker = None
        if use_reranking:
            console.print("Loading reranker...")
            try:
                reranker = Reranker()
            except Exception as e:
                console.print(f"[yellow]⚠️  Reranker not available: {e}[/yellow]")

        # LLM generator
        console.print("Initializing LLM...")
        llm = LLMGenerator(
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL,
        )

    console.print("[green]✓[/green] Components loaded\n")

    # Display query
    console.print(Panel(query, title="Query", border_style="cyan"))

    # Step 1: Retrieval
    console.print("\n[bold]Step 1: Retrieving relevant chunks[/bold]")
    with console.status("[bold green]Searching...") as status:
        chunks_with_scores = retriever.retrieve(
            query=query,
            top_k=top_k * 2 if use_reranking else top_k,
        )

    if not chunks_with_scores:
        console.print("[red]No relevant chunks found![/red]")
        return

    console.print(f"[green]✓[/green] Retrieved {len(chunks_with_scores)} chunks")

    # Step 2: Reranking
    if use_reranking and reranker:
        console.print("\n[bold]Step 2: Reranking results[/bold]")
        with console.status("[bold green]Reranking...") as status:
            chunks_with_scores = reranker.rerank(
                query=query,
                chunks_with_scores=chunks_with_scores,
                top_k=top_k,
            )
        console.print(f"[green]✓[/green] Reranked to top {len(chunks_with_scores)}")

    # Display sources
    console.print("\n[bold]Retrieved Sources:[/bold]\n")
    for idx, (chunk, score) in enumerate(chunks_with_scores[:top_k], 1):
        metadata = chunk.metadata
        source_info = f"**Source {idx}** (score: {score:.3f})"

        if metadata.get("filename"):
            source_info += f" - {metadata['filename']}"
        if metadata.get("page_range"):
            source_info += f" ({metadata['page_range']})"

        console.print(Panel(
            chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
            title=source_info,
            border_style="blue",
        ))

    # Step 3: Generation
    console.print("\n[bold]Step 3: Generating response[/bold]\n")

    if stream:
        # Streaming response
        console.print("[bold cyan]Response (streaming):[/bold cyan]\n")

        async def stream_response():
            full_response = ""
            async for chunk in llm.generate_rag_response_stream(
                query=query,
                chunks_with_scores=chunks_with_scores,
            ):
                console.print(chunk, end="")
                full_response += chunk

            console.print("\n")
            return full_response

        # Run async
        response = asyncio.run(stream_response())

    else:
        # Non-streaming response
        with console.status("[bold green]Generating...") as status:
            response = llm.generate_rag_response(
                query=query,
                chunks_with_scores=chunks_with_scores,
            )

        console.print(Panel(
            Markdown(response),
            title="Response",
            border_style="green",
        ))

    console.print("\n[bold green]✅ Query completed![/bold green]\n")


if __name__ == "__main__":
    main()
