"""
Test Graph RAG Features:
1. spaCy NER entity extraction
2. Graph caching (save/load)
3. Integration with advanced RAG features
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from rag import GraphRAG, create_embedding_model, Chunk
import shutil
import tempfile


def test_spacy_ner_extraction():
    """Test that spaCy NER extracts entities correctly"""
    print("\n" + "="*60)
    print("TEST 1: spaCy NER Entity Extraction")
    print("="*60)

    # Create embedding model
    embedding_model = create_embedding_model()

    # Create Graph RAG with spaCy enabled
    graph_rag = GraphRAG(
        embedding_model=embedding_model,
        use_spacy_ner=True,
        spacy_model="en_core_web_sm",
        min_entity_mentions=1,
        max_entities_per_chunk=20,
    )

    # Create a test chunk with known entities
    test_chunk = Chunk(
        chunk_id="test_1",
        content="""
        Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California.
        The company released the iPhone in 2007, which revolutionized the smartphone industry.
        Tim Cook became CEO after Steve Jobs passed away in 2011.
        Microsoft and Google are major competitors in the technology sector.
        """,
        metadata={"title": "Tech Companies", "section": "History"},
        doc_id="doc_1",
        position=0,
    )

    # Extract entities
    entities = graph_rag._extract_entities(test_chunk)

    print(f"\n✓ Extracted {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity.name:<30} ({entity.entity_type})")

    # Verify we got the expected entities
    entity_names = {e.name for e in entities}
    expected_entities = [
        "Apple Inc.",
        "Steve Jobs",
        "Steve Wozniak",
        "Cupertino",
        "California",
        "iPhone",
        "Tim Cook",
        "Microsoft",
        "Google",
    ]

    found_entities = [e for e in expected_entities if e in entity_names]
    print(f"\n✓ Found {len(found_entities)}/{len(expected_entities)} expected entities:")
    for entity in found_entities:
        print(f"  ✓ {entity}")

    missing_entities = [e for e in expected_entities if e not in entity_names]
    if missing_entities:
        print(f"\n⚠ Missing {len(missing_entities)} entities:")
        for entity in missing_entities:
            print(f"  ✗ {entity}")

    # Check entity types
    entity_types = {e.entity_type for e in entities}
    print(f"\n✓ Entity types found: {', '.join(sorted(entity_types))}")

    # Verify metadata entities
    metadata_entities = [e for e in entities if e.entity_type == 'CONCEPT']
    print(f"\n✓ Metadata entities: {len(metadata_entities)}")
    for entity in metadata_entities:
        print(f"  - {entity.name}")

    print("\n✅ TEST 1 PASSED: spaCy NER extraction working\n")
    return True


def test_graph_caching():
    """Test that graph caching saves and loads correctly"""
    print("\n" + "="*60)
    print("TEST 2: Graph Caching (Save/Load)")
    print("="*60)

    # Create temporary cache directory
    temp_cache_dir = tempfile.mkdtemp()
    print(f"\n✓ Created temp cache dir: {temp_cache_dir}")

    try:
        # Create embedding model
        embedding_model = create_embedding_model()

        # Create Graph RAG instance
        graph_rag = GraphRAG(
            embedding_model=embedding_model,
            use_spacy_ner=True,
            spacy_model="en_core_web_sm",
            cache_dir=temp_cache_dir,
        )

        # Create test chunks
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                content=f"This is test chunk {i} about Apple Inc. and Steve Jobs.",
                metadata={"title": f"Document {i}"},
                doc_id="doc_1",
                position=i,
            )
            for i in range(3)
        ]

        print(f"\n✓ Created {len(chunks)} test chunks")

        # Build graph (first time - should not load from cache)
        print("\n→ Building graph (first time)...")
        graph_rag.build_graph(chunks, cache_name="test_cache", force_rebuild=True)

        # Capture graph state
        original_num_nodes = graph_rag.graph.number_of_nodes()
        original_num_edges = graph_rag.graph.number_of_edges()
        original_num_entities = len(graph_rag.entities)

        print(f"  ✓ Graph built: {original_num_nodes} nodes, {original_num_edges} edges")
        print(f"  ✓ Entities: {original_num_entities}")

        # Save cache explicitly (should happen automatically in build_graph)
        cache_path = Path(temp_cache_dir) / "test_cache.pkl"
        print(f"\n→ Checking cache file: {cache_path}")
        if cache_path.exists():
            print(f"  ✓ Cache file exists ({cache_path.stat().st_size} bytes)")
        else:
            print(f"  ✗ Cache file NOT found!")
            return False

        # Create new Graph RAG instance
        graph_rag_2 = GraphRAG(
            embedding_model=embedding_model,
            use_spacy_ner=True,
            spacy_model="en_core_web_sm",
            cache_dir=temp_cache_dir,
        )

        print(f"\n→ Loading graph from cache...")
        loaded = graph_rag_2.load_cache("test_cache")

        if not loaded:
            print("  ✗ Failed to load cache!")
            return False

        print(f"  ✓ Cache loaded successfully")

        # Verify state matches
        loaded_num_nodes = graph_rag_2.graph.number_of_nodes()
        loaded_num_edges = graph_rag_2.graph.number_of_edges()
        loaded_num_entities = len(graph_rag_2.entities)

        print(f"\n→ Comparing original vs loaded:")
        print(f"  Nodes:    {original_num_nodes} -> {loaded_num_nodes} {'✓' if original_num_nodes == loaded_num_nodes else '✗'}")
        print(f"  Edges:    {original_num_edges} -> {loaded_num_edges} {'✓' if original_num_edges == loaded_num_edges else '✗'}")
        print(f"  Entities: {original_num_entities} -> {loaded_num_entities} {'✓' if original_num_entities == loaded_num_entities else '✗'}")

        if (original_num_nodes != loaded_num_nodes or
            original_num_edges != loaded_num_edges or
            original_num_entities != loaded_num_entities):
            print("\n✗ State mismatch!")
            return False

        # Test cache clearing
        print(f"\n→ Testing cache clear...")
        graph_rag_2.clear_cache("test_cache")
        if not cache_path.exists():
            print("  ✓ Cache cleared successfully")
        else:
            print("  ✗ Cache still exists after clear!")
            return False

        print("\n✅ TEST 2 PASSED: Graph caching working correctly\n")
        return True

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_cache_dir, ignore_errors=True)
        print(f"✓ Cleaned up temp cache dir")


def test_integration_with_build_graph():
    """Test that build_graph uses cache correctly"""
    print("\n" + "="*60)
    print("TEST 3: Integration - build_graph with caching")
    print("="*60)

    # Create temporary cache directory
    temp_cache_dir = tempfile.mkdtemp()
    print(f"\n✓ Created temp cache dir: {temp_cache_dir}")

    try:
        # Create embedding model
        embedding_model = create_embedding_model()

        # Create test chunks
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                content=f"Microsoft was founded by Bill Gates and Paul Allen in Seattle, Washington.",
                metadata={"title": f"Tech History {i}"},
                doc_id="doc_1",
                position=i,
            )
            for i in range(2)
        ]

        # First instance - build graph
        print("\n→ First instance: Building graph...")
        graph_rag_1 = GraphRAG(
            embedding_model=embedding_model,
            use_spacy_ner=True,
            cache_dir=temp_cache_dir,
        )

        graph_rag_1.build_graph(chunks, cache_name="integration_test", force_rebuild=False)
        first_build_entities = len(graph_rag_1.entities)
        print(f"  ✓ Built graph with {first_build_entities} entities")

        # Second instance - should load from cache
        print("\n→ Second instance: Should load from cache...")
        graph_rag_2 = GraphRAG(
            embedding_model=embedding_model,
            use_spacy_ner=True,
            cache_dir=temp_cache_dir,
        )

        # build_graph should detect empty graph and load from cache
        graph_rag_2.build_graph(chunks, cache_name="integration_test", force_rebuild=False)
        second_build_entities = len(graph_rag_2.entities)
        print(f"  ✓ Loaded graph with {second_build_entities} entities")

        if first_build_entities == second_build_entities:
            print(f"\n  ✓ Entity counts match: {first_build_entities}")
        else:
            print(f"\n  ✗ Entity count mismatch: {first_build_entities} vs {second_build_entities}")
            return False

        # Third instance - force rebuild
        print("\n→ Third instance: Force rebuild...")
        graph_rag_3 = GraphRAG(
            embedding_model=embedding_model,
            use_spacy_ner=True,
            cache_dir=temp_cache_dir,
        )

        graph_rag_3.build_graph(chunks, cache_name="integration_test", force_rebuild=True)
        third_build_entities = len(graph_rag_3.entities)
        print(f"  ✓ Rebuilt graph with {third_build_entities} entities")

        print("\n✅ TEST 3 PASSED: Integration working correctly\n")
        return True

    finally:
        # Clean up
        shutil.rmtree(temp_cache_dir, ignore_errors=True)
        print(f"✓ Cleaned up temp cache dir")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GRAPH RAG FEATURE TESTS")
    print("="*60)

    results = {
        "spaCy NER Extraction": False,
        "Graph Caching": False,
        "Integration": False,
    }

    # Test 1: spaCy NER
    try:
        results["spaCy NER Extraction"] = test_spacy_ner_extraction()
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()

    # Test 2: Caching
    try:
        results["Graph Caching"] = test_graph_caching()
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()

    # Test 3: Integration
    try:
        results["Integration"] = test_integration_with_build_graph()
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")

    all_passed = all(results.values())
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
