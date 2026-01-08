"""
Tests for Collection Management
"""

import pytest
import json
import os
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings


class TestCollectionManagement:
    """Test collection CRUD operations"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Create temp directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.collections_file = os.path.join(self.test_dir, "collections.json")

        # Mock the collections file path
        import rag.main as main_module
        self.original_collections_file = main_module.COLLECTIONS_FILE
        main_module.COLLECTIONS_FILE = self.collections_file

        yield

        # Cleanup
        main_module.COLLECTIONS_FILE = self.original_collections_file
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_empty_collections(self):
        """Test loading collections when file doesn't exist"""
        from rag.main import _load_collections

        collections = _load_collections()
        assert collections == {}
        assert isinstance(collections, dict)

    def test_save_and_load_collections(self):
        """Test saving and loading collections"""
        from rag.main import _save_collections, _load_collections

        test_collections = {
            "test-collection": {
                "id": "test-collection",
                "title": "Test Collection",
                "llm_model": "llama3:latest",
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "file_count": 0,
                "chunk_count": 0,
                "files": [],
                "file_metadata": {}
            }
        }

        _save_collections(test_collections)
        loaded = _load_collections()

        assert "test-collection" in loaded
        assert loaded["test-collection"]["title"] == "Test Collection"
        assert loaded["test-collection"]["llm_model"] == "llama3:latest"

    def test_create_new_collection(self):
        """Test creating a new collection"""
        from rag.main import _get_or_create_collection

        collection = _get_or_create_collection(
            collection_id="new-test",
            title="New Test Collection",
            llm_model="llama3:latest",
            embedding_model="BAAI/bge-large-en-v1.5",
            embedding_dimension=1024
        )

        assert collection["id"] == "new-test"
        assert collection["title"] == "New Test Collection"
        assert collection["file_count"] == 0
        assert collection["chunk_count"] == 0
        assert "files" in collection
        assert "file_metadata" in collection
        assert "created_at" in collection

    def test_get_existing_collection(self):
        """Test getting an existing collection"""
        from rag.main import _get_or_create_collection

        # Create collection
        _get_or_create_collection(
            collection_id="existing",
            title="Existing Collection",
            llm_model="llama3:latest",
            embedding_model="BAAI/bge-large-en-v1.5"
        )

        # Get same collection again (should not create new)
        collection = _get_or_create_collection(
            collection_id="existing",
            title="Different Title",  # Should be ignored
            llm_model="different-model",  # Should be ignored
            embedding_model="different-embedding"  # Should be ignored
        )

        # Original values should be preserved
        assert collection["title"] == "Existing Collection"
        assert collection["llm_model"] == "llama3:latest"

    def test_update_collection_stats(self):
        """Test updating collection stats after file upload"""
        from rag.main import _get_or_create_collection, _update_collection_stats, _load_collections

        # Create collection
        _get_or_create_collection(
            collection_id="stats-test",
            title="Stats Test",
            llm_model="llama3:latest",
            embedding_model="BAAI/bge-large-en-v1.5"
        )

        # Update stats
        _update_collection_stats(
            collection_id="stats-test",
            filename="test.pdf",
            num_chunks=10,
            file_size=1024
        )

        # Verify
        collections = _load_collections()
        collection = collections["stats-test"]

        assert collection["file_count"] == 1
        assert collection["chunk_count"] == 10
        assert "test.pdf" in collection["files"]
        assert "test.pdf" in collection["file_metadata"]
        assert collection["file_metadata"]["test.pdf"]["size"] == 1024
        assert collection["file_metadata"]["test.pdf"]["chunks"] == 10
        assert "uploaded_at" in collection["file_metadata"]["test.pdf"]

    def test_update_collection_stats_multiple_files(self):
        """Test adding multiple files to collection"""
        from rag.main import _get_or_create_collection, _update_collection_stats, _load_collections

        # Create collection
        _get_or_create_collection(
            collection_id="multi-file",
            title="Multi File Test",
            llm_model="llama3:latest",
            embedding_model="BAAI/bge-large-en-v1.5"
        )

        # Add first file
        _update_collection_stats("multi-file", "file1.pdf", 5, 512)

        # Add second file
        _update_collection_stats("multi-file", "file2.pdf", 8, 1024)

        # Verify
        collections = _load_collections()
        collection = collections["multi-file"]

        assert collection["file_count"] == 2
        assert collection["chunk_count"] == 13  # 5 + 8
        assert len(collection["files"]) == 2
        assert "file1.pdf" in collection["files"]
        assert "file2.pdf" in collection["files"]
        assert collection["file_metadata"]["file1.pdf"]["chunks"] == 5
        assert collection["file_metadata"]["file2.pdf"]["chunks"] == 8

    def test_delete_collection(self):
        """Test deleting a collection"""
        from rag.main import _get_or_create_collection, _delete_collection, _load_collections

        # Create collection
        _get_or_create_collection(
            collection_id="to-delete",
            title="To Delete",
            llm_model="llama3:latest",
            embedding_model="BAAI/bge-large-en-v1.5"
        )

        # Verify it exists
        collections = _load_collections()
        assert "to-delete" in collections

        # Delete
        _delete_collection("to-delete")

        # Verify deletion
        collections = _load_collections()
        assert "to-delete" not in collections

    def test_file_metadata_backward_compatibility(self):
        """Test that legacy collections without file_metadata still work"""
        from rag.main import _save_collections, _update_collection_stats, _load_collections

        # Create legacy collection (without file_metadata)
        legacy_collection = {
            "legacy": {
                "id": "legacy",
                "title": "Legacy Collection",
                "llm_model": "llama3:latest",
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "file_count": 0,
                "chunk_count": 0,
                "files": []
                # Note: no file_metadata field
            }
        }

        _save_collections(legacy_collection)

        # Update stats (should add file_metadata automatically)
        _update_collection_stats("legacy", "test.pdf", 5, 512)

        # Verify file_metadata was added
        collections = _load_collections()
        assert "file_metadata" in collections["legacy"]
        assert "test.pdf" in collections["legacy"]["file_metadata"]

    def test_validate_embedding_compatibility(self):
        """Test embedding model compatibility validation"""
        from rag.main import _get_or_create_collection, _validate_embedding_compatibility

        # Create collection with specific dimension
        _get_or_create_collection(
            collection_id="compat-test",
            title="Compatibility Test",
            llm_model="llama3:latest",
            embedding_model="BAAI/bge-large-en-v1.5",
            embedding_dimension=1024
        )

        # Test compatible model (same dimension)
        is_compatible, error, dim = _validate_embedding_compatibility(
            collection_id="compat-test",
            new_embedding_model="BAAI/bge-large-en-v1.5"
        )
        assert is_compatible is True
        assert error is None

    def test_get_collection_models(self):
        """Test retrieving LLM and embedding models for collection"""
        from rag.main import _get_or_create_collection, _get_collection_llm_model, _get_collection_embedding_model

        # Create collection with specific models
        _get_or_create_collection(
            collection_id="model-test",
            title="Model Test",
            llm_model="custom-llm:latest",
            embedding_model="custom-embedding-model"
        )

        # Retrieve models
        llm_model = _get_collection_llm_model("model-test")
        embedding_model = _get_collection_embedding_model("model-test")

        assert llm_model == "custom-llm:latest"
        assert embedding_model == "custom-embedding-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
