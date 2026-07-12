import pytest
import os
import numpy as np
from src.db.corpus_db import (
    init_corpus_db,
    add_document,
    get_document_by_hash,
    get_all_documents,
    add_chunks,
    get_chunk_registry,
    get_all_embeddings,
    delete_document,
    clear_all_data,
    get_document_chunks_count,
    _DB_PATH
)


@pytest.fixture(autouse=True)
def setup_test_db():
    # Initialize database
    init_corpus_db()
    # Clear any leftover records
    clear_all_data()
    yield
    # Cleanup after tests
    clear_all_data()


def test_add_document_metadata():
    # Add first document
    res1 = add_document("test1.pdf", "hash_abc_123")
    assert res1 is True

    # Try adding a duplicate hash/document
    res2 = add_document("test2.pdf", "hash_abc_123")
    assert res2 is False  # Unique hash constraint triggers

    # Try adding a duplicate filename
    res3 = add_document("test1.pdf", "different_hash")
    assert res3 is False  # Unique filename constraint triggers


def test_get_document_by_hash():
    add_document("doc_alpha.txt", "hash_xyz_789")
    
    match = get_document_by_hash("hash_xyz_789")
    assert match == "doc_alpha.txt"
    
    no_match = get_document_by_hash("nonexistent_hash")
    assert no_match is None


def test_add_and_retrieve_chunks():
    add_document("doc1.pdf", "hash_1")
    
    # Format of chunk insertion tuples: (vector_id, filename, chunk_index, chunk_text, embedding)
    dummy_emb_1 = np.ones(384, dtype=np.float32) * 0.5
    dummy_emb_2 = np.ones(384, dtype=np.float32) * 1.5
    
    chunks = [
        (0, "doc1.pdf", 0, "Paragraph 1 text", dummy_emb_1),
        (1, "doc1.pdf", 1, "Paragraph 2 text", dummy_emb_2)
    ]
    
    add_chunks(chunks)
    
    # Check count
    assert get_document_chunks_count("doc1.pdf") == 2
    
    # Check registry loading
    registry = get_chunk_registry()
    assert len(registry) == 2
    assert registry[0].doc_name == "doc1.pdf"
    assert registry[0].chunk_text == "Paragraph 1 text"
    
    # Check embeddings extraction
    embs = get_all_embeddings()
    assert embs.shape == (2, 384)
    assert np.allclose(embs[0], dummy_emb_1)
    assert np.allclose(embs[1], dummy_emb_2)


def test_delete_document_cascades():
    add_document("doc1.pdf", "hash_1")
    add_document("doc2.pdf", "hash_2")
    
    dummy_emb = np.zeros(384, dtype=np.float32)
    
    chunks = [
        (0, "doc1.pdf", 0, "Paragraph 1", dummy_emb),
        (1, "doc2.pdf", 0, "Paragraph 2", dummy_emb)
    ]
    add_chunks(chunks)
    
    # Delete doc1
    delete_document("doc1.pdf")
    
    # Check document counts
    all_docs = get_all_documents()
    assert len(all_docs) == 1
    assert all_docs[0]["filename"] == "doc2.pdf"
    
    # Check that remaining chunks have compact vector_ids starting at 0
    registry = get_chunk_registry()
    assert len(registry) == 1
    assert registry[0].doc_name == "doc2.pdf"
    
    embs = get_all_embeddings()
    assert embs.shape == (1, 384)
