import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from utils.embedding_model import embed_chunks, embed_documents, get_document_embedding


def _mock_encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True):
    return np.random.rand(len(texts), 384).astype("float32")


@pytest.fixture(autouse=True)
def mock_model():
    model = MagicMock()
    model.encode.side_effect = _mock_encode
    with patch("utils.embedding_model._get_model", return_value=model):
        yield model


def test_embed_chunks_shape():
    chunks = ["Hello world.", "Another sentence here for testing purposes."]
    result = embed_chunks(chunks)
    assert result.shape == (2, 384)


def test_embed_chunks_empty():
    result = embed_chunks([])
    assert result.size == 0


def test_embed_chunks_returns_float32(mock_model):
    mock_model.encode.side_effect = lambda texts, **kw: np.ones((len(texts), 384), dtype="float32")
    result = embed_chunks(["test chunk"])
    assert result.dtype == np.float32


def test_embed_documents_keys():
    docs = {"doc1": ["chunk one", "chunk two"], "doc2": ["another chunk"]}
    result = embed_documents(docs)
    assert set(result.keys()) == {"doc1", "doc2"}
    assert result["doc1"].shape == (2, 384)
    assert result["doc2"].shape == (1, 384)


def test_embed_documents_empty_doc(capsys):
    docs = {"empty_doc": []}
    result = embed_documents(docs)
    assert result["empty_doc"].size == 0


def test_get_document_embedding_mean_pool():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = get_document_embedding(emb)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, [0.5, 0.5])


def test_get_document_embedding_single_vector():
    vec = np.array([0.1, 0.2, 0.3])
    result = get_document_embedding(vec)
    np.testing.assert_array_equal(result, vec)
