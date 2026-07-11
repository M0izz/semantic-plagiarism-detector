"""
conftest.py
-----------
Session-level stub for sentence_transformers so tests can run without
a fully compatible TensorFlow / Keras installation.
The embedding_model tests mock _get_model() directly, so no real model
is needed.
"""
import sys
import types
from unittest.mock import MagicMock

# Stub sentence_transformers before any test module imports it.
if "sentence_transformers" not in sys.modules:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = MagicMock  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = stub
