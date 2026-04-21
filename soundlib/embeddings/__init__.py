"""Pluggable embedding backends."""
from soundlib.embeddings.base import Backend, EmbeddingResult
from soundlib.embeddings.registry import get_backend, list_backends

__all__ = ["Backend", "EmbeddingResult", "get_backend", "list_backends"]
