"""Composition helpers — build providers (real or fake) from settings.

Keeps the real/fake selection in one place so the API, the benchmark CLI, and
tests all wire dependencies the same way.
"""

from __future__ import annotations

from app.blob import BlobStore, FilesystemBlobStore, InMemoryBlobStore
from app.config import Settings
from app.embeddings import Embedder, FakeEmbedder, LiteLLMEmbedder
from app.llm import LLM, FakeLLM, LiteLLMContextualizer
from app.reranker import IdentityReranker, LiteLLMReranker, Reranker
from app.vectorstore import InMemoryStore, MilvusStore, VectorStore


def make_blob(settings: Settings) -> BlobStore:
    if settings.use_fakes:
        return InMemoryBlobStore()
    return FilesystemBlobStore(settings.blob_dir)


def make_vector_store(settings: Settings) -> VectorStore:
    if settings.use_fakes:
        return InMemoryStore()
    return MilvusStore(settings.milvus_uri, settings.milvus_token)


def make_llm(settings: Settings, model: str | None) -> LLM | None:
    """`model is None` means the no-context control (no LLM at all)."""
    if model is None:
        return None
    if settings.use_fakes:
        return FakeLLM(model)
    return LiteLLMContextualizer(model)


def make_embedder(settings: Settings, model: str, *, dim: int | None = None) -> Embedder:
    if settings.use_fakes:
        return FakeEmbedder(model, dim=dim or settings.dense_dim)
    return LiteLLMEmbedder(model, dim=dim)


def make_reranker(settings: Settings) -> Reranker:
    """Identity (no-op) unless a real rerank model is configured and fakes are off."""
    if settings.use_fakes or not settings.rerank_model:
        return IdentityReranker()
    return LiteLLMReranker(settings.rerank_model)
