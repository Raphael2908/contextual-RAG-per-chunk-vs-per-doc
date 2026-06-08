"""Environment-driven settings for the minimal slice.

A deliberately small subset of `.env.example` — only what ingestion + the
benchmark need. Model strings default to LiteLLM's provider-prefixed form
(`anthropic/...`, `voyage/...`); `app.litellm_util.normalize_model` also accepts
the bare names used in `.env.example` (`claude-haiku-4-5`, `voyage-law-2`).
"""

from __future__ import annotations

import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Postgres ---
    database_url: str = Field(
        "postgresql://postgres:postgres@localhost:5432/postgres",
        validation_alias=AliasChoices("APP_DATABASE_URL", "DATABASE_URL"),
    )

    # --- Milvus ---
    milvus_uri: str = Field(
        "http://localhost:19530",
        validation_alias=AliasChoices("RAG_MILVUS_URI", "MILVUS_URI"),
    )
    milvus_token: str = Field("", validation_alias=AliasChoices("RAG_MILVUS_TOKEN"))

    # --- Default models (any LiteLLM model string; swappable per benchmark combo) ---
    context_model: str = Field(
        "anthropic/claude-haiku-4-5",
        validation_alias=AliasChoices("RAG_CONTEXT_MODEL", "CONTEXT_MODEL"),
    )
    embedding_model: str = Field(
        "voyage/voyage-law-2",
        validation_alias=AliasChoices("RAG_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    # Contextual-retrieval strategy for the live ingest path (the benchmark sweeps
    # all three as an axis): "none" (bare chunks), "per_chunk" (one LLM call/chunk →
    # a situating sentence, the default), or "per_document" (one longer LLM call/doc,
    # reused across every chunk, carrying the doc id + filename). See app/contextual.py.
    context_mode: str = Field(
        "per_chunk", validation_alias=AliasChoices("RAG_CONTEXT_MODE")
    )
    # Stronger model for the cited-answer path (kept separate from the cheap
    # context_model used for per-chunk enrichment — architecture.md §Contextual
    # retrieval / §Retrieval step 6).
    answer_model: str = Field(
        "anthropic/claude-opus-4-8",
        validation_alias=AliasChoices("RAG_ANTHROPIC_MODEL", "ANTHROPIC_MODEL"),
    )
    dense_dim: int = Field(1024, validation_alias=AliasChoices("RAG_DENSE_DIM"))

    # --- Retrieval / rerank (the read path + retrieval-quality eval) ---
    # "dense" (cosine ANN only) or "hybrid" (dense + sparse/BM25, RRF-fused —
    # architecture.md's required mode). Default stays dense for backward-compat;
    # the eval sweeps both. Hybrid needs Milvus server 2.5+ (BM25 function).
    retrieval_mode: str = Field("dense", validation_alias=AliasChoices("RAG_RETRIEVAL_MODE"))
    # Empty rerank model → IdentityReranker (dense order kept); set a LiteLLM
    # rerank string (`voyage/rerank-2`, `cohere/rerank-3.5`, ...) to rerank for real.
    rerank_model: str = Field("", validation_alias=AliasChoices("RAG_RERANK_MODEL"))
    retrieval_top_k: int = Field(10, validation_alias=AliasChoices("RAG_TOP_K"))
    rerank_candidates: int = Field(
        30, validation_alias=AliasChoices("RAG_RERANK_CANDIDATES")
    )
    # RRF constant for dense+sparse fusion — wired for the (deferred) hybrid step.
    rrf_k: int = Field(60, validation_alias=AliasChoices("RAG_RRF_K"))
    # k for recall@k / nDCG@k in the eval harness.
    eval_k: int = Field(5, validation_alias=AliasChoices("RAG_EVAL_K"))

    # --- Blob store (docker volume) ---
    blob_dir: str = Field(
        "/data/blobs", validation_alias=AliasChoices("APP_BLOB_DIR", "BLOB_DIR")
    )

    # --- Bulk upload (in-process async ingest worker pool; no Celery/Redis) ---
    # Bounded true parallelism for the slow ingest path. Default 4 is conservative
    # for Anthropic/Voyage rate limits; raise via env (and bump the psycopg pool
    # max_size if pushing above ~8).
    bulk_ingest_concurrency: int = Field(
        4, validation_alias=AliasChoices("RAG_BULK_INGEST_CONCURRENCY")
    )
    # Backpressure: the upload endpoint awaits queue.put when this many jobs are
    # already queued. Jobs are tiny descriptors (blob-first), so this can be large.
    bulk_ingest_queue_max: int = Field(
        2000, validation_alias=AliasChoices("RAG_BULK_INGEST_QUEUE_MAX")
    )

    # --- Provider API keys. LiteLLM reads ANTHROPIC_API_KEY / VOYAGE_API_KEY /
    #     OPENAI_API_KEY / OPENROUTER_API_KEY from the process env; we mirror the
    #     RAG_* names from .env.example into those on startup (see export_provider_keys). ---
    anthropic_api_key: str = Field(
        "", validation_alias=AliasChoices("RAG_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    )
    voyage_api_key: str = Field(
        "", validation_alias=AliasChoices("RAG_EMBEDDING_API_KEY", "VOYAGE_API_KEY")
    )
    openai_api_key: str = Field("", validation_alias=AliasChoices("OPENAI_API_KEY"))
    openrouter_api_key: str = Field(
        "", validation_alias=AliasChoices("OPENROUTER_API_KEY")
    )

    # --- Auth / access (beta: local Postgres + Auth.js, no Supabase) ---
    # "header" (dev: identity from a header, no service token) or "proxy" (beta:
    # the Next.js BFF / Telegram bot must present APP_SERVICE_TOKEN). See security.py.
    auth_mode: str = Field("header", validation_alias=AliasChoices("APP_AUTH_MODE"))
    # Shared secret between the unpublished FastAPI core and its trusted front doors
    # (the Next.js proxy + the Telegram bot).
    service_token: str = Field("", validation_alias=AliasChoices("APP_SERVICE_TOKEN"))
    # Login allow-list for the web app — exact emails ∪ whole domains (comma-sep).
    allowed_emails: str = Field("", validation_alias=AliasChoices("APP_ALLOWED_EMAILS"))
    allowed_email_domains: str = Field(
        "", validation_alias=AliasChoices("APP_ALLOWED_EMAIL_DOMAINS")
    )

    # --- Telegram (long-polling bot; no public URL needed) ---
    telegram_bot_token: str = Field(
        "", validation_alias=AliasChoices("APP_TELEGRAM_BOT_TOKEN")
    )
    # Comma-sep allow-list of Telegram numeric user ids permitted to use the bot.
    telegram_allowed_users: str = Field(
        "", validation_alias=AliasChoices("APP_TELEGRAM_ALLOWED_USERS")
    )

    # --- Run everything on dependency-free fakes (no network, no keys) ---
    use_fakes: bool = Field(
        False, validation_alias=AliasChoices("APP_USE_FAKES", "USE_FAKES")
    )

    def allowed_email_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_emails.split(",") if e.strip()}

    def allowed_domain_set(self) -> set[str]:
        return {
            d.strip().lower().lstrip("@")
            for d in self.allowed_email_domains.split(",")
            if d.strip()
        }

    def telegram_allowed_set(self) -> set[str]:
        return {u.strip() for u in self.telegram_allowed_users.split(",") if u.strip()}

    def export_provider_keys(self) -> None:
        """Mirror configured keys into the env vars LiteLLM expects."""
        for value, name in (
            (self.anthropic_api_key, "ANTHROPIC_API_KEY"),
            (self.voyage_api_key, "VOYAGE_API_KEY"),
            (self.openai_api_key, "OPENAI_API_KEY"),
            (self.openrouter_api_key, "OPENROUTER_API_KEY"),
        ):
            if value and not os.getenv(name):
                os.environ[name] = value


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
