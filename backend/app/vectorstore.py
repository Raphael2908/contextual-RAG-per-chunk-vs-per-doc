"""Vector store, behind a Protocol with a Milvus impl + in-memory fake.

One collection per embedding model (dims differ across models), so a benchmark
sweep keeps each combo's vectors isolated. Supports two retrieval modes:

  - **dense** — cosine ANN over the embedded vectors (the original slice).
  - **hybrid** — dense + sparse/BM25, RRF-fused (architecture.md §"Search layer:
    Milvus only" requires this — sparse recovers exact terms dense misses:
    citations, defined terms, party names). Milvus does BM25 natively (a BM25
    `Function` over a `text` field → `SPARSE_FLOAT_VECTOR`, needs server 2.5+);
    the in-memory fake implements BM25 + RRF in Python so the hybrid path + fusion
    are verifiable with no datastore.

A hybrid collection has a different schema (text + sparse field + BM25 function)
and so a different name (`collection_for(..., hybrid=True)` adds a `_hybrid` tag);
the BM25 function must exist at creation time, so switching modes re-ingests.

`search` returns ranked `Hit`s (chunk id + score + document id), consumed by the
query path and the retrieval-quality eval.
"""

from __future__ import annotations

import math
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")


def collection_for(
    embedding_model: str, dim: int, *, context_mode: str = "per_chunk", hybrid: bool = False
) -> str:
    """Stable, Milvus-safe collection name for a model+dim+context_mode.

    The context_mode segment gives each contextual-retrieval system (none /
    per_chunk / per_document) its OWN collection, so their vectors are never
    commingled — a clean per-system store you can query and score independently
    (architecture.md keeps modes separate). Older names lacked it; re-ingest.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    mode = re.sub(r"[^a-z0-9]+", "_", context_mode.lower()).strip("_") or "per_chunk"
    return f"chunks_{slug}_{dim}_{mode}" + ("_hybrid" if hybrid else "")


@dataclass
class Hit:
    id: str
    score: float
    document_id: str | None = None


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, name: str, dim: int, *, hybrid: bool = False) -> None: ...

    def upsert(
        self,
        name: str,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        *,
        texts: list[str] | None = None,
        hybrid: bool = False,
    ) -> None: ...

    def search(
        self,
        name: str,
        query_vector: list[float],
        k: int,
        *,
        filter: str = "",
        query_text: str | None = None,
        hybrid: bool = False,
        rrf_k: int = 60,
    ) -> list[Hit]: ...

    def count(self, name: str) -> int: ...

    def drop(self, name: str) -> None: ...


class MilvusStore:
    def __init__(self, uri: str, token: str = ""):
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri, token=token or None)
        self._ready: set[str] = set()
        # Bulk ingest calls ensure_collection from many worker threads at once;
        # serialize creation so two threads don't race on create_collection or
        # the unsynchronized _ready set. A no-op after warm-up (the membership
        # check short-circuits before taking the lock).
        self._lock = threading.Lock()

    def ensure_collection(self, name: str, dim: int, *, hybrid: bool = False) -> None:
        if name in self._ready:
            return
        with self._lock:
            if name in self._ready:
                return
            if not self._client.has_collection(name):
                if hybrid:
                    self._create_hybrid(name, dim)
                else:
                    self._client.create_collection(
                        collection_name=name,
                        dimension=dim,
                        id_type="string",
                        max_length=64,
                        metric_type="COSINE",
                        auto_id=False,
                    )
            if hybrid:
                self._client.load_collection(name)  # hybrid_search needs it loaded
            self._ready.add(name)

    def _create_hybrid(self, name: str, dim: int) -> None:
        """Explicit schema: dense `vector` + analyzed `text` → BM25 `sparse_vector`."""
        from pymilvus import DataType, Function, FunctionType

        schema = self._client.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(
            "text", DataType.VARCHAR, max_length=8192,
            enable_analyzer=True, analyzer_params={"type": "standard"},
        )
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("document_id", DataType.VARCHAR, max_length=64)
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,
            )
        )
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(
            field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )
        self._client.create_collection(
            collection_name=name, schema=schema, index_params=index_params
        )

    def upsert(
        self, name, ids, vectors, documents, *, texts=None, hybrid=False
    ) -> None:
        texts = texts or [""] * len(ids)
        if hybrid:
            # sparse_vector is generated server-side by the BM25 function from `text`.
            rows = [
                {"id": cid, "vector": vec, "text": txt, "document_id": doc}
                for cid, vec, txt, doc in zip(ids, vectors, texts, documents)
            ]
        else:
            rows = [
                {"id": cid, "vector": vec, "document_id": doc}
                for cid, vec, doc in zip(ids, vectors, documents)
            ]
        if rows:
            self._client.upsert(collection_name=name, data=rows)

    def search(
        self, name, query_vector, k, *, filter="", query_text=None, hybrid=False, rrf_k=60
    ) -> list[Hit]:
        if not self._client.has_collection(name):
            return []
        if hybrid and query_text is not None:
            return self._hybrid_search(name, query_vector, query_text, k, filter, rrf_k)
        res = self._client.search(
            collection_name=name,
            data=[query_vector],
            limit=k,
            filter=filter or "",
            output_fields=["document_id"],
            search_params={"metric_type": "COSINE"},
            consistency_level="Strong",
        )
        return self._to_hits(res)

    def _hybrid_search(self, name, query_vector, query_text, k, filter, rrf_k) -> list[Hit]:
        from pymilvus import AnnSearchRequest, RRFRanker

        self._client.load_collection(name)
        depth = max(k * 4, 20)
        dense_req = AnnSearchRequest(
            data=[query_vector], anns_field="vector",
            param={"metric_type": "COSINE"}, limit=depth, expr=filter or None,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text], anns_field="sparse_vector",
            param={"metric_type": "BM25"}, limit=depth, expr=filter or None,
        )
        res = self._client.hybrid_search(
            collection_name=name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(rrf_k),
            limit=k,
            output_fields=["document_id"],
        )
        return self._to_hits(res)

    @staticmethod
    def _to_hits(res) -> list[Hit]:
        hits: list[Hit] = []
        for h in (res[0] if res else []):
            entity = h.get("entity") or {}
            hits.append(
                Hit(
                    id=str(h.get("id")),
                    score=float(h.get("distance", 0.0)),
                    document_id=entity.get("document_id"),
                )
            )
        return hits

    def count(self, name: str) -> int:
        if not self._client.has_collection(name):
            return 0
        # Strong consistency so a count taken right after upsert reflects it
        # (the default Bounded level can lag behind fresh writes).
        res = self._client.query(
            collection_name=name,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(res[0]["count(*)"]) if res else 0

    def drop(self, name: str) -> None:
        if self._client.has_collection(name):
            self._client.drop_collection(name)
        self._ready.discard(name)


class InMemoryStore:
    """Dependency-free store. Hybrid mode runs a real (compact) BM25 + RRF fusion
    so the fusion logic — not just plumbing — is exercised on fakes."""

    def __init__(self) -> None:
        # name -> {chunk_id: (vector, document_id, text)}
        self._data: dict[str, dict[str, tuple[list[float], str | None, str]]] = {}

    def ensure_collection(self, name: str, dim: int, *, hybrid: bool = False) -> None:
        self._data.setdefault(name, {})

    def upsert(
        self, name, ids, vectors, documents, *, texts=None, hybrid=False
    ) -> None:
        coll = self._data.setdefault(name, {})
        docs = documents or [None] * len(ids)
        texts = texts or [""] * len(ids)
        for cid, vec, doc, txt in zip(ids, vectors, docs, texts):
            coll[cid] = (vec, doc, txt)

    def search(
        self, name, query_vector, k, *, filter="", query_text=None, hybrid=False, rrf_k=60
    ) -> list[Hit]:
        coll = self._data.get(name, {})
        if not coll:
            return []
        dense = self._dense_ranked(coll, query_vector)
        if not (hybrid and query_text):
            return [
                Hit(id=cid, score=score, document_id=coll[cid][1])
                for cid, score in dense[:k]
            ]
        sparse = self._bm25_ranked(coll, query_text)
        fused = _rrf([cid for cid, _ in dense], [cid for cid, _ in sparse], rrf_k)
        return [Hit(id=cid, score=score, document_id=coll[cid][1]) for cid, score in fused[:k]]

    @staticmethod
    def _dense_ranked(coll, query_vector) -> list[tuple[str, float]]:
        qn = _norm(query_vector)
        scored = [(cid, _cosine(query_vector, vec, qn)) for cid, (vec, _, _) in coll.items()]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    @staticmethod
    def _bm25_ranked(coll, query_text, k1: float = 1.5, b: float = 0.75) -> list[tuple[str, float]]:
        tokens = {cid: _TOKEN.findall(txt.lower()) for cid, (_, _, txt) in coll.items()}
        n = len(tokens)
        avgdl = (sum(len(t) for t in tokens.values()) / n) if n else 0.0
        df: dict[str, int] = defaultdict(int)
        for toks in tokens.values():
            for term in set(toks):
                df[term] += 1
        q_terms = _TOKEN.findall(query_text.lower())
        scored: list[tuple[str, float]] = []
        for cid, toks in tokens.items():
            if not toks:
                continue
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            dl = len(toks)
            s = 0.0
            for qt in q_terms:
                if qt not in tf:
                    continue
                idf = math.log(1 + (n - df[qt] + 0.5) / (df[qt] + 0.5))
                s += idf * (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * dl / (avgdl or 1)))
            if s > 0:
                scored.append((cid, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def count(self, name: str) -> int:
        return len(self._data.get(name, {}))

    def drop(self, name: str) -> None:
        self._data.pop(name, None)


def _rrf(dense_ids: list[str], sparse_ids: list[str], rrf_k: int) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion: score = Σ 1/(rrf_k + rank), rank starting at 1."""
    scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] += 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda t: t[1], reverse=True)


def _norm(v: list[float]) -> float:
    return sum(x * x for x in v) ** 0.5 or 1.0


def _cosine(a: list[float], b: list[float], a_norm: float | None = None) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((a_norm or _norm(a)) * _norm(b))
