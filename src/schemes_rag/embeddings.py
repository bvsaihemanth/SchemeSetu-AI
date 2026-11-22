"""BGE-M3 embedding layer.

One forward pass of BGE-M3 yields three representations; we take two:
  - dense  : 1024-d, semantic similarity
  - sparse : lexical weights (token_id -> weight), exact-term matching

Storing both at ingest time is the single cheapest decision in this project.
Dense-only retrieval fails on exactly the queries this corpus attracts:
scheme acronyms, section numbers, amounts ("PM-KISAN", "Rs 6000", "SC/ST").
Sparse catches those. Enabling hybrid search later then costs one config
flag instead of re-embedding 725 documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    dense: list[list[float]]
    sparse: list[dict[int, float]] | None = None


class BGEM3Embedder:
    """Lazy-loading wrapper around FlagEmbedding's BGEM3FlagModel."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            s = self._settings
            logger.info("Loading %s on %s (fp16=%s)", s.embed_model, s.embed_device, s.embed_use_fp16)
            self._model = BGEM3FlagModel(
                s.embed_model,
                use_fp16=s.embed_use_fp16 and s.embed_device != "cpu",
                devices=s.embed_device,
            )
        return self._model

    def encode_documents(self, texts: Iterable[str]) -> EmbeddingResult:
        return self._encode(list(texts), with_sparse=self._settings.embed_store_sparse)

    def encode_query(self, text: str) -> EmbeddingResult:
        """BGE-M3 needs no query prefix — unlike bge-large-en, which wants
        'Represent this sentence for searching...'. Do not add one."""
        return self._encode([text], with_sparse=self._settings.retrieval_hybrid)

    def encode_queries(self, texts: list[str]) -> EmbeddingResult:
        """Batch several query texts into one forward pass. Used to embed a
        rewritten search query and the user's original question together,
        which costs the same as embedding either one alone."""
        return self._encode(list(texts), with_sparse=self._settings.retrieval_hybrid)

    def _encode(self, texts: list[str], with_sparse: bool) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(dense=[], sparse=[] if with_sparse else None)

        model = self._ensure_model()
        s = self._settings
        out = model.encode(
            texts,
            batch_size=s.embed_batch_size,
            max_length=s.embed_max_length,
            return_dense=True,
            return_sparse=with_sparse,
            return_colbert_vecs=False,
        )

        dense = [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in out["dense_vecs"]]

        sparse: list[dict[int, float]] | None = None
        if with_sparse:
            sparse = [
                {int(tok): float(w) for tok, w in weights.items() if float(w) > 0.0}
                for weights in out["lexical_weights"]
            ]

        return EmbeddingResult(dense=dense, sparse=sparse)


class RemoteEmbedder:
    """Talks to a standalone embedding service (see ops/ for a TEI container).

    Use this the moment the API runs more than one worker: BGE-M3 is ~2.3 GB
    in fp16, and N uvicorn workers means N copies of it in RAM.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, texts: list[str], with_sparse: bool) -> EmbeddingResult:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/embed",
            json={"texts": texts, "return_sparse": with_sparse},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        sparse = None
        if with_sparse and data.get("sparse") is not None:
            sparse = [{int(k): float(v) for k, v in d.items()} for d in data["sparse"]]
        return EmbeddingResult(dense=data["dense"], sparse=sparse)

    def encode_documents(self, texts: Iterable[str]) -> EmbeddingResult:
        return self._post(list(texts), with_sparse=get_settings().embed_store_sparse)

    def encode_query(self, text: str) -> EmbeddingResult:
        return self._post([text], with_sparse=get_settings().retrieval_hybrid)

    def encode_queries(self, texts: list[str]) -> EmbeddingResult:
        return self._post(list(texts), with_sparse=get_settings().retrieval_hybrid)


@lru_cache
def get_embedder():
    s = get_settings()
    if s.embed_service_url:
        logger.info("Using remote embedder at %s", s.embed_service_url)
        return RemoteEmbedder(s.embed_service_url)
    return BGEM3Embedder()
