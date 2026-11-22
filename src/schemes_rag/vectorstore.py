"""Qdrant access layer. The only module that imports qdrant_client.

Collection design:
  - named vectors: "dense" (1024-d cosine) + "sparse" (BGE-M3 lexical)
  - int8 scalar quantization, originals on disk, quantized in RAM
  - payload indexes on every field the agents filter by

Named vectors from day one is what makes hybrid search a config flag later.
A single unnamed vector cannot be extended without recreating the collection.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient, models as qm

from .config import get_settings
from .models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"

# Deterministic namespace -> same chunk always gets the same point id,
# so re-ingesting a document upserts instead of duplicating.
_NAMESPACE = uuid.UUID("6f9b1e2a-0c4d-4b7e-9f3a-1d8c5e7a2b40")

# field -> index schema. Keyword indexes make filtered search O(filtered set)
# instead of a full scan with post-filtering.
_PAYLOAD_INDEXES: dict[str, Any] = {
    "doc_id": qm.PayloadSchemaType.KEYWORD,
    "doc_name": qm.PayloadSchemaType.KEYWORD,
    "scheme_name": qm.PayloadSchemaType.KEYWORD,
    "scheme_slug": qm.PayloadSchemaType.KEYWORD,
    "ministry": qm.PayloadSchemaType.KEYWORD,
    "level": qm.PayloadSchemaType.KEYWORD,
    "state": qm.PayloadSchemaType.KEYWORD,
    "tags": qm.PayloadSchemaType.KEYWORD,
    "section": qm.PayloadSchemaType.KEYWORD,
    "schema_version": qm.PayloadSchemaType.INTEGER,
    # Eligibility facets, dot-notation into the nested payload. These are what
    # make "which schemes fit this person" a filter rather than a guess.
    "facets.occupations": qm.PayloadSchemaType.KEYWORD,
    "facets.categories": qm.PayloadSchemaType.KEYWORD,
    "facets.genders": qm.PayloadSchemaType.KEYWORD,
    "facets.education_levels": qm.PayloadSchemaType.KEYWORD,
    "facets.special_statuses": qm.PayloadSchemaType.KEYWORD,
    "facets.beneficiary_type": qm.PayloadSchemaType.KEYWORD,
    "facets.age_min": qm.PayloadSchemaType.INTEGER,
    "facets.age_max": qm.PayloadSchemaType.INTEGER,
    "facets.income_max": qm.PayloadSchemaType.FLOAT,
    "facets.land_max_ha": qm.PayloadSchemaType.FLOAT,
}


def chunk_point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{chunk_index}"))


class VectorStore:
    def __init__(self, client: QdrantClient | None = None) -> None:
        s = get_settings()
        self.settings = s
        self.collection = s.qdrant_collection
        if client is not None:
            self.client = client
        elif s.qdrant_mode == "local":
            # Exclusive folder lock: only one process at a time.
            logger.warning(
                "Qdrant local mode at %s - no payload indexes, brute-force "
                "search, and ingest cannot run while the API is up.",
                s.qdrant_local_path,
            )
            Path(s.qdrant_local_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=s.qdrant_local_path)
        else:
            self.client = QdrantClient(
                url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=s.qdrant_timeout
            )

    # ---------------- schema ----------------

    def ensure_collection(self, recreate: bool = False) -> None:
        s = self.settings
        exists = self.client.collection_exists(self.collection)

        if exists and recreate:
            logger.warning("Deleting existing collection %s", self.collection)
            self.client.delete_collection(self.collection)
            exists = False

        if not exists:
            logger.info("Creating collection %s (dim=%s)", self.collection, s.embed_dim)
            quant = (
                qm.ScalarQuantization(
                    scalar=qm.ScalarQuantizationConfig(
                        type=qm.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                )
                if s.qdrant_quantization
                else None
            )
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE: qm.VectorParams(
                        size=s.embed_dim,
                        distance=qm.Distance.COSINE,
                        on_disk=s.qdrant_on_disk,
                    )
                },
                sparse_vectors_config={
                    SPARSE: qm.SparseVectorParams(
                        index=qm.SparseIndexParams(on_disk=s.qdrant_on_disk)
                    )
                },
                quantization_config=quant,
                # indexing_threshold=0 DISABLES index building (it is not a
                # "0 KB minimum"). That is what we want during bulk load:
                # no HNSW graph churn while 12k+ points stream in.
                # finalize_index() restores the default afterwards to build
                # the graph once. Forgetting that call leaves the collection
                # on brute-force search forever, with no error to tell you.
                optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=0),
                # Immutable after creation -- see config notes.
                shard_number=s.qdrant_shard_number,
                replication_factor=s.qdrant_replication_factor,
                on_disk_payload=s.qdrant_on_disk,
            )

        for field, schema in _PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(self.collection, field, field_schema=schema)
            except Exception:  # already exists
                pass

    # ---------------- write ----------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[dict[int, float]] | None = None,
        batch_size: int = 128,
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(dense):
            raise ValueError("chunks and dense vectors length mismatch")

        points: list[qm.PointStruct] = []
        for i, chunk in enumerate(chunks):
            vectors: dict[str, Any] = {DENSE: list(dense[i])}
            if sparse is not None and sparse[i]:
                items = sorted(sparse[i].items())
                vectors[SPARSE] = qm.SparseVector(
                    indices=[k for k, _ in items], values=[v for _, v in items]
                )
            payload = chunk.model_dump(mode="json")
            points.append(
                qm.PointStruct(
                    id=chunk_point_id(chunk.doc_id, chunk.chunk_index),
                    vector=vectors,
                    payload=payload,
                )
            )

        for start in range(0, len(points), batch_size):
            self.client.upsert(
                self.collection, points=points[start : start + batch_size], wait=False
            )
        return len(points)

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of a document. Needed for clean re-ingest when a
        doc's chunk count shrinks (stale tail chunks would otherwise linger)."""
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                )
            ),
        )

    # Qdrant's default. Units are KB of vector data, where 1 KB = one 256-dim
    # vector -- so a 1024-dim vector counts as 4. 12.8k chunks is ~51,000 KB,
    # comfortably past the threshold, so the HNSW graph will be built.
    DEFAULT_INDEXING_THRESHOLD = 20_000

    def finalize_index(self, wait: bool = True) -> bool:
        """Re-enable indexing after bulk load so the HNSW graph gets built.

        Returns True when the collection reports green and the optimizer is
        done; False if it is still working or errored.

        Must be called at the end of every ingest run. Until it is, searches
        fall back to brute force -- correct results, but linear in collection
        size, and nothing reports it. `/stats` surfaces optimizer_status so
        you can confirm the graph was actually built.
        """
        self.client.update_collection(
            self.collection,
            optimizers_config=qm.OptimizersConfigDiff(
                indexing_threshold=self.DEFAULT_INDEXING_THRESHOLD
            ),
        )
        return self._await_green() if wait else True

    def _await_green(self, timeout_s: float = 300.0, poll_s: float = 2.0) -> bool:
        """Block until the optimizer finishes building, so an ingest script
        does not exit reporting success while the index is still cold."""
        import time as _time

        deadline = _time.time() + timeout_s
        while _time.time() < deadline:
            try:
                info = self.client.get_collection(self.collection)
            except Exception:
                return False
            status = str(getattr(info, "optimizer_status", "")).lower()
            if "ok" in status and str(info.status).lower() == "green":
                return True
            if "error" in status:
                logger.error("Qdrant optimizer error: %s", status)
                return False
            _time.sleep(poll_s)
        logger.warning("Index build still running after %.0fs; it continues server-side", timeout_s)
        return False

    # ---------------- read ----------------

    def search(
        self,
        dense_vec: Sequence[float],
        sparse_vec: dict[int, float] | None = None,
        limit: int = 24,
        query_filter: qm.Filter | None = None,
        hybrid: bool = True,
    ) -> list[RetrievedChunk]:
        """Hybrid = dense + sparse prefetch fused with Reciprocal Rank Fusion.

        RRF over raw score mixing: cosine and lexical scores live on different
        scales, so fusing by rank is the only defensible way to combine them
        without hand-tuned weights that drift as the corpus grows.
        """
        use_hybrid = hybrid and sparse_vec

        if use_hybrid:
            items = sorted(sparse_vec.items())
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    qm.Prefetch(
                        query=list(dense_vec), using=DENSE, limit=limit * 2, filter=query_filter
                    ),
                    qm.Prefetch(
                        query=qm.SparseVector(
                            indices=[k for k, _ in items], values=[v for _, v in items]
                        ),
                        using=SPARSE,
                        limit=limit * 2,
                        filter=query_filter,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(dense_vec),
                using=DENSE,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                search_params=qm.SearchParams(
                    quantization=qm.QuantizationSearchParams(rescore=True, oversampling=2.0)
                ),
            )

        results: list[RetrievedChunk] = []
        for point in response.points:
            try:
                results.append(
                    RetrievedChunk(chunk=Chunk(**point.payload), score=float(point.score))
                )
            except Exception as exc:  # payload from an older schema_version
                logger.warning("Skipping malformed point %s: %s", point.id, exc)
        return results

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


def build_filter(
    scheme_names: list[str] | None = None,
    sections: list[str] | None = None,
    state: str | None = None,
    level: str | None = None,
    tags: list[str] | None = None,
    doc_ids: list[str] | None = None,
) -> qm.Filter | None:
    """Translate agent intent into a Qdrant filter.

    This is why the metadata schema matters: the Eligibility agent passes
    sections=["eligibility"] and searches 1/8th of the corpus with far less
    noise, rather than hoping cosine similarity separates eligibility prose
    from benefits prose.
    """
    must: list[qm.Condition] = []

    if scheme_names:
        must.append(qm.FieldCondition(key="scheme_name", match=qm.MatchAny(any=scheme_names)))
    if sections:
        must.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=sections)))
    if tags:
        must.append(qm.FieldCondition(key="tags", match=qm.MatchAny(any=tags)))
    if doc_ids:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids)))
    if level:
        must.append(qm.FieldCondition(key="level", match=qm.MatchValue(value=level)))
    if state:
        # Central schemes apply everywhere, so a state query must also return them.
        must.append(
            qm.Filter(
                should=[
                    qm.FieldCondition(key="state", match=qm.MatchValue(value=state)),
                    qm.FieldCondition(key="level", match=qm.MatchValue(value="central")),
                ]
            )
        )

    return qm.Filter(must=must) if must else None


def build_profile_filter(
    profile,
    *,
    sections: list[str] | None = None,
    state: str | None = None,
    include_institutions: bool = False,
    strict: bool = True,
) -> qm.Filter | None:
    """Translate a UserProfile into "schemes this person could qualify for".

    Each axis is a two-sided test rather than a match:

      age           scheme.age_min <= user.age <= scheme.age_max
      income        user.income <= scheme.income_max
      land          user.land   <= scheme.land_max_ha
      occupation    user.occupation IN scheme.occupations
      category      user.category   IN scheme.categories

    Unconstrained axes are stored as the sentinel "any" (or the numeric
    sentinels in facets.py), so a scheme with no age rule matches every age
    without a special case, and a scheme restricted to SC applicants is
    excluded for a general-category user instead of being ranked low.

    `strict=False` drops EVERY facet condition, keeping only section and
    state. That is the pre-enrichment fallback: before scripts/enrich.py has
    run, `facets` is absent on all 12,811 points, so any condition naming a
    facet field -- numeric ones included -- matches nothing and discovery
    returns an empty list rather than degrading. Dropping them entirely lets
    semantic search over the right sections do the work instead.
    """
    from .facets import ANY

    must: list[qm.Condition] = []

    if sections:
        must.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=sections)))

    if strict and not include_institutions:
        must.append(
            qm.FieldCondition(
                key="facets.beneficiary_type",
                match=qm.MatchAny(any=["individual", "both"]),
            )
        )

    if strict and profile.age is not None:
        must.append(qm.FieldCondition(key="facets.age_min", range=qm.Range(lte=profile.age)))
        must.append(qm.FieldCondition(key="facets.age_max", range=qm.Range(gte=profile.age)))

    if strict and profile.annual_income is not None:
        must.append(
            qm.FieldCondition(
                key="facets.income_max", range=qm.Range(gte=profile.annual_income)
            )
        )

    if strict and profile.landholding_ha is not None:
        must.append(
            qm.FieldCondition(
                key="facets.land_max_ha", range=qm.Range(gte=profile.landholding_ha)
            )
        )

    if strict:
        if profile.occupation:
            must.append(
                qm.FieldCondition(
                    key="facets.occupations",
                    match=qm.MatchAny(any=[profile.occupation, ANY]),
                )
            )
        if profile.category:
            must.append(
                qm.FieldCondition(
                    key="facets.categories", match=qm.MatchAny(any=[profile.category, ANY])
                )
            )
        if profile.gender:
            must.append(
                qm.FieldCondition(
                    key="facets.genders", match=qm.MatchAny(any=[profile.gender, ANY])
                )
            )
        if profile.education_level:
            must.append(
                qm.FieldCondition(
                    key="facets.education_levels",
                    match=qm.MatchAny(any=[profile.education_level, ANY]),
                )
            )
        if profile.special_statuses:
            # A scheme demanding a status the user lacks is excluded; a scheme
            # demanding nothing carries ANY and matches regardless.
            must.append(
                qm.FieldCondition(
                    key="facets.special_statuses",
                    match=qm.MatchAny(any=[*profile.special_statuses, ANY]),
                )
            )

    target_state = state or profile.state
    if target_state:
        must.append(
            qm.Filter(
                should=[
                    qm.FieldCondition(key="state", match=qm.MatchValue(value=target_state)),
                    qm.FieldCondition(key="level", match=qm.MatchValue(value="central")),
                ]
            )
        )

    return qm.Filter(must=must) if must else None


@lru_cache
def get_vectorstore() -> VectorStore:
    return VectorStore()
