"""Smoke test: exercises collection creation, idempotent upsert, hybrid RRF
search and payload filtering against an in-memory Qdrant. No GPU, no server.

    python -m tests.test_vectorstore_smoke
"""

import random
import sys

from qdrant_client import QdrantClient

from schemes_rag.models import Chunk, Section, SchemeLevel
from schemes_rag.vectorstore import VectorStore, build_filter, chunk_point_id

random.seed(7)
DIM = 1024


def fake_dense(seed: int) -> list[float]:
    rnd = random.Random(seed)
    return [rnd.uniform(-1, 1) for _ in range(DIM)]


def fake_sparse(seed: int) -> dict[int, float]:
    rnd = random.Random(seed)
    return {rnd.randint(0, 250_000): rnd.uniform(0.1, 1.0) for _ in range(12)}


def make_chunk(i: int, section: Section, scheme: str, state: str | None = None) -> Chunk:
    return Chunk(
        doc_id=f"doc-{scheme}",
        chunk_index=i,
        text=f"{scheme} {section.value} clause {i}",
        token_count=50,
        doc_name=scheme,
        scheme_name=scheme,
        scheme_slug=scheme.lower().replace(" ", "-"),
        ministry="Ministry of Agriculture",
        level=SchemeLevel.STATE if state else SchemeLevel.CENTRAL,
        state=state,
        tags=["farmer"],
        section=section,
        section_title=section.value.replace("_", " ").title(),
        source_url=f"https://www.myscheme.gov.in/schemes/{scheme.lower()}",
    )


def main() -> int:
    store = VectorStore(client=QdrantClient(":memory:"))
    store.ensure_collection(recreate=True)
    print("collection created")

    chunks = [
        make_chunk(0, Section.ELIGIBILITY, "PM-KISAN"),
        make_chunk(1, Section.BENEFITS, "PM-KISAN"),
        make_chunk(2, Section.ELIGIBILITY, "Rythu Bandhu", state="Telangana"),
        make_chunk(3, Section.DOCUMENTS_REQUIRED, "Rythu Bandhu", state="Telangana"),
    ]
    dense = [fake_dense(i) for i in range(len(chunks))]
    sparse = [fake_sparse(i) for i in range(len(chunks))]

    n = store.upsert_chunks(chunks, dense, sparse)
    assert n == 4, n
    print(f"upserted {n}")

    # idempotency: same chunks again must not grow the collection
    store.upsert_chunks(chunks, dense, sparse)
    total = store.count()
    assert total == 4, f"expected 4 after re-upsert, got {total}"
    print(f"idempotent re-upsert verified (count still {total})")
    assert chunk_point_id("doc-PM-KISAN", 0) == chunk_point_id("doc-PM-KISAN", 0)

    # dense-only
    hits = store.search(fake_dense(0), limit=4, hybrid=False)
    assert len(hits) == 4, len(hits)
    print(f"dense search -> {len(hits)} hits, top score {hits[0].score:.4f}")

    # hybrid RRF
    hits = store.search(fake_dense(0), sparse_vec=fake_sparse(0), limit=4, hybrid=True)
    assert len(hits) == 4, len(hits)
    print(f"hybrid RRF search -> {len(hits)} hits, top score {hits[0].score:.4f}")

    # section filter: the move that powers the Eligibility agent
    flt = build_filter(sections=[Section.ELIGIBILITY.value])
    hits = store.search(fake_dense(0), sparse_vec=fake_sparse(0), limit=10, query_filter=flt)
    assert len(hits) == 2, [h.chunk.section for h in hits]
    assert all(h.chunk.section == Section.ELIGIBILITY for h in hits)
    print(f"section filter -> {len(hits)} eligibility chunks only")

    # state filter must also surface central schemes
    flt = build_filter(state="Telangana")
    hits = store.search(fake_dense(0), limit=10, query_filter=flt, hybrid=False)
    names = {h.chunk.scheme_name for h in hits}
    assert names == {"PM-KISAN", "Rythu Bandhu"}, names
    print(f"state filter -> {sorted(names)} (central schemes included)")

    # scheme filter: Comparison agent fetching two named schemes
    flt = build_filter(scheme_names=["PM-KISAN"], sections=[Section.BENEFITS.value])
    hits = store.search(fake_dense(1), limit=10, query_filter=flt, hybrid=False)
    assert len(hits) == 1 and hits[0].chunk.scheme_name == "PM-KISAN"
    print(f"scheme+section filter -> {hits[0].chunk.locator()}")

    # citations
    print(f"locator format -> {chunks[0].locator()!r}")
    print(f"embed_text  -> {chunks[0].embed_text[:70]!r}")

    # delete-by-doc for clean re-ingest
    store.delete_document("doc-PM-KISAN")
    assert store.count() == 2, store.count()
    print("delete_document -> 2 remaining")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
