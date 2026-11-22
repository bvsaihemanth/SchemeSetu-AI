"""Ingest orchestration: corpus directory -> indexed Qdrant collection.

Run as a CLI, never from the API process. Properties worth knowing:

- IDEMPOTENT. Point IDs are uuid5(doc_id, chunk_index), so re-running
  upserts in place instead of duplicating. Safe to re-run after a crash.
- RESUMABLE. A manifest of content hashes means an unchanged document is
  skipped on the next run. Re-ingesting one fixed file costs one file.
- STREAMING. Documents are processed in batches, so memory stays flat at
  725 docs or 72,500.
- FAILURE-TOLERANT. One unparseable file logs a warning; the run continues.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Iterator

from ..chunk import chunk_document
from ..config import get_settings
from ..embeddings import get_embedder
from ..extract import iter_documents
from ..models import Chunk, SchemeDoc
from ..scheme_index import INDEX_FILENAME, SchemeEntry, SchemeIndex, generate_aliases
from ..vectorstore import VectorStore, get_vectorstore

logger = logging.getLogger(__name__)

MANIFEST_NAME = "ingest_manifest.json"


class Manifest:
    """doc_id -> content_hash, so unchanged files are skipped."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Ignoring unreadable manifest: %s", exc)

    def unchanged(self, doc: SchemeDoc) -> bool:
        return self.data.get(doc.doc_id) == doc.content_hash

    def record(self, doc: SchemeDoc) -> None:
        self.data[doc.doc_id] = doc.content_hash

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


def _batched(it: Iterable, size: int) -> Iterator[list]:
    batch: list = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_and_store(chunks: list[Chunk], store: VectorStore) -> int:
    if not chunks:
        return 0
    embedder = get_embedder()
    # embed_text, not text: the scheme/section header is part of what gets
    # vectorised. See Chunk.embed_text for why.
    result = embedder.encode_documents([c.embed_text for c in chunks])
    return store.upsert_chunks(chunks, result.dense, result.sparse)


def run_ingest(
    corpus_dir: str | Path | None = None,
    *,
    recreate: bool = False,
    force: bool = False,
    doc_batch_size: int = 16,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    s = get_settings()
    corpus = Path(corpus_dir or s.raw_dir)
    started = time.perf_counter()

    # Build no client in dry-run: in local mode merely constructing one takes
    # an exclusive lock on the storage folder.
    store: VectorStore | None = None
    if not dry_run:
        store = get_vectorstore()
        store.ensure_collection(recreate=recreate)

    manifest_path = Path(s.processed_dir) / MANIFEST_NAME
    manifest = Manifest(manifest_path)
    if recreate:
        manifest.data = {}

    # Sections the myScheme layout is expected to carry. A document missing
    # one is not an error, but it is worth seeing: a scheme with no
    # eligibility section is invisible to the eligibility route.
    EXPECTED = ("details", "benefits", "eligibility", "application_process",
                "documents_required", "faq")

    stats = {
        "documents_seen": 0,
        "documents_indexed": 0,
        "documents_skipped": 0,
        "chunks_indexed": 0,
        "sections_found": {},
        "chunks_by_section": {},
        "faq_pairs_total": 0,
        "docs_missing_sections": {},
        "unmapped_headings": {},
        "no_faq_docs": [],
        "errors": 0,
    }

    # Built from EVERY document walked, including ones skipped as unchanged:
    # the alias index must cover the whole corpus or a resumed ingest would
    # silently shrink it and scheme-name queries would start missing.
    scheme_entries: list[SchemeEntry] = []

    def flush(docs: list[SchemeDoc]) -> None:
        chunks: list[Chunk] = []
        for doc in docs:
            doc_chunks = chunk_document(doc)
            if not doc_chunks:
                stats["errors"] += 1
                continue
            # Clear old chunks first: if a doc shrank, stale tail chunks from
            # a previous run would otherwise survive with no file behind them.
            if store is not None and manifest.data.get(doc.doc_id):
                store.delete_document(doc.doc_id)
            for c in doc_chunks:
                key = c.section.value
                stats["chunks_by_section"][key] = stats["chunks_by_section"].get(key, 0) + 1
            chunks.extend(doc_chunks)
            manifest.record(doc)
            stats["documents_indexed"] += 1

        if store is None:
            stats["chunks_indexed"] += len(chunks)
            return

        for sub in _batched(chunks, 64):
            stats["chunks_indexed"] += embed_and_store(sub, store)

    pending: list[SchemeDoc] = []
    for doc in iter_documents(corpus):
        stats["documents_seen"] += 1

        scheme_entries.append(
            SchemeEntry(
                doc_id=doc.doc_id,
                doc_name=doc.doc_name,
                scheme_name=doc.scheme_name,
                slug=doc.scheme_slug,
                source_url=doc.source_url,
                aliases=generate_aliases(doc.scheme_name, doc.scheme_slug),
            )
        )

        present: set[str] = set()
        for section in doc.sections:
            key = section.section.value
            present.add(key)
            stats["sections_found"][key] = stats["sections_found"].get(key, 0) + 1
            if key == "other":
                t = section.section_title
                stats["unmapped_headings"][t] = stats["unmapped_headings"].get(t, 0) + 1

        # FAQ content lives in doc.faqs, not doc.sections, so it must be
        # counted separately or it vanishes from the report entirely.
        if doc.faqs:
            present.add("faq")
            stats["sections_found"]["faq"] = stats["sections_found"].get("faq", 0) + 1
            stats["faq_pairs_total"] += len(doc.faqs)
        else:
            stats["no_faq_docs"].append(doc.doc_name)

        if missing := [k for k in EXPECTED if k not in present]:
            stats["docs_missing_sections"][doc.doc_name] = missing

        if not force and manifest.unchanged(doc):
            stats["documents_skipped"] += 1
            continue

        pending.append(doc)
        if len(pending) >= doc_batch_size:
            flush(pending)
            pending = []
            logger.info(
                "%s docs / %s chunks indexed",
                stats["documents_indexed"],
                stats["chunks_indexed"],
            )

        if limit and stats["documents_seen"] >= limit:
            break

    flush(pending)

    # The alias index is what makes "PM-KISAN" resolvable. Write it even on a
    # dry run: it costs nothing and lets you inspect resolution before
    # committing to an embedding pass.
    if scheme_entries:
        index = SchemeIndex(scheme_entries)
        index_path = Path(s.processed_dir) / INDEX_FILENAME
        index.save(index_path)
        stats["scheme_index"] = {
            "schemes": len(index),
            "aliases": len(index.alias_map),
            "ambiguous_aliases": len(index.ambiguous),
            "schemes_without_slug": sum(1 for e in scheme_entries if not e.slug),
            "path": str(index_path),
        }
        logger.info("Scheme index -> %s (%d schemes)", index_path, len(index))

    if store is not None:
        manifest.save()
        # Re-enable indexing and wait for the HNSW graph to be built. Skipping
        # this leaves every query on brute-force search, silently.
        logger.info("Building vector index...")
        indexed = store.finalize_index(wait=True)
        stats["index_built"] = indexed
        stats["collection_count"] = store.count()

    stats["elapsed_s"] = round(time.perf_counter() - started, 2)
    return stats
