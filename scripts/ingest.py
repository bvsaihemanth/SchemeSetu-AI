"""Ingest CLI.

    # see what the corpus contains, without embedding anything
    python scripts/ingest.py --corpus "C:\\Users\\bomma\\Documents\\RAG\\myscheme-ai\\schemes_txt" --dry-run

    # first real run
    python scripts/ingest.py --corpus "...\\schemes_txt" --recreate

    # after editing a few files: only changed docs are re-embedded
    python scripts/ingest.py --corpus "...\\schemes_txt"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schemes_rag.config import get_settings  # noqa: E402
from schemes_rag.ingest.pipeline import run_ingest  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest myScheme documents into Qdrant")
    p.add_argument("--corpus", help="Directory of .txt scheme files")
    p.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection")
    p.add_argument("--force", action="store_true", help="Re-embed even unchanged documents")
    p.add_argument("--limit", type=int, help="Stop after N documents (smoke testing)")
    p.add_argument("--batch-size", type=int, default=16, help="Documents per flush")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only: no embedding, no Qdrant writes",
    )
    args = p.parse_args()

    s = get_settings()
    logging.basicConfig(
        level=s.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.recreate and not args.dry_run:
        target = s.qdrant_local_path if s.qdrant_mode == "local" else s.qdrant_url
        reply = input(f"Delete and rebuild collection {s.qdrant_collection!r} at {target}? [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    stats = run_ingest(
        corpus_dir=args.corpus,
        recreate=args.recreate,
        force=args.force,
        doc_batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 64)
    print("INGEST SUMMARY" + ("  (DRY RUN - nothing written)" if args.dry_run else ""))
    print("=" * 64)
    for key in (
        "documents_seen",
        "documents_indexed",
        "documents_skipped",
        "chunks_indexed",
        "collection_count",
        "errors",
        "index_built",
        "elapsed_s",
    ):
        if key in stats:
            print(f"  {key:20} {stats[key]}")

    if stats["documents_indexed"]:
        per = stats["chunks_indexed"] / stats["documents_indexed"]
        print(f"  {'chunks_per_doc':20} {per:.1f}")

    total_docs = stats["documents_seen"] or 1
    print("\n  sections found (documents carrying each):")
    for name, count in sorted(stats["sections_found"].items(), key=lambda kv: -kv[1]):
        gap = total_docs - count
        flag = f"   <- {gap} doc(s) missing" if gap else ""
        print(f"    {name:24} {count:5}{flag}")

    if stats.get("faq_pairs_total"):
        per = stats["faq_pairs_total"] / total_docs
        print(f"\n  faq pairs total        {stats['faq_pairs_total']} ({per:.1f} per doc)")

    if stats.get("chunks_by_section"):
        print("\n  chunks by section:")
        tot = sum(stats["chunks_by_section"].values()) or 1
        for name, count in sorted(stats["chunks_by_section"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:24} {count:6}  {100 * count / tot:5.1f}%")

    si = stats.get("scheme_index")
    if si:
        print("\n  scheme alias index:")
        print(f"    schemes                {si['schemes']}")
        print(f"    aliases                {si['aliases']}")
        print(f"    ambiguous aliases      {si['ambiguous_aliases']}")
        print(f"    schemes without slug   {si['schemes_without_slug']}")
        print(f"    path                   {si['path']}")

    missing = stats.get("docs_missing_sections") or {}
    if missing:
        print(f"\n  DOCUMENTS WITH MISSING SECTIONS ({len(missing)}):")
        for doc_name, gaps in list(missing.items())[:25]:
            print(f"    {doc_name[:58]:60} missing: {', '.join(gaps)}")
        if len(missing) > 25:
            print(f"    ... and {len(missing) - 25} more (see the JSON report)")

    if stats["unmapped_headings"]:
        print("\n  UNMAPPED HEADINGS (add to SECTION_ALIASES in extract.py):")
        for title, count in sorted(stats["unmapped_headings"].items(), key=lambda kv: -kv[1]):
            print(f"    {title!r:40} {count}")
    else:
        print("\n  all headings mapped to known sections")

    Path(s.processed_dir).mkdir(parents=True, exist_ok=True)
    report = Path(s.processed_dir) / "last_ingest_report.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if stats.get("index_built") is False:
        print("\n  WARNING: index build did not confirm green. Check GET /stats")
        print("           optimizer_status before trusting query latency.")

    print(f"\n  report -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
