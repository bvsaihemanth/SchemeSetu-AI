"""Offline eligibility-facet extraction.

Run ONCE after ingest. Turns eligibility prose into structured fields so
"suggest schemes for me" becomes a filter instead of a guess.

    # see what it would do, on 20 schemes, without touching Qdrant
    python scripts\\enrich.py --limit 20 --dry-run

    # the real pass (~1 hour for 725 on a 4 GB GPU; resumable)
    python scripts\\enrich.py

    # re-extract everything after changing the prompt or vocabulary
    python scripts\\enrich.py --force

Cost note: facets live in the point PAYLOAD, so this uses set_payload and
never re-embeds. The only expense is the LLM extraction, which is cached by
content hash -- interrupt it freely and run it again.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schemes_rag.config import get_settings  # noqa: E402
from schemes_rag.enrich import CACHE_NAME, run_enrichment  # noqa: E402
from schemes_rag.extract import iter_documents  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Extract structured eligibility facets")
    p.add_argument("--corpus", help="Directory of .txt scheme files (default: RAW_DIR)")
    p.add_argument("--limit", type=int, help="Stop after N schemes")
    p.add_argument("--force", action="store_true", help="Ignore cache, re-extract all")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Extract and cache, but do not write facets to Qdrant",
    )
    args = p.parse_args()

    s = get_settings()
    logging.basicConfig(
        level=s.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    store = None
    if not args.dry_run:
        from schemes_rag.vectorstore import get_vectorstore

        store = get_vectorstore()
        if not store.health():
            print("Qdrant unreachable. Start it, or use --dry-run.")
            return 1
        if store.count() == 0:
            print("Collection is empty. Run scripts/ingest.py first.")
            return 1

    docs = iter_documents(args.corpus or s.raw_dir)
    stats = run_enrichment(
        docs, store, force=args.force, limit=args.limit, dry_run=args.dry_run
    )

    print("\n" + "=" * 64)
    print("ENRICHMENT" + ("  (DRY RUN - Qdrant untouched)" if args.dry_run else ""))
    print("=" * 64)
    for key in ("seen", "extracted", "cached", "applied", "unconstrained"):
        print(f"  {key:22} {stats[key]}")

    print("\n  extraction confidence:")
    for name, count in sorted(stats["confidence"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:20} {count}")

    print("\n  beneficiary type:")
    for name, count in sorted(stats["beneficiary_type"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:20} {count}")

    sizes = stats.get("axis_sizes") or {}
    if sizes.get("occupations"):
        print("\n  axis list sizes (large = the model copied the vocabulary):")
        for axis, values in sizes.items():
            if not values:
                continue
            over = sum(1 for v in values if v > 4)
            print(f"    {axis:20} min={min(values)} max={max(values)} "
                  f"mean={sum(values) / len(values):.1f}   >4 values: {over}")
        n_open = stats.get("open_occupations", 0)
        pct = 100 * n_open / max(1, stats["seen"])
        print(f"    occupations = ['any']: {n_open} ({pct:.0f}%)")

    print("\n  top occupations found:")
    top = sorted(stats["occupation_coverage"].items(), key=lambda kv: -kv[1])[:15]
    for name, count in top:
        print(f"    {name:28} {count}")

    if stats["unconstrained"]:
        pct = 100 * stats["unconstrained"] / max(1, stats["seen"])
        print(f"\n  NOTE: {stats['unconstrained']} scheme(s) ({pct:.0f}%) came back with no")
        print("        restrictions at all. Some genuinely have none, but a high")
        print("        share means the extractor is failing -- inspect a few in")
        print(f"        {Path(s.processed_dir) / CACHE_NAME}")

    report = Path(s.processed_dir) / "last_enrichment_report.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\n  cache  -> {Path(s.processed_dir) / CACHE_NAME}")
    print(f"  report -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
