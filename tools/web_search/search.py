#!/usr/bin/env python3
"""
Complete search pipeline wrapper.
Runs quick_scrape.py -> main_content_cleaner.py
Outputs structured JSON with filtered results.

Usage (CLI):
  python search.py --query "today hot news" --max 50 --workers 6 --out results.json
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

try:
    # Package mode (used by tools.tool_registry)
    from .quick_scrape import EnterpriseSearchEngine
    from .main_content_cleaner import process_results
except Exception:
    try:
        # Script mode fallback
        from quick_scrape import EnterpriseSearchEngine
        from main_content_cleaner import process_results
    except Exception as exc:
        raise ImportError(
            "Could not import from quick_scrape.py or main_content_cleaner.py: " + str(exc)
        )


def _ensure_utf8_stdio():
    """Ensure emoji-heavy debug logs do not fail on Windows code pages."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Best effort only; logging should continue even if reconfigure is unavailable.
        pass


def run_search(query: str, max_results: int = 100, workers: int = 8, max_retries: int = 3):
    """
    Run complete search pipeline: search -> clean -> filter.

    Retries are applied in two places:
    1) Raw search results are empty.
    2) Structured results are empty after cleaning/filtering.

    Args:
        query: Search query string
        max_results: Max results to fetch
        workers: Parallel workers
        max_retries: Max retry attempts per empty-result condition

    Returns:
        (structured_results, stats) tuple
    """
    _ensure_utf8_stdio()
    print(f"🔎 Web search initiated for: '{query}' (max results={max_results})")

    max_retries = max(0, int(max_retries))
    raw_retry_count = 0
    structured_retry_count = 0
    attempt = 0

    raw_results = []
    structured_results = []
    raw_chars = 0
    structured_chars = 0
    engine_stats = {}
    cleaner_stats = {
        "total_input": 0,
        "successful": 0,
        "failed": 0,
        "processed": 0,
    }

    while True:
        attempt += 1
        print(f"\nAttempt {attempt}: running search pipeline")

        # Use a fresh engine per attempt so retries do not carry over old in-memory results.
        engine = EnterpriseSearchEngine(max_workers=workers)
        raw_results = engine.execute_search(query, max_results)
        engine_stats = dict(engine.stats or {})

        # Calculate total characters in raw results
        raw_chars = sum(
            len(str(getattr(result, "main_content", "")))
            + len(str(getattr(result, "title", "")))
            + len(str(getattr(result, "url", "")))
            for result in raw_results
        )
        print(f"📈 Raw search results: {len(raw_results)} items, {raw_chars} total chars")

        # Retry path 1: no raw results.
        if not raw_results:
            if raw_retry_count < max_retries:
                raw_retry_count += 1
                print(
                    "⚠️  Search returned zero raw results. "
                    f"Retrying raw search ({raw_retry_count}/{max_retries})..."
                )
                continue

            print("⚠️  Raw search results remain zero after max retries.")
            structured_results = []
            structured_chars = 0
            break

        # Convert dataclass results to plain dicts.
        results_dicts = [asdict(result) for result in raw_results]

        # Phase 2: Clean and filter by extraction_status == "success".
        structured_results, cleaner_stats = process_results(results_dicts)

        # Calculate total characters in structured results
        structured_chars = sum(
            len(json.dumps(result, ensure_ascii=False))
            for result in structured_results
        )
        print(f"📈 Structured results: {len(structured_results)} items, {structured_chars} total chars")
        print(
            f"   Pipeline: {len(raw_results)} total raw results, "
            f"{len(structured_results)} successfully extracted"
        )

        # Success path: structured output exists.
        if structured_results:
            break

        # Retry path 2: raw exists but structured cleaned to zero.
        if structured_retry_count < max_retries:
            structured_retry_count += 1
            print(
                "⚠️  Structured results became zero after cleaning. "
                f"Retrying full pipeline ({structured_retry_count}/{max_retries})..."
            )
            continue

        print("⚠️  Structured results remain zero after max retries.")
        break

    combined_stats = {
        "search_engine": engine_stats,
        "cleaner": cleaner_stats,
        "retry": {
            "attempts": attempt,
            "max_retries": max_retries,
            "raw_retries": raw_retry_count,
            "structured_retries": structured_retry_count,
            "final_raw_results": len(raw_results),
            "final_structured_results": len(structured_results),
        },
    }

    return structured_results, combined_stats


def main():
    parser = argparse.ArgumentParser(description="Complete search pipeline: search -> clean -> filter")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--max", "-m", type=int, default=100, help="Max results")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Parallel workers")
    parser.add_argument(
        "--out",
        "-o",
        default="struct_format_results.json",
        help="Output structured JSON path",
    )

    args = parser.parse_args()

    structured_results, stats = run_search(args.query, max_results=args.max, workers=args.workers)

    output = {
        "query": args.query,
        "parameters": {"max_results": args.max, "workers": args.workers},
        "stats": stats,
        "structured_results": structured_results,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSEARCH PIPELINE COMPLETE")
    print(f"Query: {args.query}")
    print(f"Total results from search: {stats['search_engine']['total']}")
    print(f"Successfully extracted: {stats['cleaner']['successful']}")
    print(f"Failed (ignored): {stats['cleaner']['failed']}")
    print(f"Structured JSON: {out_path}")
    print(f"Execution time: {stats['search_engine']['execution_time']:.1f}s")


if __name__ == "__main__":
    main()
