from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .crawler import MAX_CONCURRENCY, CrawlConfig, DMXCrawler
from .storage import Storage
from .sentiment import (
    DEFAULT_BATCH_SIZE as SENTIMENT_BATCH_SIZE,
    MAX_BATCH_SIZE as SENTIMENT_MAX_BATCH_SIZE,
    GeminiSentimentLabeler,
    SentimentWorkspace,
)
from .toxicity import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    MAX_BATCH_SIZE,
    GeminiToxicityLabeler,
    ToxicityWorkspace,
    run_labeling,
)


DEFAULT_DB = Path("data/raw/dmx_reviews.sqlite3")
DEFAULT_EXPORT = Path("data/raw/export")
DEFAULT_TOXICITY_WORKSPACE = Path("data/processed/toxicity/toxicity.sqlite3")
DEFAULT_TOXICITY_EXPORT = Path("data/processed/toxicity/export")
DEFAULT_SENTIMENT_WORKSPACE = Path("data/processed/sentiment/sentiment.sqlite3")
DEFAULT_SENTIMENT_EXPORT = Path("data/processed/sentiment/export")
DEFAULT_DMX_TOXICITY_10K_EXPORT = Path(
    "data/processed/toxicity/dmx_toxicity_10k"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmx-crawl",
        description="Playwright crawler for public Dien May Xanh product reviews",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="Seed product URLs from the official sitemap")
    seed.add_argument("--since-year", type=int, default=2020)
    seed.add_argument("--until-year", type=int)
    seed.add_argument("--sitemap-limit", type=int)
    seed.add_argument("--oldest-first", action="store_true")

    crawl = sub.add_parser("crawl", help="Crawl queued product pages and reviews")
    crawl.add_argument("--target-reviews", type=int, default=10_000)
    crawl.add_argument("--max-products", type=int, default=2_000)
    crawl.add_argument("--concurrency", type=int, default=3)
    crawl.add_argument("--min-delay", type=float, default=0.8)
    crawl.add_argument("--max-delay", type=float, default=1.8)
    crawl.add_argument("--max-attempts", type=int, default=3)
    crawl.add_argument("--max-review-pages", type=int, default=500)
    crawl.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reset terminal errors and exhausted pending URLs before crawling",
    )
    crawl.add_argument("--show-browser", action="store_true")

    run = sub.add_parser("run", help="Seed then crawl in one command")
    run.add_argument("--since-year", type=int, default=2020)
    run.add_argument("--until-year", type=int)
    run.add_argument("--sitemap-limit", type=int)
    run.add_argument("--oldest-first", action="store_true")
    run.add_argument("--target-reviews", type=int, default=10_000)
    run.add_argument("--max-products", type=int, default=2_000)
    run.add_argument("--concurrency", type=int, default=3)
    run.add_argument("--min-delay", type=float, default=0.8)
    run.add_argument("--max-delay", type=float, default=1.8)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--max-review-pages", type=int, default=500)
    run.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reset terminal errors and exhausted pending URLs before crawling",
    )
    run.add_argument("--show-browser", action="store_true")

    export = sub.add_parser("export", help="Export canonical SQLite rows to JSONL and CSV")
    export.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT)
    sub.add_parser("stats", help="Print crawl and dataset statistics")
    sub.add_parser("validate", help="Validate keys, text, ratings, and product references")
    sub.add_parser("retry-errors", help="Return terminal error URLs to the pending queue")

    prepare_toxicity = sub.add_parser(
        "prepare-toxicity",
        help="Clean and deduplicate reviews for binary toxicity labeling",
    )
    prepare_toxicity.add_argument(
        "--workspace", type=Path, default=DEFAULT_TOXICITY_WORKSPACE
    )
    prepare_toxicity.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing toxicity workspace and its labels",
    )

    sync_toxicity = sub.add_parser(
        "sync-toxicity",
        help="Append newly crawled reviews without deleting existing labels",
    )
    sync_toxicity.add_argument(
        "--workspace", type=Path, default=DEFAULT_TOXICITY_WORKSPACE
    )

    label_toxicity = sub.add_parser(
        "label-toxicity",
        help="Label prepared reviews with Gemini in resumable batches",
    )
    label_toxicity.add_argument(
        "--workspace", type=Path, default=DEFAULT_TOXICITY_WORKSPACE
    )
    label_toxicity.add_argument("--model", default=DEFAULT_MODEL)
    label_toxicity.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    label_toxicity.add_argument("--max-attempts", type=int, default=3)
    label_toxicity.add_argument("--max-batches", type=int)
    label_toxicity.add_argument("--timeout-seconds", type=int, default=900)
    label_toxicity.add_argument("--retry-delay", type=float, default=2.0)

    toxicity_stats = sub.add_parser(
        "toxicity-stats", help="Show toxicity labeling progress"
    )
    toxicity_stats.add_argument(
        "--workspace", type=Path, default=DEFAULT_TOXICITY_WORKSPACE
    )
    toxicity_stats.add_argument("--max-attempts", type=int, default=3)

    export_toxicity = sub.add_parser(
        "export-toxicity", help="Export cleaned and toxicity-labeled datasets"
    )
    export_toxicity.add_argument(
        "--workspace", type=Path, default=DEFAULT_TOXICITY_WORKSPACE
    )
    export_toxicity.add_argument(
        "--output-dir", type=Path, default=DEFAULT_TOXICITY_EXPORT
    )

    export_dmx_10k = sub.add_parser(
        "export-dmx-toxicity-10k",
        help="Export the independently created DMX-Toxicity10K dataset",
    )
    export_dmx_10k.add_argument(
        "--workspace", type=Path, default=DEFAULT_TOXICITY_WORKSPACE
    )
    export_dmx_10k.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DMX_TOXICITY_10K_EXPORT
    )
    export_dmx_10k.add_argument("--toxic-count", type=int, default=1_000)
    export_dmx_10k.add_argument("--non-toxic-count", type=int, default=9_000)
    export_dmx_10k.add_argument("--seed", default="dmx-toxicity10k-v1")

    for command, help_text in (
        ("prepare-sentiment", "Clean raw reviews for three-class satisfaction labeling"),
        ("sync-sentiment", "Append new reviews to the satisfaction workspace"),
        ("label-sentiment", "Label satisfaction with Gemini in batches of up to 100"),
        ("sentiment-stats", "Show satisfaction labeling progress"),
        ("export-sentiment", "Export satisfaction labels and unique training rows"),
    ):
        sentiment = sub.add_parser(command, help=help_text)
        sentiment.add_argument(
            "--workspace", type=Path, default=DEFAULT_SENTIMENT_WORKSPACE
        )
        if command == "prepare-sentiment":
            sentiment.add_argument(
                "--force", action="store_true",
                help="Replace the sentiment workspace and its labels",
            )
        elif command == "label-sentiment":
            sentiment.add_argument("--model", default=DEFAULT_MODEL)
            sentiment.add_argument(
                "--batch-size", type=int, default=SENTIMENT_BATCH_SIZE
            )
            sentiment.add_argument("--max-attempts", type=int, default=3)
            sentiment.add_argument("--max-batches", type=int)
            sentiment.add_argument("--timeout-seconds", type=int, default=900)
            sentiment.add_argument("--retry-delay", type=float, default=2.0)
        elif command == "sentiment-stats":
            sentiment.add_argument("--max-attempts", type=int, default=3)
        elif command == "export-sentiment":
            sentiment.add_argument(
                "--output-dir", type=Path, default=DEFAULT_SENTIMENT_EXPORT
            )

    curate = sub.add_parser(
        "curate-sentiment", help="Audit suspect labels, deduplicate, split and ZIP a dataset"
    )
    curate.add_argument("--workspace", type=Path, default=DEFAULT_SENTIMENT_WORKSPACE)
    curate.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/sentiment/DMX-Sentiment-v1"),
    )
    curate.add_argument("--seed", type=int, default=42)
    return parser


async def run_network_command(args: argparse.Namespace, storage: Storage) -> None:
    config = CrawlConfig(
        database=str(args.database),
        concurrency=getattr(args, "concurrency", 3),
        min_delay=getattr(args, "min_delay", 0.8),
        max_delay=getattr(args, "max_delay", 1.8),
        max_attempts=getattr(args, "max_attempts", 3),
        max_review_pages=getattr(args, "max_review_pages", 500),
        target_reviews=getattr(args, "target_reviews", 10_000),
        max_products=getattr(args, "max_products", 2_000),
        headless=not getattr(args, "show_browser", False),
    )
    async with DMXCrawler(storage, config) as crawler:
        if args.command in {"seed", "run"}:
            changed = await crawler.seed(
                since_year=args.since_year,
                until_year=args.until_year,
                sitemap_limit=args.sitemap_limit,
                newest_first=not args.oldest_first,
            )
            print(f"[seed] database changes: {changed}")
        if args.command in {"crawl", "run"}:
            recovered = storage.recover_in_progress()
            if recovered:
                print(f"[resume] recovered interrupted URLs: {recovered}")
            if args.retry_errors:
                reset_errors = storage.reset_errors()
                reset_exhausted = storage.reset_exhausted_pending(args.max_attempts)
                print(
                    f"[resume] reset terminal errors: {reset_errors}; "
                    f"exhausted pending: {reset_exhausted}"
                )
            await crawler.crawl()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args()
    if args.command == "curate-sentiment":
        from .curation import build_release

        print(json.dumps(
            build_release(args.workspace, args.output_dir, seed=args.seed),
            ensure_ascii=False, indent=2,
        ))
        return
    is_sentiment = args.command in {
        "prepare-sentiment", "sync-sentiment", "label-sentiment",
        "sentiment-stats", "export-sentiment",
    }
    workspace_class = SentimentWorkspace if is_sentiment else ToxicityWorkspace
    if args.command in {"prepare-toxicity", "prepare-sentiment"}:
        if args.workspace.resolve() == args.database.resolve():
            raise SystemExit("--workspace must be different from the raw database")
        protected_workspace = (
            DEFAULT_TOXICITY_WORKSPACE if is_sentiment else DEFAULT_SENTIMENT_WORKSPACE
        )
        if args.workspace.resolve() == protected_workspace.resolve():
            raise SystemExit("Use a separate workspace for each labeling task")
        result = workspace_class.prepare(
            args.database, args.workspace, force=args.force
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command in {
        "sync-toxicity",
        "label-toxicity",
        "toxicity-stats",
        "export-toxicity",
        "export-dmx-toxicity-10k",
        "sync-sentiment", "label-sentiment", "sentiment-stats", "export-sentiment",
    }:
        if not args.workspace.exists():
            raise SystemExit(
                f"workspace not found: {args.workspace}; run "
                f"prepare-{'sentiment' if is_sentiment else 'toxicity'} first"
            )
        workspace = workspace_class(args.workspace)
        try:
            if args.command in {"sync-toxicity", "sync-sentiment"}:
                print(
                    json.dumps(
                        workspace.sync_from_source(args.database),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            elif args.command in {"toxicity-stats", "sentiment-stats"}:
                print(
                    json.dumps(
                        workspace.stats(max_attempts=args.max_attempts),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            elif args.command in {"export-toxicity", "export-sentiment"}:
                paths = workspace.export(args.output_dir)
                print("\n".join(f"{name}: {path}" for name, path in paths.items()))
            elif args.command == "export-dmx-toxicity-10k":
                print(
                    json.dumps(
                        workspace.export_dmx_toxicity_10k(
                            args.output_dir,
                            toxic_count=args.toxic_count,
                            non_toxic_count=args.non_toxic_count,
                            seed=args.seed,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                batch_limit = SENTIMENT_MAX_BATCH_SIZE if is_sentiment else MAX_BATCH_SIZE
                if not 1 <= args.batch_size <= batch_limit:
                    raise SystemExit(
                        f"--batch-size must be between 1 and {batch_limit}"
                    )
                if args.max_batches is not None and args.max_batches < 1:
                    raise SystemExit("--max-batches must be >= 1")
                api_key = os.environ.get("GEMINI_API_KEY", "")
                if not api_key:
                    raise SystemExit(
                        "GEMINI_API_KEY is not set; configure it in the current shell"
                    )
                labeler_class = (
                    GeminiSentimentLabeler if is_sentiment else GeminiToxicityLabeler
                )
                labeler = labeler_class(
                    api_key=api_key,
                    model=args.model,
                    timeout_seconds=args.timeout_seconds,
                )
                try:
                    result = run_labeling(
                        workspace,
                        labeler,
                        batch_size=args.batch_size,
                        max_attempts=args.max_attempts,
                        max_batches=args.max_batches,
                        retry_delay_seconds=args.retry_delay,
                    )
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                finally:
                    labeler.close()
        finally:
            workspace.close()
        return

    storage = Storage(args.database)
    try:
        if args.command in {"seed", "crawl", "run"}:
            if (
                hasattr(args, "concurrency")
                and not 1 <= args.concurrency <= MAX_CONCURRENCY
            ):
                raise SystemExit(
                    f"--concurrency must be between 1 and {MAX_CONCURRENCY}"
                )
            if hasattr(args, "min_delay") and args.min_delay < 0:
                raise SystemExit("--min-delay must be >= 0")
            if hasattr(args, "max_delay") and args.max_delay < args.min_delay:
                raise SystemExit("--max-delay must be >= --min-delay")
            asyncio.run(run_network_command(args, storage))
        elif args.command == "export":
            paths = storage.export(args.output_dir)
            print("\n".join(str(path) for path in paths))
        elif args.command == "stats":
            print(json.dumps(storage.stats(), ensure_ascii=False, indent=2))
        elif args.command == "validate":
            result = storage.validate()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result["valid"]:
                raise SystemExit(1)
        elif args.command == "retry-errors":
            print(f"reset URLs: {storage.reset_errors()}")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
