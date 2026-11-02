from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import Product, Review


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS crawl_queue (
                url TEXT PRIMARY KEY,
                lastmod TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_queue_status
            ON crawl_queue(status, attempts, lastmod DESC);

            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                site_id INTEGER NOT NULL,
                object_type INTEGER NOT NULL,
                reported_review_count INTEGER,
                crawled_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reviews (
                source_review_id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL,
                product_url TEXT NOT NULL,
                product_title TEXT NOT NULL,
                category TEXT NOT NULL,
                rating INTEGER,
                text_raw TEXT NOT NULL,
                verified_purchase INTEGER NOT NULL,
                helpful_count INTEGER NOT NULL,
                usage_duration TEXT,
                source_page INTEGER NOT NULL,
                crawled_at TEXT NOT NULL,
                FOREIGN KEY(product_id) REFERENCES products(product_id)
            );

            CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
            """
        )
        self.conn.commit()

    def recover_in_progress(self) -> int:
        """Return work interrupted by a stopped crawler to the pending queue.

        The claim attempt is rolled back because the URL did not reach either
        mark_done or mark_error. This method must only be called when starting a
        crawler, never by read-only commands such as stats.
        """

        before = self.conn.total_changes
        self.conn.execute(
            """
            UPDATE crawl_queue
            SET status='pending',
                attempts=CASE WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                updated_at=?
            WHERE status='in_progress'
            """,
            (utc_now(),),
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def seed_urls(self, rows: Iterable[tuple[str, str | None]]) -> int:
        before = self.conn.total_changes
        now = utc_now()
        self.conn.executemany(
            """
            INSERT INTO crawl_queue(url, lastmod, status, attempts, updated_at)
            VALUES (?, ?, 'pending', 0, ?)
            ON CONFLICT(url) DO UPDATE SET
                lastmod=COALESCE(excluded.lastmod, crawl_queue.lastmod)
            """,
            ((url, lastmod, now) for url, lastmod in rows),
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def claim_urls(self, limit: int, max_attempts: int) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT url FROM crawl_queue
            WHERE status='pending' AND attempts < ?
            ORDER BY COALESCE(lastmod, '') DESC, url
            LIMIT ?
            """,
            (max_attempts, limit),
        ).fetchall()
        urls = [str(row["url"]) for row in rows]
        if urls:
            now = utc_now()
            self.conn.executemany(
                """
                UPDATE crawl_queue
                SET status='in_progress', attempts=attempts+1, updated_at=?
                WHERE url=?
                """,
                ((now, url) for url in urls),
            )
            self.conn.commit()
        return urls

    def mark_done(self, url: str) -> None:
        self.conn.execute(
            "UPDATE crawl_queue SET status='done', last_error=NULL, updated_at=? WHERE url=?",
            (utc_now(), url),
        )
        self.conn.commit()

    def mark_error(
        self,
        url: str,
        error: str,
        retryable: bool = True,
        max_attempts: int = 3,
    ) -> str:
        row = self.conn.execute(
            "SELECT attempts FROM crawl_queue WHERE url=?", (url,)
        ).fetchone()
        attempts = int(row["attempts"]) if row else max_attempts
        status = "pending" if retryable and attempts < max_attempts else "error"
        self.conn.execute(
            "UPDATE crawl_queue SET status=?, last_error=?, updated_at=? WHERE url=?",
            (status, error[:1000], utc_now(), url),
        )
        self.conn.commit()
        return status

    def save_product(self, product: Product) -> None:
        self.conn.execute(
            """
            INSERT INTO products(
                product_id, url, title, category, site_id, object_type,
                reported_review_count, crawled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                url=excluded.url,
                title=excluded.title,
                category=excluded.category,
                site_id=excluded.site_id,
                object_type=excluded.object_type,
                reported_review_count=excluded.reported_review_count,
                crawled_at=excluded.crawled_at
            """,
            (
                product.product_id,
                product.url,
                product.title,
                product.category,
                product.site_id,
                product.object_type,
                product.reported_review_count,
                utc_now(),
            ),
        )
        self.conn.commit()

    def save_reviews(self, reviews: Iterable[Review]) -> int:
        rows = list(reviews)
        if not rows:
            return 0
        before = self.conn.total_changes
        now = utc_now()
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO reviews(
                source_review_id, product_id, product_url, product_title,
                category, rating, text_raw, verified_purchase, helpful_count,
                usage_duration, source_page, crawled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    row.source_review_id,
                    row.product_id,
                    row.product_url,
                    row.product_title,
                    row.category,
                    row.rating,
                    row.text_raw,
                    int(row.verified_purchase),
                    row.helpful_count,
                    row.usage_duration,
                    row.source_page,
                    now,
                )
                for row in rows
            ),
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def count_reviews(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])

    def reset_errors(self) -> int:
        before = self.conn.total_changes
        self.conn.execute(
            """
            UPDATE crawl_queue
            SET status='pending', attempts=0, last_error=NULL, updated_at=?
            WHERE status='error'
            """,
            (utc_now(),),
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def reset_exhausted_pending(self, max_attempts: int) -> int:
        """Repair pending rows left unclaimable by an older crawler version."""

        before = self.conn.total_changes
        self.conn.execute(
            """
            UPDATE crawl_queue
            SET attempts=0, last_error=NULL, updated_at=?
            WHERE status='pending' AND attempts >= ?
            """,
            (utc_now(), max_attempts),
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def stats(self) -> dict:
        counts = {
            "products": int(self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]),
            "reviews": self.count_reviews(),
        }
        counts["queue"] = {
            str(row["status"]): int(row["n"])
            for row in self.conn.execute(
                "SELECT status, COUNT(*) AS n FROM crawl_queue GROUP BY status"
            )
        }
        counts["ratings"] = {
            str(row["rating"]): int(row["n"])
            for row in self.conn.execute(
                "SELECT rating, COUNT(*) AS n FROM reviews GROUP BY rating ORDER BY rating"
            )
        }
        counts["categories"] = {
            str(row["category"]): int(row["n"])
            for row in self.conn.execute(
                """
                SELECT category, COUNT(*) AS n FROM reviews
                GROUP BY category ORDER BY n DESC LIMIT 20
                """
            )
        }
        return counts

    def validate(self) -> dict:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT source_review_id) AS distinct_review_ids,
                SUM(CASE WHEN TRIM(text_raw) = '' THEN 1 ELSE 0 END) AS empty_texts,
                SUM(CASE WHEN rating NOT BETWEEN 1 AND 5 OR rating IS NULL THEN 1 ELSE 0 END)
                    AS invalid_ratings,
                SUM(CASE WHEN verified_purchase = 1 THEN 1 ELSE 0 END) AS verified_rows,
                COUNT(DISTINCT product_id) AS products_with_reviews
            FROM reviews
            """
        ).fetchone()
        orphan_count = int(
            self.conn.execute(
                """
                SELECT COUNT(*) FROM reviews r
                LEFT JOIN products p ON p.product_id = r.product_id
                WHERE p.product_id IS NULL
                """
            ).fetchone()[0]
        )
        result = dict(row)
        result["duplicate_review_ids"] = (
            int(result["total_rows"]) - int(result["distinct_review_ids"])
        )
        result["orphan_reviews"] = orphan_count
        result["valid"] = all(
            int(result[key] or 0) == 0
            for key in (
                "duplicate_review_ids",
                "empty_texts",
                "invalid_ratings",
                "orphan_reviews",
            )
        )
        return result

    def export(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "reviews_raw.jsonl"
        csv_path = output_dir / "reviews_raw.csv"
        columns = [column.name for column in fields(Review)] + ["crawled_at"]
        query = "SELECT " + ", ".join(columns) + " FROM reviews ORDER BY source_review_id"

        with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl, csv_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            for row in self.conn.execute(query):
                item = dict(row)
                item["verified_purchase"] = bool(item["verified_purchase"])
                jsonl.write(json.dumps(item, ensure_ascii=False) + "\n")
                writer.writerow(item)
        return jsonl_path, csv_path

