from pathlib import Path

from dmx_crawler.models import Product, Review
from dmx_crawler.storage import Storage


def test_storage_deduplicates_and_validates(tmp_path: Path):
    storage = Storage(tmp_path / "reviews.sqlite3")
    try:
        product = Product("1", "https://example.test/tu-lanh/demo", "Demo", "tu-lanh")
        review = Review(
            source_review_id="r1",
            product_id="1",
            product_url=product.url,
            product_title=product.title,
            category=product.category,
            rating=5,
            text_raw="Tốt",
            verified_purchase=True,
            helpful_count=0,
            usage_duration=None,
            source_page=1,
        )
        storage.save_product(product)
        assert storage.save_reviews([review, review]) == 1
        result = storage.validate()
        assert result["total_rows"] == 1
        assert result["duplicate_review_ids"] == 0
        assert result["valid"] is True
    finally:
        storage.close()


def test_retry_becomes_terminal_after_max_attempts(tmp_path: Path):
    storage = Storage(tmp_path / "retry.sqlite3")
    try:
        url = "https://example.test/product"
        storage.seed_urls([(url, None)])
        assert storage.claim_urls(1, 2) == [url]
        assert storage.mark_error(url, "socket hang up", max_attempts=2) == "pending"
        assert storage.claim_urls(1, 2) == [url]
        assert storage.mark_error(url, "socket hang up", max_attempts=2) == "error"
        assert storage.stats()["queue"] == {"error": 1}
        assert storage.reset_errors() == 1
        assert storage.stats()["queue"] == {"pending": 1}
    finally:
        storage.close()


def test_opening_storage_does_not_mutate_in_progress_queue(tmp_path: Path):
    path = tmp_path / "resume.sqlite3"
    storage = Storage(path)
    try:
        url = "https://example.test/interrupted"
        storage.seed_urls([(url, None)])
        assert storage.claim_urls(1, 3) == [url]

        observer = Storage(path)
        try:
            assert observer.stats()["queue"] == {"in_progress": 1}
        finally:
            observer.close()

        assert storage.recover_in_progress() == 1
        row = storage.conn.execute(
            "SELECT status, attempts FROM crawl_queue WHERE url=?", (url,)
        ).fetchone()
        assert dict(row) == {"status": "pending", "attempts": 0}
    finally:
        storage.close()


def test_reset_exhausted_pending_repairs_legacy_rows(tmp_path: Path):
    storage = Storage(tmp_path / "legacy.sqlite3")
    try:
        url = "https://example.test/legacy"
        storage.seed_urls([(url, None)])
        storage.conn.execute(
            "UPDATE crawl_queue SET attempts=5 WHERE url=?", (url,)
        )
        storage.conn.commit()
        assert storage.claim_urls(1, 5) == []
        assert storage.reset_exhausted_pending(5) == 1
        assert storage.claim_urls(1, 5) == [url]
    finally:
        storage.close()
