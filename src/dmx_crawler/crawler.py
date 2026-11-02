from __future__ import annotations

import asyncio
import random
import re
import sys
from dataclasses import dataclass
from playwright.async_api import (
    APIRequestContext,
    APIResponse,
    Error as PlaywrightError,
    Playwright,
    async_playwright,
)

from .models import Product
from .parsers import parse_product, parse_product_sitemap, parse_reviews, parse_sitemap_index
from .storage import Storage

BASE_URL = "https://www.dienmayxanh.com"
PRODUCT_SITEMAP = f"{BASE_URL}/newsitemap/sitemap-product"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36 "
    "DMXResearchCrawler/1.0"
)
MAX_CONCURRENCY = 32


class RetryableError(RuntimeError):
    pass


def short_error(exc: BaseException) -> str:
    """Keep logs single-line and avoid leaking Playwright call logs/cookies."""
    first_line = str(exc).splitlines()[0].strip()
    return first_line or type(exc).__name__


def log(message: str) -> None:
    """Print safely even when a Windows console uses a legacy code page."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        safe = message.encode(encoding, errors="backslashreplace").decode(encoding)
        print(safe, flush=True)


@dataclass(slots=True)
class CrawlConfig:
    database: str
    concurrency: int = 3
    min_delay: float = 0.8
    max_delay: float = 1.8
    navigation_timeout_ms: int = 45_000
    request_timeout_ms: int = 30_000
    max_attempts: int = 3
    max_review_pages: int = 500
    target_reviews: int = 10_000
    max_products: int = 2_000
    headless: bool = True


def sitemap_year(url: str) -> int | None:
    match = re.search(r"sitemap-product-(\d{4})-", url)
    return int(match.group(1)) if match else None


def sitemap_sort_key(url: str) -> tuple[int, int, int]:
    match = re.search(r"sitemap-product-(\d{4})-(\d+).*?[?&]page=(\d+)", url)
    if not match:
        return (0, 0, 0)
    return tuple(int(value) for value in match.groups())


class DMXCrawler:
    def __init__(self, storage: Storage, config: CrawlConfig):
        self.storage = storage
        self.config = config
        self.playwright: Playwright | None = None
        self.api: APIRequestContext | None = None
        self._write_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self.products_processed = 0

    async def __aenter__(self) -> "DMXCrawler":
        self.playwright = await async_playwright().start()
        self.api = await self.playwright.request.new_context(
            user_agent=USER_AGENT,
            extra_http_headers={"Accept-Language": "vi-VN,vi;q=0.9,en;q=0.5"},
            timeout=self.config.request_timeout_ms,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.api:
            await self.api.dispose()
        if self.playwright:
            await self.playwright.stop()

    async def _sleep(self) -> None:
        await asyncio.sleep(random.uniform(self.config.min_delay, self.config.max_delay))

    async def _request_text(self, url: str) -> str:
        assert self.api is not None
        await self._sleep()
        try:
            response = await self.api.get(
                url,
                headers={"Accept": "application/xml,text/xml,text/html;q=0.9,*/*;q=0.8"},
                timeout=self.config.request_timeout_ms,
            )
            return await self._response_text(response, url)
        except RetryableError:
            raise
        except PlaywrightError as exc:
            raise RetryableError(short_error(exc)) from exc

    async def _response_text(self, response: APIResponse, url: str) -> str:
        if response.status == 429 or response.status >= 500:
            raise RetryableError(f"HTTP {response.status}: {url}")
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status}: {url}")
        return await response.text()

    async def seed(
        self,
        since_year: int | None = None,
        until_year: int | None = None,
        sitemap_limit: int | None = None,
        newest_first: bool = True,
    ) -> int:
        index = parse_sitemap_index(await self._request_text(PRODUCT_SITEMAP))
        if since_year is not None:
            index = [url for url in index if (sitemap_year(url) or 0) >= since_year]
        if until_year is not None:
            index = [url for url in index if (sitemap_year(url) or 9999) <= until_year]
        index.sort(key=sitemap_sort_key, reverse=newest_first)
        if sitemap_limit is not None:
            index = index[:sitemap_limit]

        total = 0
        for position, sitemap_url in enumerate(index, 1):
            try:
                rows = parse_product_sitemap(await self._request_text(sitemap_url))
            except Exception as exc:
                log(f"[seed] skip {sitemap_url}: {short_error(exc)}")
                continue
            total += self.storage.seed_urls(rows)
            log(f"[seed] {position}/{len(index)}: {len(rows)} URLs ({sitemap_url})")
        return total

    async def _fetch_product(self, url: str) -> Product | None:
        try:
            html = await self._request_text(url)
        except RuntimeError as exc:
            if "HTTP 404" in str(exc) or "HTTP 410" in str(exc):
                return None
            raise
        return parse_product(html, url)

    async def _fetch_review_page(self, product: Product, page_number: int) -> str:
        assert self.api is not None
        await self._sleep()
        try:
            response = await self.api.post(
                f"{BASE_URL}/comment/PagingRating",
                form={
                    "siteId": str(product.site_id),
                    "productid": product.product_id,
                    "page": str(page_number),
                },
                headers={
                    "Accept": "text/html, */*; q=0.01",
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": product.url,
                },
                timeout=self.config.request_timeout_ms,
            )
            if response.status in {404, 410}:
                return ""
            return await self._response_text(response, product.url)
        except RetryableError:
            raise
        except PlaywrightError as exc:
            raise RetryableError(short_error(exc)) from exc

    async def _crawl_reviews(self, product: Product) -> int:
        if product.reported_review_count == 0:
            return 0

        inserted = 0
        seen_ids: set[str] = set()
        for page_number in range(1, self.config.max_review_pages + 1):
            html = await self._fetch_review_page(product, page_number)
            reviews = parse_reviews(html, product, page_number)
            fresh = [row for row in reviews if row.source_review_id not in seen_ids]
            if not fresh:
                break
            seen_ids.update(row.source_review_id for row in fresh)
            async with self._write_lock:
                inserted += self.storage.save_reviews(fresh)
            if (
                product.reported_review_count is not None
                and len(seen_ids) >= product.reported_review_count
            ):
                break
        return inserted

    async def _process_url(self, url: str) -> None:
        try:
            product = await self._fetch_product(url)
            if product is None:
                async with self._write_lock:
                    self.storage.mark_done(url)
                log(f"[skip] no public rating block: {url}")
                return

            async with self._write_lock:
                self.storage.save_product(product)
            inserted = await self._crawl_reviews(product)
            async with self._write_lock:
                self.storage.mark_done(url)
                self.products_processed += 1
                total = self.storage.count_reviews()
            log(
                f"[done] product={product.product_id} reviews=+{inserted} "
                f"total={total} url={url}"
            )
            if total >= self.config.target_reviews or self.products_processed >= self.config.max_products:
                self._stop.set()
        except RetryableError as exc:
            async with self._write_lock:
                status = self.storage.mark_error(
                    url,
                    short_error(exc),
                    retryable=True,
                    max_attempts=self.config.max_attempts,
                )
            log(f"[{status}] {url}: {short_error(exc)}")
        except Exception as exc:
            async with self._write_lock:
                status = self.storage.mark_error(
                    url,
                    f"{type(exc).__name__}: {short_error(exc)}",
                    retryable=True,
                    max_attempts=self.config.max_attempts,
                )
            log(f"[{status}] {url}: {type(exc).__name__}: {short_error(exc)}")

    async def _worker(self, worker_id: int) -> None:
        while not self._stop.is_set():
            async with self._write_lock:
                urls = self.storage.claim_urls(1, self.config.max_attempts)
            if not urls:
                return
            await self._process_url(urls[0])

    async def crawl(self) -> None:
        if self.storage.count_reviews() >= self.config.target_reviews:
            log("[crawl] target already reached")
            return
        workers = [
            asyncio.create_task(self._worker(worker_id), name=f"worker-{worker_id}")
            for worker_id in range(1, self.config.concurrency + 1)
        ]
        await asyncio.gather(*workers)
