from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from html import unescape
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from .models import Product, Review

_INT_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")


def _text(node) -> str:
    if node is None:
        return ""
    return _SPACE_RE.sub(" ", node.get_text(" ", strip=True)).strip()


def parse_sitemap_index(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    return [loc.text.strip() for loc in root.findall(".//{*}loc") if loc.text]


def parse_product_sitemap(xml_text: str) -> list[tuple[str, str | None]]:
    root = ET.fromstring(xml_text)
    rows: list[tuple[str, str | None]] = []
    for item in root.findall(".//{*}url"):
        loc = item.find("{*}loc")
        lastmod = item.find("{*}lastmod")
        if loc is not None and loc.text:
            rows.append((loc.text.strip(), lastmod.text.strip() if lastmod is not None and lastmod.text else None))
    return rows


def category_from_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[0] if len(parts) >= 2 else "unknown"


def _reported_count(soup: BeautifulSoup) -> int | None:
    node = soup.select_one(".point-alltimerate")
    if node is None:
        return None
    raw = _text(node).lower().replace(",", ".")
    number = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not number:
        return None
    value = float(number.group(1))
    if "k" in raw:
        value *= 1000
    return int(value)


def parse_product(html: str, url: str) -> Product | None:
    soup = BeautifulSoup(html, "lxml")
    rating = soup.select_one(".wrap_rating[data-objectid]")
    if rating is None:
        return None

    product_id = str(rating.get("data-objectid") or "").strip()
    if not product_id:
        return None

    title = unescape(str(rating.get("data-title") or "")).strip()
    if not title:
        heading = soup.select_one("h1")
        title = _text(heading)

    def as_int(name: str, default: int) -> int:
        try:
            return int(str(rating.get(name) or default))
        except ValueError:
            return default

    return Product(
        product_id=product_id,
        url=url,
        title=title,
        category=category_from_url(url),
        site_id=as_int("data-siteid", 2),
        object_type=as_int("data-objecttype", 2),
        reported_review_count=_reported_count(soup),
    )


def parse_reviews(html: str, product: Product, page: int) -> list[Review]:
    soup = BeautifulSoup(html, "lxml")
    reviews: list[Review] = []
    for item in soup.select("ul.comment-list > li.par"):
        raw_id = str(item.get("id") or "")
        review_id = raw_id.removeprefix("r-").strip()
        content = _text(item.select_one(".cmt-content .cmt-txt"))
        if not review_id or not content:
            continue

        stars = len(item.select(".cmt-top-star .iconcmt-starbuy"))
        rating = stars if 1 <= stars <= 5 else None
        helpful_node = item.select_one(".cmt-command .cmtl")
        helpful_raw = str(helpful_node.get("data-like") or "0") if helpful_node else "0"
        helpful_match = _INT_RE.search(helpful_raw)
        usage = _text(item.select_one(".cmt-command .cmtd")) or None

        reviews.append(
            Review(
                source_review_id=review_id,
                product_id=product.product_id,
                product_url=product.url,
                product_title=product.title,
                category=product.category,
                rating=rating,
                text_raw=content,
                verified_purchase=item.select_one(".confirm-buy") is not None,
                helpful_count=int(helpful_match.group(0)) if helpful_match else 0,
                usage_duration=usage,
                source_page=page,
            )
        )
    return reviews


def parse_product_jsonld(html: str) -> dict | None:
    """Optional fallback metadata reader retained for diagnostics."""
    soup = BeautifulSoup(html, "lxml")
    for node in soup.select('script[type="application/ld+json"]'):
        try:
            value = json.loads(node.string or "")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("@type") == "Product":
            return value
    return None

