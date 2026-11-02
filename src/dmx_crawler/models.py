from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Product:
    product_id: str
    url: str
    title: str
    category: str
    site_id: int = 2
    object_type: int = 2
    reported_review_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Review:
    source_review_id: str
    product_id: str
    product_url: str
    product_title: str
    category: str
    rating: int | None
    text_raw: str
    verified_purchase: bool
    helpful_count: int
    usage_duration: str | None
    source_page: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

