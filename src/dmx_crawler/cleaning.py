from __future__ import annotations

import hashlib
import html
import re
import unicodedata


CLEANING_VERSION = "clean-v1"

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{6,}(?!\d)")
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PLACEHOLDER_RE = re.compile(r"<(?:URL|EMAIL|PHONE|NUMBER)>")


def _drop_invisible_and_control_characters(text: str) -> str:
    cleaned: list[str] = []
    for char in text:
        if char in "\t\r\n":
            cleaned.append(" ")
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cf", "Cs"}:
            continue
        cleaned.append(char)
    return "".join(cleaned)


def clean_review_text(value: str) -> str:
    """Minimally normalize a review while preserving Vietnamese and noisy language."""

    text = unicodedata.normalize("NFC", html.unescape(value or ""))
    text = _drop_invisible_and_control_characters(text)
    text = _URL_RE.sub("<URL>", text)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    text = _LONG_NUMBER_RE.sub("<NUMBER>", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def duplicate_key(cleaned_text: str) -> str:
    """Return a stable key for exact duplicates after safe normalization."""

    normalized = _WHITESPACE_RE.sub(" ", cleaned_text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def is_eligible_for_labeling(text: str) -> bool:
    """Exclude rows that contain no letters; keep short real-language reviews."""

    content_without_pii = _PLACEHOLDER_RE.sub("", text)
    return any(char.isalpha() for char in content_without_pii)


def is_low_information(text: str) -> bool:
    return word_count(text) <= 1
