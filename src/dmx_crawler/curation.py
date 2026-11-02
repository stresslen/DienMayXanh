"""Auditable sentiment curation, grouped splits, and offline dataset packaging."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .cleaning import clean_review_text, duplicate_key, is_eligible_for_labeling
from .sentiment import LABEL_NAMES, VALID_LABELS


VERSION = "sentiment-curation-v1"
LABEL_IDS = {"dissatisfied": 0, "neutral": 1, "satisfied": 2}
SPLITS = {"train": 0.8, "validation": 0.1, "test": 0.1}
FIELDS = ["id", "text", "label", "label_name", "product_id", "category", "group_id"]
_WORDS = re.compile(r"\w+", re.UNICODE)


def fold(text: str) -> str:
    value = text.casefold().replace("đ", "d")
    value = "".join(c for c in unicodedata.normalize("NFD", value) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value).strip()


def hits(patterns: tuple[str, ...], text: str) -> list[str]:
    result = []
    for pattern in patterns:
        for match in re.finditer(r"\b(?:" + pattern + r")\b", text):
            prefix = text[max(0, match.start() - 24):match.start()]
            if re.search(r"(?:khong|ko|chua|chang|cha)(?:\s+(?:he|con|bao gio|co))?\s*$", prefix):
                continue
            result.append(match.group())
    return result


EXPLICIT_NEGATIVE = (
    r"(?:khong|ko|chua) (?:hai long|ung y|vua y)",
    r"(?:rat |qua |that su )?that vong", r"(?:khong|ko) nen mua",
    r"dung (?:nen )?mua", r"hoi han", r"(?:rat|qua) te",
    r"(?:rat|qua) kem", r"lua dao", r"khong dang (?:tien|mua)",
)
FAULTS = (
    r"(?:khong|ko) (?:hoat dong|chay|len nguon|len lua|sac duoc|dung duoc)",
    r"(?:bi|da|lai|moi mua da) (?:hong|loi|chay|nut|vo|ro dien)",
    r"(?:hong|loi) (?:roi|lien|sau|lien tuc)", r"khong tot",
    r"ep khong kiet", r"(?:phai )?vat lai", r"can nhieu", r"khong ra nuoc",
    r"khong ben", r"chai pin", r"khong mat", r"khong nong",
    r"ro (?:nuoc|gas|ga|dien)", r"bao hanh nhieu lan", r"doi tra", r"doi may",
)
POSITIVE = (
    r"(?:rat |kha |van )?hai long", r"(?:rat )?ung y", r"(?:rat )?ung",
    r"san pham (?:rat )?tot", r"chat luong (?:rat )?tot",
    r"(?:su dung|dung|xai|chay) (?:rat )?(?:tot|on|em)",
    r"rat (?:dep|ben|tot|tien dung|tien loi)", r"gia (?:ca |thanh )?hop ly",
    r"nhiet tinh", r"than thien", r"tuyet voi", r"dang mua", r"nen mua",
)
NEGATIVE_WORDS = (r"kem", r"te", r"yeu", r"on ao", r"bat tien", r"kho chiu",
                  r"khong duoc", r"khong vua y", r"khong ung", r"hong", r"loi")
QUESTION = re.compile(r"\b(?:con hang khong|co hang khong|gia bao nhieu|bao nhieu tien|"
                      r"bao gio co hang|may nang bao nhieu|co ban le khong)\b")
MIXED_OR_UNCERTAIN = re.compile(
    r"\b(?:nhung|tuy nhien|co dieu|diem tru|khuyet diem|tiec|hoi han|khong|ko|"
    r"k|chua|on qua|keu to|keu rat to|tieng to|rung|tran|vang|ro|rot|de nga|"
    r"thieu sot|mat thoi gian|chua biet|hong|loi|kho|hoi|do ec|kem|te|yeu|"
    r"chan|that vong|kho chiu)\b"
)
SHORT_PRAISE = re.compile(
    r"(?:san pham |sp |may |hang |dich vu )?(?:(?:rat |kha )?"
    r"(?:tot|on|dep|ben|ok|okay|tuyet voi|hai long|ung y|tien dung|tien loi))"
    r"(?:[ ,.]+(?:rat )?(?:tot|on|dep|ben|tien dung|tien loi))*[ .!]*"
)


def review_label(text: str, label: str, rating: int | None) -> dict:
    """Stars flag candidates; only textual evidence can propose a correction."""
    normalized = fold(text)
    normalized = re.sub(r"\b(?:ko|k0|khong)\b", "khong", normalized)
    normalized = re.sub(
        r"\b(?:khong|chua)\s+(?:co\s+)?(?:bi\s+)?(?:hong|loi)"
        r"(?:\s+(?:hay|hoac)\s+(?:hong|loi))*\b",
        "hoat dong binh thuong", normalized,
    )
    star_label = {1: "dissatisfied", 2: "dissatisfied", 3: "neutral",
                  4: "satisfied", 5: "satisfied"}.get(rating)
    star_mismatch = star_label is not None and star_label != label
    explicit = hits(EXPLICIT_NEGATIVE, normalized)
    # Accent folding is useful for noisy text but must not confuse dùng mùa
    # with đừng mua, or ồn with ổn.
    if "đừng mua" not in text.casefold():
        explicit = [item for item in explicit if item != "dung mua"]
    faults = hits(FAULTS, normalized)
    positive = hits(POSITIVE, normalized)
    negative = hits(NEGATIVE_WORDS, normalized)
    if "ồn" in text.casefold() and not re.search(r"(?:không|ko|chưa)\s+ồn", text.casefold()):
        negative.append("ồn")
    # A clear final overall verdict can supersede a minor earlier complaint.
    conclusion = re.search(
        r"\b(?:noi chung|tong the|nhin chung|du sao|van rat)\b[^.!?]{0,65}$",
        normalized,
    )
    suggestion = None
    reason = ""
    resolved = re.search(r"\b(?:duoc doi|doi may moi|doi qua|doi sang|da sua|da khac phuc|"
                         r"da duoc khac phuc|het loi|may moi chay)\b", normalized)
    uncertain_context = re.search(
        r"\b(?:hy vong|hi vong|mong|neu|doc cmd|doc cmt|doc binh luan|nghe noi|"
        r"chua bao gio|khong bao gio|khong lam|chua co gi|chua co j|chua co lan nao|"
        r"giam can|so ky|kinh doanh)\b|\bai .{0,35}\b(?:thi|nen)\b",
        normalized,
    )
    strong_positive = re.search(r"\b(?:rat hai long|vo cung hai long|hoan toan hai long|rat ung y)\b", normalized)
    if conclusion and re.search(r"\b(?:hai long|ung y|rat tot)\b", conclusion.group()) and not (
        faults or explicit or MIXED_OR_UNCERTAIN.search(conclusion.group())
    ):
        suggestion, reason = "satisfied", "explicit_positive_overall_conclusion"
    elif explicit and not positive and not resolved and not uncertain_context:
        suggestion, reason = "dissatisfied", "explicit_dissatisfaction"
    elif len(set(faults)) >= 2 and not resolved and not uncertain_context and (not positive or "duoc cai" in normalized or "chi duoc" in normalized):
        suggestion, reason = "dissatisfied", "multiple_functional_faults"
    elif (strong_positive or SHORT_PRAISE.fullmatch(normalized)) and not (
        explicit or faults or negative or MIXED_OR_UNCERTAIN.search(normalized)
        or "ồn" in text.casefold() or uncertain_context
    ):
        suggestion, reason = "satisfied", "clear_positive_verdict"
    elif QUESTION.search(normalized) and len(_WORDS.findall(text)) <= 15 and not (explicit or faults or negative or positive):
        suggestion, reason = "neutral", "information_question_without_evaluation"
    elif re.fullmatch(r"(?:moi (?:nhan|mua)[, .]*)?(?:chua dung|chua su dung|chua trai nghiem)[ .!]*", normalized):
        suggestion, reason = "neutral", "not_yet_used"
    elif re.fullmatch(r"(?:binh thuong|tam tam|khong co y kien|chua danh gia)[ .!]*", normalized):
        suggestion, reason = "neutral", "explicit_neutral_verdict"

    # Mild product/service trade-offs require contextual review, even when a
    # narrow negative phrase was found. Never turn them into definite negatives.
    mild_positive = re.search(
        r"\b(?:ok|oke|cung duoc|con lai deu on|rat thich|mau ma dep|thiet ke dep|"
        r"danh gia cao|nhung cai khac)\b", normalized,
    )
    if suggestion == "dissatisfied" and (
        mild_positive or (label == "neutral" and re.search(r"\b(?:nhung|tuy nhien|co dieu)\b", normalized))
    ):
        suggestion, reason = None, "mixed_positive_negative_evaluation"

    # Strong mixed polarity is ambiguous: retain it for review, do not guess.
    suspected = star_mismatch
    if len(_WORDS.findall(text)) > 80:
        suggestion = None
    action = "keep"
    if suspected and suggestion is not None:
        action = "correct" if suggestion != label else "keep_text_supported"
    elif suspected:
        # A complaint is valid even at 4-5 stars; do not neutralize 3-star complaints.
        if label == "dissatisfied" and (explicit or faults or negative) and not positive:
            action, reason = "keep_text_supported", "negative_text_despite_stars"
        elif label == "satisfied" and positive and not (
            explicit or faults or negative or MIXED_OR_UNCERTAIN.search(normalized)
        ):
            action, reason = "keep_text_supported", "positive_text_despite_stars"
        else:
            action, reason = "needs_review", "star_text_mismatch_or_mixed_evaluation"
    return {"action": action, "suggested_label": suggestion, "reason": reason,
            "star_mismatch": star_mismatch,
            "evidence": "; ".join(dict.fromkeys(explicit + faults + positive + negative))}


def read_snapshot(path: Path) -> tuple[list[dict], dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        if conn.execute("SELECT COUNT(*) FROM batches WHERE status='running'").fetchone()[0]:
            raise RuntimeError("Stop the labeler before curating a release")
        rows = [dict(row) for row in conn.execute(
            "SELECT source_review_id,product_id,product_title,category,rating,"
            "text_clean,label,label_source,model,prompt_version,is_canonical,eligible "
            "FROM records ORDER BY source_review_id"
        )]
        pending = sum(r["eligible"] and r["label"] is None for r in rows)
        if pending:
            raise RuntimeError(f"Finish labeling first: {pending} eligible rows are unlabeled")
        return rows, {"total_rows": len(rows), "eligible_rows": sum(r["eligible"] for r in rows),
                      "label_counts": dict(Counter(r["label"] for r in rows if r["label"]))}
    finally:
        conn.close()


def apply_corrections(path: Path, rows: list[dict], audits: dict, backup_dir: Path) -> tuple[Path, int]:
    corrections = [(r, audits[r["source_review_id"]]) for r in rows
                   if audits.get(r["source_review_id"], {}).get("action") == "correct"]
    backup_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    backup = backup_dir / f"sentiment_before_curation_{run_id[:12]}.sqlite3"
    source = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    target = sqlite3.connect(backup)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    conn = sqlite3.connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT COUNT(*) FROM batches WHERE status='running'").fetchone()[0]:
            raise RuntimeError("Labeling resumed; no corrections applied")
        conn.execute("""CREATE TABLE IF NOT EXISTS label_corrections (
            run_id TEXT NOT NULL, source_review_id TEXT NOT NULL, old_label TEXT NOT NULL,
            new_label TEXT NOT NULL, old_label_source TEXT, reason TEXT NOT NULL,
            evidence TEXT NOT NULL, method TEXT NOT NULL, corrected_at TEXT NOT NULL,
            PRIMARY KEY(run_id,source_review_id))""")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for row, audit in corrections:
            cursor = conn.execute(
                "UPDATE records SET label=?,label_source='text_rule_review',updated_at=? "
                "WHERE source_review_id=? AND label=? AND text_clean=?",
                (audit["suggested_label"], now, row["source_review_id"], row["label"], row["text_clean"]),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Workspace changed during curation; transaction rolled back")
            conn.execute("INSERT INTO label_corrections VALUES (?,?,?,?,?,?,?,?,?)",
                         (run_id, row["source_review_id"], row["label"], audit["suggested_label"],
                          row["label_source"], audit["reason"], audit["evidence"], VERSION, now))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    return backup, len(corrections)


def fingerprint(text: str) -> str:
    """Punctuation-insensitive key preserving accents, negation, questions and emoji."""
    tokens = _WORDS.findall(text.casefold())
    symbols = "".join(c for c in text if unicodedata.category(c) == "So")
    return " ".join(tokens) + (" ?" if "?" in text else "") + " " + symbols


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        self.parent[max(a, b)] = min(a, b)


def near_duplicate_groups(rows: list[dict], seed: int = 42) -> tuple[UnionFind, dict]:
    """MinHash LSH candidate search followed by exact bigram Jaccard >= .90."""
    import numpy as np

    uf = UnionFind(len(rows))
    rng = np.random.default_rng(seed)
    multipliers = rng.integers(1, 2**32, size=64, dtype=np.uint64) | np.uint64(1)
    offsets = rng.integers(0, 2**32, size=64, dtype=np.uint64)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    shingles: list[frozenset] = []
    protected: list[frozenset] = []
    edge_count = 0
    candidates = 0
    for index, row in enumerate(rows):
        tokens = _WORDS.findall(row["text_clean"].casefold())
        grams = frozenset(zip(tokens, tokens[1:]))
        shingles.append(grams)
        protected.append(frozenset(t for t in tokens if t in {
            "không", "ko", "k", "chưa", "chẳng", "chả", "đừng", "tốt", "tệ",
            "hỏng", "lỗi", "đẹp", "kém", "ok", "ổn", "nhưng",
        } or t.isdigit()))
        if len(tokens) < 8 or len(grams) < 5:
            continue
        values = np.array([int.from_bytes(hashlib.blake2b(
            (a + "\0" + b).encode("utf-8"), digest_size=8).digest(), "little")
            for a, b in grams], dtype=np.uint64)
        signature = ((values[:, None] * multipliers + offsets) & np.uint64(0xffffffff)).min(axis=0)
        keys = [(band, signature[band * 4:(band + 1) * 4].tobytes()) for band in range(16)]
        possible = set()
        for key in keys:
            possible.update(buckets[key])
        for other in possible:
            if protected[index] != protected[other]:
                continue
            previous = shingles[other]
            if min(len(grams), len(previous)) / max(len(grams), len(previous)) < 0.9:
                continue
            candidates += 1
            if len(grams & previous) / len(grams | previous) >= 0.9:
                uf.union(index, other)
                edge_count += 1
        for key in keys:
            buckets[key].append(index)
        if (index + 1) % 10_000 == 0:
            print(f"[curation:dedup] rows={index + 1}/{len(rows)} pairs={edge_count}", flush=True)
    return uf, {"algorithm": "64-permutation MinHash, 16x4 LSH, exact bigram Jaccard",
                "jaccard_threshold": 0.9, "min_tokens": 8,
                "candidate_pairs_verified": candidates, "detected_edges": edge_count,
                "search_is_approximate": True, "protected_negation_and_numbers": True}


def grouped_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Assign whole products with a greedy stratified size/label objective."""
    products: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        products[row["product_id"]].append(row)
    totals = Counter(row["label"] for row in rows)
    counts = {name: Counter() for name in SPLITS}
    result: dict[str, list[dict]] = {name: [] for name in SPLITS}
    ordered = sorted(products.items(), key=lambda item: (
        -len(item[1]), hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()))
    for product, group in ordered:
        vector = Counter(row["label"] for row in group)
        def cost(name):
            total = 0.0
            for label, count in vector.items():
                target = max(1.0, totals[label] * SPLITS[name])
                current = counts[name][label]
                total += ((current + count - target) ** 2 - (current - target) ** 2) / target
            target_size = max(1.0, len(rows) * SPLITS[name])
            size = len(result[name])
            total += 0.5 * (((size + len(group) - target_size) ** 2 - (size - target_size) ** 2) / target_size)
            return total
        name = min(SPLITS, key=cost)
        result[name].extend(group)
        counts[name].update(vector)
    for name in result:
        result[name].sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['source_review_id']}".encode()).hexdigest())
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    fields = fields or (list(rows[0]) if rows else ["source_review_id", "reason"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_splits(splits: dict[str, list[dict]]) -> dict:
    report = {"overlapping_ids": 0, "overlapping_products": 0, "overlapping_texts": 0,
              "invalid_labels": 0, "empty_texts": 0}
    seen_ids, seen_products, seen_texts = set(), set(), set()
    for name, rows in splits.items():
        ids = {r["source_review_id"] for r in rows}
        products = {r["product_id"] for r in rows}
        texts = {fingerprint(r["text_clean"]) for r in rows}
        if len(ids) != len(rows) or len(texts) != len(rows):
            raise RuntimeError(f"Duplicate id or normalized text in {name}")
        report["overlapping_ids"] += len(ids & seen_ids)
        report["overlapping_products"] += len(products & seen_products)
        report["overlapping_texts"] += len(texts & seen_texts)
        report["invalid_labels"] += sum(r["label"] not in VALID_LABELS for r in rows)
        report["empty_texts"] += sum(not r["text_clean"].strip() for r in rows)
        seen_ids.update(ids)
        seen_products.update(products)
        seen_texts.update(texts)
    if any(report.values()):
        raise RuntimeError(f"Dataset validation failed: {report}")
    report["passed"] = True
    return report


def build_release(workspace: Path, output_dir: Path, seed: int = 42) -> dict:
    """Correct only text-supported suspect labels, deduplicate, export and ZIP."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_dir.exists():
        raise FileExistsError(f"Output already exists; choose a new --output-dir: {output_dir}")
    rows, original_stats = read_snapshot(workspace)
    original_digest = hashlib.sha256(json.dumps(
        [(r["source_review_id"], r["text_clean"], r["label"]) for r in rows],
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    audits = {r["source_review_id"]: review_label(r["text_clean"], r["label"], r["rating"])
              for r in rows if r["eligible"] and r["label"] in VALID_LABELS}
    backup, corrected = apply_corrections(workspace, rows, audits, output_dir.parent / "backups")
    output_dir.mkdir(parents=True)
    for directory in ("data", "csv", "parquet", "audit", "scripts"):
        (output_dir / directory).mkdir()
    exclusions, correction_log, audit_rows = [], [], []
    clean_candidates = []
    for row in rows:
        identifier = row["source_review_id"]
        original_label = row["label"]
        audit = audits.get(identifier)
        text = clean_review_text(row["text_clean"])
        base = {"source_review_id": identifier, "text": text, "rating": row["rating"],
                "original_label": original_label}
        if not row["eligible"] or not is_eligible_for_labeling(text):
            exclusions.append({**base, "reason": "no_language_content"})
            continue
        if audit is None:
            exclusions.append({**base, "reason": "missing_or_invalid_label"})
            continue
        if audit["star_mismatch"] or audit["action"] != "keep":
            audit_rows.append({**base, **audit})
        if audit["action"] == "correct":
            correction_log.append({**base, "new_label": audit["suggested_label"],
                                   "reason": audit["reason"], "evidence": audit["evidence"]})
            row["label"] = audit["suggested_label"]
            row["label_source"] = "text_rule_review"
        if audit["action"] == "needs_review":
            exclusions.append({**base, "reason": "unresolved_label_review"})
            continue
        words = _WORDS.findall(text)
        if len(words) <= 1 and fold(text).strip(" .!?") not in {
            "tot", "on", "ok", "okay", "dep", "te", "kem", "tuyet", "tam", "bt", "bth",
        }:
            exclusions.append({**base, "reason": "low_information"})
            continue
        row["text_clean"] = text
        clean_candidates.append(row)

    by_text: dict[str, list[dict]] = defaultdict(list)
    for row in clean_candidates:
        by_text[fingerprint(row["text_clean"])].append(row)
    duplicate_rows, representatives = [], []
    for group in by_text.values():
        if len({r["label"] for r in group}) > 1:
            for row in group:
                exclusions.append({"source_review_id": row["source_review_id"], "text": row["text_clean"],
                                   "rating": row["rating"], "original_label": row["label"],
                                   "reason": "conflicting_labels_for_identical_text"})
            continue
        representative = group[0]
        representatives.append(representative)
        for row in group[1:]:
            duplicate_rows.append({"source_review_id": row["source_review_id"],
                                   "kept_id": representative["source_review_id"], "kind": "normalized_exact"})
    print(f"[curation] corrected={corrected} candidates={len(clean_candidates)} exact_unique={len(representatives)}", flush=True)
    uf, near_report = near_duplicate_groups(representatives, seed)
    clusters: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(representatives):
        clusters[uf.find(index)].append(row)
    final = []
    for group in clusters.values():
        if len({r["label"] for r in group}) > 1:
            for row in group:
                exclusions.append({"source_review_id": row["source_review_id"], "text": row["text_clean"],
                                   "rating": row["rating"], "original_label": row["label"],
                                   "reason": "conflicting_labels_in_near_duplicate_cluster"})
            continue
        final.append(group[0])
        for row in group[1:]:
            duplicate_rows.append({"source_review_id": row["source_review_id"],
                                   "kept_id": group[0]["source_review_id"], "kind": "near_duplicate"})
    splits = grouped_split(final, seed)
    validation = validate_splits(splits)
    if len(final) + len(exclusions) + len(duplicate_rows) != len(rows):
        raise RuntimeError("Input/output accounting failed")
    split_counts = {}
    provenance = []
    for name, split_rows in splits.items():
        public = []
        for row in split_rows:
            public.append({"id": row["source_review_id"], "text": row["text_clean"],
                           "label": LABEL_IDS[row["label"]], "label_name": row["label"],
                           "product_id": row["product_id"], "category": row["category"],
                           "group_id": "product-" + row["product_id"]})
            provenance.append({"id": row["source_review_id"], "split": name,
                               "rating": row["rating"], "label_source": row["label_source"],
                               "model": row["model"], "prompt_version": row["prompt_version"],
                               "curation_version": VERSION})
        write_csv(output_dir / "csv" / f"{name}.csv", public, FIELDS)
        with (output_dir / "data" / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as file:
            for item in public:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        pq.write_table(pa.Table.from_pylist(public), output_dir / "parquet" / f"{name}.parquet", compression="zstd")
        split_counts[name] = {"rows": len(public), "labels": dict(Counter(r["label"] for r in split_rows))}

    audit_fields = ["source_review_id", "text", "rating", "original_label", "action", "suggested_label", "reason", "star_mismatch", "evidence"]
    write_csv(output_dir / "audit" / "label_audit.csv", audit_rows, audit_fields)
    write_csv(output_dir / "audit" / "label_corrections.csv", correction_log,
              ["source_review_id", "text", "rating", "original_label", "new_label", "reason", "evidence"])
    write_csv(output_dir / "audit" / "excluded.csv", exclusions,
              ["source_review_id", "text", "rating", "original_label", "reason"])
    write_csv(output_dir / "audit" / "duplicates.csv", duplicate_rows,
              ["source_review_id", "kept_id", "kind"])
    write_csv(output_dir / "audit" / "provenance.csv", provenance,
              ["id", "split", "rating", "label_source", "model", "prompt_version", "curation_version"])
    report = {
        "dataset_name": "DMX-Sentiment", "version": VERSION, "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_snapshot_sha256": original_digest, "input_stats": original_stats,
        "rows_corrected_in_workspace": corrected,
        "review_actions": dict(Counter(a["action"] for a in audits.values())),
        "star_mismatch_candidates": sum(a["star_mismatch"] for a in audits.values()),
        "excluded": dict(Counter(r["reason"] for r in exclusions)),
        "duplicates_removed": dict(Counter(r["kind"] for r in duplicate_rows)),
        "final_rows": len(final), "splits": split_counts,
        "target_split_ratios": SPLITS, "split_method": "label-stratified greedy grouping by product_id",
        "label_mapping": {label: {"id": identifier, "vi": LABEL_NAMES[label]} for label, identifier in LABEL_IDS.items()},
        "near_duplicate_detection": near_report, "validation": validation,
        "label_quality": "silver: Gemini labels with deterministic text-rule review; not human gold",
        "rating_usage": "candidate flag only; never a target label; excluded from model features",
        "backup_database": str(backup.resolve()),
    }
    write_json(output_dir / "quality_report.json", report)
    write_json(output_dir / "label_mapping.json", report["label_mapping"])
    write_json(output_dir / "schema.json", {"id": "string", "text": "string", "label": "int64",
                                            "label_name": "string", "product_id": "string", "category": "string", "group_id": "string"})
    write_dataset_card(output_dir, report)
    write_loader(output_dir)
    hashes = {p.relative_to(output_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output_dir.rglob("*")) if p.is_file()}
    write_json(output_dir / "checksums.json", hashes)
    archive = output_dir.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zipped.write(path, (Path(output_dir.name) / path.relative_to(output_dir)).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError("ZIP integrity validation failed")
    return {"dataset": str(output_dir), "zip": str(archive), "backup": str(backup),
            "corrected": corrected, "final_rows": len(final), "splits": split_counts,
            "exclusions": report["excluded"], "duplicates_removed": report["duplicates_removed"]}


def write_dataset_card(output_dir: Path, report: dict):
    counts = report["splits"]
    table = "\n".join(f"| {name} | {value['rows']} | {value['labels'].get('dissatisfied', 0)} | "
                      f"{value['labels'].get('neutral', 0)} | {value['labels'].get('satisfied', 0)} |"
                      for name, value in counts.items())
    content = f"""---
language:
- vi
pretty_name: DMX-Sentiment
task_categories:
- text-classification
task_ids:
- sentiment-classification
tags:
- reviews
- vietnamese
- silver-labels
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# DMX-Sentiment

Bộ dữ liệu tiếng Việt phân loại mức độ hài lòng từ bình luận sản phẩm/dịch vụ
Điện Máy Xanh do dự án tự thu thập và xử lý. Không có câu synthetic trong bản này.

| Split | Tổng | Không hài lòng | Trung tính | Hài lòng |
| --- | ---: | ---: | ---: | ---: |
{table}

## Nhãn và trường dữ liệu

- `label`: 0 = `dissatisfied` (không hài lòng), 1 = `neutral` (trung tính),
  2 = `satisfied` (hài lòng).
- `id`: ID bình luận nguồn; `text`: nội dung chuẩn hóa nhẹ, che số điện thoại,
  email và URL; `label_name`: tên nhãn máy đọc được.
- `product_id`, `category`, `group_id`: phục vụ kiểm tra và phân nhóm.
- Chỉ dùng `text` làm đầu vào, `label` làm mục tiêu. Sao đánh giá và nguồn nhãn
  được giữ riêng ở `audit/provenance.csv`, không đưa vào model.

## Chuẩn bị và kiểm tra nhãn

Nhãn đầu vào do Gemini gán trên nội dung. Sao chỉ dùng phát hiện câu nghi ngờ.
Chỉ sửa khi có bằng chứng văn bản rõ theo quy tắc; lệch sao không tự động đổi
nhãn. Các câu khen/chê cân bằng hoặc chưa đủ bằng chứng nằm trong
`audit/excluded.csv`, lý do `unresolved_label_review`.

Đã sửa {report['rows_corrected_in_workspace']} bản ghi trong database nguồn của
pipeline sentiment, có backup và log trước/sau ở `audit/label_corrections.csv`.
Đây là nhãn **silver** (Gemini + quy tắc), chưa phải nhãn gold do người đánh giá.
Validation/test cũng là silver; điểm số trên chúng không thay thế kiểm tra thủ công.

## Làm sạch, trùng và chia tập

Giữ nguyên dấu tiếng Việt, phủ định, tiếng lóng và cảm xúc. Loại nội dung không có
ngôn ngữ, câu ít thông tin, trùng nội dung chuẩn hóa và nhóm trùng có nhãn mâu thuẫn.
MinHash tìm ứng viên gần trùng; xác minh bằng Jaccard bigram >= 0.90 và bảo vệ
phủ định/từ đánh giá/số. Tìm kiếm gần trùng là xấp xỉ, có thể bỏ sót một số cặp.

Giữ một đại diện mỗi nhóm trùng/gần trùng. Chia khoảng 80/10/10 bằng seed
{report['seed']}, phân tầng nhãn và gom theo sản phẩm; sản phẩm, ID và nội dung
chuẩn hóa không giao nhau giữa các split. Do gom sản phẩm nên số lượng không
nhất thiết đúng chính xác 80/10/10. Không cân bằng lớp bằng cách sinh dữ liệu giả.

## Định dạng và sử dụng

- `data/*.jsonl`: mặc định cho Hugging Face Datasets.
- `csv/*.csv`: cùng dữ liệu ở UTF-8 BOM.
- `parquet/*.parquet`: cùng dữ liệu, nén Zstandard.
- `quality_report.json`, `checksums.json`, `schema.json`, `label_mapping.json`.
- `scripts/load_dataset.py`: nạp và kiểm tra nhanh bằng thư viện `datasets`.

```python
from datasets import load_dataset
ds = load_dataset("json", data_files={{
    "train": "data/train.jsonl",
    "validation": "data/validation.jsonl",
    "test": "data/test.jsonl",
}})
```

## Hạn chế và nguồn

Dữ liệu thuộc miền điện máy, lệch lớp hài lòng, chưa bảo đảm tổng quát cho mạng
xã hội hoặc lĩnh vực khác. Một số nhãn tự động có thể còn sai, nhất là mỉa mai,
khen/chê hỗn hợp, tiếng lóng. Quy tắc rà soát không bao phủ toàn bộ ngôn ngữ.
Chưa xác định giấy phép phân phối lại từ nguồn; gói này không tự cấp giấy phép
cho nội dung khách hàng. Không kèm tên người dùng hay số sao trong dữ liệu model.

Nguồn crawl: https://www.dienmayxanh.com/
Tham khảo cấu trúc dataset card: https://huggingface.co/docs/hub/datasets-cards
Tham khảo định dạng/split: https://huggingface.co/docs/datasets/loading
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def write_loader(output_dir: Path):
    content = '''from pathlib import Path
from datasets import ClassLabel, load_dataset

root = Path(__file__).resolve().parents[1]
files = {split: str(root / "data" / f"{split}.jsonl")
         for split in ("train", "validation", "test")}
dataset = load_dataset("json", data_files=files)
dataset = dataset.cast_column("label", ClassLabel(names=["dissatisfied", "neutral", "satisfied"]))
if __name__ == "__main__":
    print(dataset)
    print(dataset["train"].features)
'''
    (output_dir / "scripts" / "load_dataset.py").write_text(content, encoding="utf-8")


def verify_release(output_dir: Path) -> dict:
    """Independently read all formats and check disk/ZIP checksums and leakage."""
    import pyarrow.parquet as pq

    report = json.loads((output_dir / "quality_report.json").read_text(encoding="utf-8"))
    checksums = json.loads((output_dir / "checksums.json").read_text(encoding="utf-8"))
    for filename, expected in checksums.items():
        actual = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {filename}")
    splits = {}
    for split in SPLITS:
        with (output_dir / "data" / f"{split}.jsonl").open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file]
        with (output_dir / "csv" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as file:
            csv_rows = [{**row, "label": int(row["label"])} for row in csv.DictReader(file)]
        parquet_rows = pq.read_table(output_dir / "parquet" / f"{split}.parquet").to_pylist()
        if rows != csv_rows or rows != parquet_rows:
            raise RuntimeError(f"Formats disagree: {split}")
        if len(rows) != report["splits"][split]["rows"]:
            raise RuntimeError(f"Row count mismatch: {split}")
        if dict(Counter(r["label_name"] for r in rows)) != report["splits"][split]["labels"]:
            raise RuntimeError(f"Label count mismatch: {split}")
        if any(LABEL_IDS[row["label_name"]] != row["label"] for row in rows):
            raise RuntimeError(f"Label mapping mismatch: {split}")
        splits[split] = [{"source_review_id": r["id"], "text_clean": r["text"],
                          "product_id": r["product_id"], "label": r["label_name"]} for r in rows]
    validation = validate_splits(splits)
    with zipfile.ZipFile(output_dir.with_suffix(".zip")) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC error")
        expected_members = {f"{output_dir.name}/{name}" for name in checksums}
        expected_members.add(f"{output_dir.name}/checksums.json")
        if set(archive.namelist()) != expected_members:
            raise RuntimeError("ZIP inventory mismatch")
        for filename in checksums:
            payload = archive.read(f"{output_dir.name}/{filename}")
            if hashlib.sha256(payload).hexdigest() != checksums[filename]:
                raise RuntimeError(f"ZIP payload mismatch: {filename}")
    return {"passed": True, "all_formats_identical": True, "checksums_verified": len(checksums),
            "zip_crc_and_payloads_verified": True, "rows": report["final_rows"],
            "split_validation": validation}
