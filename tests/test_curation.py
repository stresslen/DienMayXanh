from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from dmx_crawler.curation import (
    LABEL_IDS, apply_corrections, build_release, fingerprint, grouped_split,
    near_duplicate_groups, read_snapshot, review_label, validate_splits, verify_release,
)
from dmx_crawler.models import Product
from dmx_crawler.sentiment import SentimentWorkspace
from dmx_crawler.storage import Storage
from test_toxicity import add_review


@pytest.mark.parametrize("text,label,rating,action,new_label", [
    ("Ép không kiệt, ít nước, cặn nhiều, ép xong phải vắt lại. Được cái vệ sinh dễ dàng.",
     "satisfied", 1, "correct", "dissatisfied"),
    ("Sản phẩm rất tốt, dùng tiện, giá hợp lý", "satisfied", 1, "keep_text_supported", None),
    ("Máy bị hỏng rồi, không chạy, rất thất vọng", "dissatisfied", 5, "keep_text_supported", "dissatisfied"),
    ("Không hài lòng", "neutral", 4, "correct", "dissatisfied"),
    ("Mẫu đẹp, máy hơi ồn, chưa biết độ bền", "satisfied", 2, "needs_review", None),
    ("Sản phẩm rất tốt nhưng không hài lòng về bảo hành", "satisfied", 1, "needs_review", None),
    ("Có hàng không?", "satisfied", 1, "correct", "neutral"),
    ("Máy chạy rất tốt, không bị hỏng hay lỗi", "satisfied", 1, "keep_text_supported", None),
    ("Bình thường", "neutral", 3, "keep", "neutral"),
    ("Sản phẩm tốt nhân viên hơi thiếu sót dặn hàng từ hôm qua hôm nay lấy chưa có. Mất thời gian", "dissatisfied", 5, "needs_review", None),
    ("Mình mua máy về xay được 2 lần thì bị lỗi sau đó được đổi máy mới. Máy mới chạy tương đối tốt.", "satisfied", 3, "needs_review", None),
    ("Sản phẩm tốt", "neutral", 4, "correct", "satisfied"),
    ("Máy ồn.", "dissatisfied", 3, "keep_text_supported", None),
    ("Mong máy không làm mình thất vọng", "neutral", 5, "needs_review", None),
    ("Quạt êm tiết kiệm điện nên dùng mùa hè", "neutral", 5, "needs_review", None),
])
def test_text_evidence_controls_corrections(text, label, rating, action, new_label):
    result = review_label(text, label, rating)
    assert result["action"] == action
    assert result["suggested_label"] == new_label


def test_fingerprint_preserves_semantic_differences():
    assert fingerprint("Sản phẩm TỐT!!!") == fingerprint("sản phẩm tốt.")
    assert fingerprint("Máy tốt") != fingerprint("Máy không tốt")
    assert fingerprint("Máy tốt?") != fingerprint("Máy tốt")
    assert fingerprint("Tốt 😡") != fingerprint("Tốt 🙂")


def test_near_duplicates_keep_negations_and_numbers():
    texts = [
        "Sản phẩm chạy rất tốt nhân viên giao hàng nhiệt tình và nhanh chóng hôm nay",
        "Sản phẩm chạy rất tốt nhân viên giao hàng nhiệt tình và nhanh chóng hôm nay nhé",
        "Sản phẩm chạy không tốt nhân viên giao hàng nhiệt tình và nhanh chóng hôm nay",
        "Sản phẩm chạy rất tốt nhân viên giao hàng nhiệt tình và nhanh chóng hôm nay 2",
    ]
    uf, report = near_duplicate_groups([{"text_clean": text} for text in texts])
    assert uf.find(0) == uf.find(1)
    assert uf.find(0) != uf.find(2)
    assert uf.find(0) != uf.find(3)
    assert report["detected_edges"] >= 1


def test_product_grouped_split_is_deterministic_and_has_no_leakage():
    rows = [{"source_review_id": str(i), "product_id": str(i // 3),
             "label": ["dissatisfied", "neutral", "satisfied"][i % 3],
             "text_clean": f"Bình luận thử nghiệm {i}"} for i in range(120)]
    first = grouped_split(rows, 42)
    assert first == grouped_split(list(reversed(rows)), 42)
    assert validate_splits(first)["passed"]
    assert all({r["label"] for r in group} == set(LABEL_IDS) for group in first.values())


def make_workspace(tmp_path):
    source = tmp_path / "raw.sqlite3"
    storage = Storage(source)
    try:
        for index in range(40):
            product = Product(str(index), f"https://example.test/{index}", "Demo", "demo")
            storage.save_product(product)
            add_review(storage, product, str(index), f"Đánh giá trải nghiệm mẫu số {index}")
        add_review(storage, product, "wrong", "Ép không kiệt, cặn nhiều, phải vắt lại")
        add_review(storage, product, "duplicate", "Đánh giá trải nghiệm mẫu số 0")
        add_review(storage, product, "conflict-a", "Nội dung mâu thuẫn")
        add_review(storage, product, "conflict-b", "NỘI DUNG MÂU THUẪN")
        add_review(storage, product, "empty", "!!!")
    finally:
        storage.close()
    path = tmp_path / "sentiment.sqlite3"
    SentimentWorkspace.prepare(source, path)
    workspace = SentimentWorkspace(path)
    try:
        workspace.conn.execute("UPDATE records SET label='satisfied' WHERE eligible=1")
        workspace.conn.execute("UPDATE records SET label='neutral',rating=3 WHERE source_review_id='conflict-b'")
        workspace.conn.execute("UPDATE records SET rating=1 WHERE source_review_id='wrong'")
        workspace.conn.commit()
    finally:
        workspace.close()
    return path


def test_release_corrects_logs_backs_up_and_zips(tmp_path):
    pytest.importorskip("pyarrow")
    path = make_workspace(tmp_path)
    output = tmp_path / "release"
    result = build_release(path, output)
    assert result["corrected"] == 1
    assert result["final_rows"] == 41
    assert result["exclusions"]["conflicting_labels_for_identical_text"] == 2
    assert result["duplicates_removed"]["normalized_exact"] == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT label FROM records WHERE source_review_id='wrong'").fetchone()[0] == "dissatisfied"
        assert conn.execute("SELECT COUNT(*) FROM label_corrections").fetchone()[0] == 1
    with sqlite3.connect(result["backup"]) as conn:
        assert conn.execute("SELECT label FROM records WHERE source_review_id='wrong'").fetchone()[0] == "satisfied"
    with zipfile.ZipFile(result["zip"]) as zipped:
        assert zipped.testzip() is None
        assert "release/data/train.jsonl" in zipped.namelist()
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert report["validation"]["passed"]
    assert report["rating_usage"].startswith("candidate flag only")
    assert verify_release(output)["passed"]
    with pytest.raises(FileExistsError):
        build_release(path, output)
