from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import pytest

from dmx_crawler.cleaning import clean_review_text
from dmx_crawler.models import Product, Review
from dmx_crawler.storage import Storage
from dmx_crawler.toxicity import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    GeminiToxicityLabeler,
    LabelInput,
    ToxicityWorkspace,
    run_labeling,
)


class FakeLabeler:
    model = "fake-gemini"

    def __init__(self) -> None:
        self.calls: list[list[LabelInput]] = []

    def label(self, records: Sequence[LabelInput]) -> dict[str, str]:
        self.calls.append(list(records))
        return {
            record.source_review_id: (
                "toxic" if "ngu" in record.text.casefold() else "non_toxic"
            )
            for record in records
        }

    def close(self) -> None:
        return None


class FailingLabeler:
    model = "fake-failing-gemini"

    def __init__(self) -> None:
        self.call_count = 0

    def label(self, records: Sequence[LabelInput]) -> dict[str, str]:
        self.call_count += 1
        raise RuntimeError("temporary API failure")

    def close(self) -> None:
        return None


def add_review(
    storage: Storage,
    product: Product,
    review_id: str,
    text: str,
) -> None:
    storage.save_reviews(
        [
            Review(
                source_review_id=review_id,
                product_id=product.product_id,
                product_url=product.url,
                product_title=product.title,
                category=product.category,
                rating=5,
                text_raw=text,
                verified_purchase=True,
                helpful_count=0,
                usage_duration=None,
                source_page=1,
            )
        ]
    )


def test_cleaning_masks_pii_but_preserves_language() -> None:
    cleaned = clean_review_text(
        "  Không tốt 😡\u200b, gọi 0901 234 567 hoặc https://example.test/a  "
    )
    assert cleaned == "Không tốt 😡, gọi <PHONE> hoặc <URL>"


def test_api_schema_requires_the_configured_number_of_results() -> None:
    schema = GeminiToxicityLabeler._schema(DEFAULT_BATCH_SIZE)
    assert schema["minItems"] == DEFAULT_BATCH_SIZE
    assert schema["maxItems"] == DEFAULT_BATCH_SIZE
    assert MAX_BATCH_SIZE == 1_000


def test_gemini_close_supports_older_sdk_client() -> None:
    class FakeHttpClient:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeApiClient:
        def __init__(self) -> None:
            self._httpx_client = FakeHttpClient()

    class OldSdkClient:
        def __init__(self) -> None:
            self._api_client = FakeApiClient()

    labeler = GeminiToxicityLabeler.__new__(GeminiToxicityLabeler)
    labeler._client = OldSdkClient()
    labeler.close()
    assert labeler._client._api_client._httpx_client.closed is True


def test_prepare_label_propagate_duplicates_and_export(tmp_path: Path) -> None:
    source_path = tmp_path / "raw.sqlite3"
    storage = Storage(source_path)
    product = Product("p1", "https://example.test/p1", "Demo", "tu-lanh")
    try:
        storage.save_product(product)
        add_review(storage, product, "r1", "Sản phẩm tốt")
        add_review(storage, product, "r2", "  sản phẩm TỐT  ")
        add_review(storage, product, "r3", "Nhân viên quá ngu")
        add_review(storage, product, "r4", "!!!")
    finally:
        storage.close()

    workspace_path = tmp_path / "toxicity.sqlite3"
    prepared = ToxicityWorkspace.prepare(source_path, workspace_path)
    assert prepared["total_rows"] == 4
    assert prepared["canonical_rows"] == 3
    assert prepared["duplicate_rows"] == 1
    assert prepared["pending_unique"] == 2

    workspace = ToxicityWorkspace(workspace_path)
    labeler = FakeLabeler()
    try:
        result = run_labeling(
            workspace,
            labeler,
            batch_size=DEFAULT_BATCH_SIZE,
            retry_delay_seconds=0,
        )
        assert len(labeler.calls) == 1
        assert len(labeler.calls[0]) == 2
        assert result["labeled_rows"] == 3
        assert result["labels"] == {"non_toxic": 2, "toxic": 1}

        duplicate = workspace.conn.execute(
            "SELECT label, label_source FROM records WHERE source_review_id='r2'"
        ).fetchone()
        assert dict(duplicate) == {
            "label": "non_toxic",
            "label_source": "exact_duplicate",
        }

        paths = workspace.export(tmp_path / "export")
        with paths["training_csv"].open(encoding="utf-8-sig", newline="") as file:
            training_rows = list(csv.DictReader(file))
        assert len(training_rows) == 2
        assert {row["label"] for row in training_rows} == {
            "toxic",
            "non_toxic",
        }
    finally:
        workspace.close()


def test_response_validation_rejects_missing_or_duplicate_indexes() -> None:
    records = [LabelInput("r1", "a"), LabelInput("r2", "b")]
    with pytest.raises(ValueError, match="expected 2"):
        GeminiToxicityLabeler._validate_response(
            [{"i": 0, "label": "non_toxic"}], records
        )
    with pytest.raises(ValueError, match="Duplicate result index"):
        GeminiToxicityLabeler._validate_response(
            [
                {"i": 0, "label": "non_toxic"},
                {"i": 0, "label": "toxic"},
            ],
            records,
        )


def test_max_batches_caps_api_calls_even_when_request_fails(tmp_path: Path) -> None:
    source_path = tmp_path / "raw.sqlite3"
    storage = Storage(source_path)
    product = Product("p1", "https://example.test/p1", "Demo", "tu-lanh")
    try:
        storage.save_product(product)
        add_review(storage, product, "r1", "Bình luận hợp lệ")
    finally:
        storage.close()

    workspace_path = tmp_path / "toxicity.sqlite3"
    ToxicityWorkspace.prepare(source_path, workspace_path)
    workspace = ToxicityWorkspace(workspace_path)
    labeler = FailingLabeler()
    try:
        result = run_labeling(
            workspace,
            labeler,
            max_batches=1,
            retry_delay_seconds=0,
        )
        assert labeler.call_count == 1
        assert result["pending_unique"] == 1
        assert result["batches"] == {"failed": 1}
    finally:
        workspace.close()


def test_prepare_requires_force_to_replace_workspace(tmp_path: Path) -> None:
    source_path = tmp_path / "raw.sqlite3"
    storage = Storage(source_path)
    storage.close()
    workspace_path = tmp_path / "toxicity.sqlite3"
    ToxicityWorkspace.prepare(source_path, workspace_path)
    with pytest.raises(FileExistsError, match="--force"):
        ToxicityWorkspace.prepare(source_path, workspace_path)


def test_incremental_sync_preserves_and_reuses_labels(tmp_path: Path) -> None:
    source_path = tmp_path / "raw.sqlite3"
    storage = Storage(source_path)
    product = Product("p1", "https://example.test/p1", "Demo", "tu-lanh")
    try:
        storage.save_product(product)
        add_review(storage, product, "r1", "Nhân viên quá ngu")
        add_review(storage, product, "r2", "Sản phẩm tốt")
    finally:
        storage.close()

    workspace_path = tmp_path / "toxicity.sqlite3"
    ToxicityWorkspace.prepare(source_path, workspace_path)
    workspace = ToxicityWorkspace(workspace_path)
    try:
        run_labeling(
            workspace,
            FakeLabeler(),
            batch_size=DEFAULT_BATCH_SIZE,
            retry_delay_seconds=0,
        )
    finally:
        workspace.close()

    storage = Storage(source_path)
    try:
        add_review(storage, product, "r3", "  NHÂN VIÊN QUÁ NGU  ")
        add_review(storage, product, "r4", "Sản phẩm dùng ổn")
    finally:
        storage.close()

    workspace = ToxicityWorkspace(workspace_path)
    try:
        result = workspace.sync_from_source(source_path)
        assert result["sync"] == {
            "source_rows": 4,
            "added_rows": 2,
            "added_unique": 1,
            "added_duplicates": 1,
            "reused_labels": 1,
        }
        reused = workspace.conn.execute(
            "SELECT label, label_source FROM records WHERE source_review_id='r3'"
        ).fetchone()
        assert dict(reused) == {
            "label": "toxic",
            "label_source": "exact_duplicate",
        }
        assert result["pending_unique"] == 1
        second_sync = workspace.sync_from_source(source_path)
        assert second_sync["sync"]["added_rows"] == 0
    finally:
        workspace.close()


def test_export_dmx_toxicity_10k_has_expected_schema_and_counts(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "raw.sqlite3"
    storage = Storage(source_path)
    product = Product("p1", "https://example.test/p1", "Demo", "tu-lanh")
    try:
        storage.save_product(product)
        for index in range(2):
            add_review(storage, product, f"t{index}", f"Nhân viên quá ngu {index}")
        for index in range(10):
            add_review(storage, product, f"n{index}", f"Sản phẩm tốt {index}")
    finally:
        storage.close()

    workspace_path = tmp_path / "toxicity.sqlite3"
    ToxicityWorkspace.prepare(source_path, workspace_path)
    workspace = ToxicityWorkspace(workspace_path)
    try:
        run_labeling(
            workspace,
            FakeLabeler(),
            batch_size=DEFAULT_BATCH_SIZE,
            retry_delay_seconds=0,
        )
        result = workspace.export_dmx_toxicity_10k(
            tmp_path / "dmx_10k", toxic_count=2, non_toxic_count=10, seed="test"
        )
        assert result["counts"]["all"] == {
            "rows": 12,
            "toxic": 2,
            "non_toxic": 10,
        }
        with Path(result["paths"]["all"]).open(
            encoding="utf-8", newline=""
        ) as file:
            exported = list(csv.DictReader(file))
        assert list(exported[0]) == [
            "Comment",
            "Constructiveness",
            "Toxicity",
            "Title",
            "Topic",
        ]
        assert {row["Toxicity"] for row in exported} == {"0", "1"}
        assert all(row["Constructiveness"] == "" for row in exported)
    finally:
        workspace.close()


def test_export_dmx_toxicity_10k_fills_shortfall_with_traced_synthetic_rows(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "raw.sqlite3"
    storage = Storage(source_path)
    product = Product("p1", "https://example.test/p1", "Máy lạnh Demo", "may-lanh")
    try:
        storage.save_product(product)
        add_review(storage, product, "t0", "Nhân viên quá ngu")
        for index in range(9):
            add_review(storage, product, f"n{index}", f"Sản phẩm dùng ổn {index}")
    finally:
        storage.close()

    workspace_path = tmp_path / "toxicity.sqlite3"
    ToxicityWorkspace.prepare(source_path, workspace_path)
    workspace = ToxicityWorkspace(workspace_path)
    try:
        run_labeling(
            workspace,
            FakeLabeler(),
            batch_size=DEFAULT_BATCH_SIZE,
            retry_delay_seconds=0,
        )
        result = workspace.export_dmx_toxicity_10k(
            tmp_path / "dmx_10k", toxic_count=4, non_toxic_count=6, seed="test-10k"
        )
        assert result["counts"]["all"] == {
            "rows": 10,
            "toxic": 4,
            "non_toxic": 6,
        }
        with Path(result["provenance"]).open(encoding="utf-8", newline="") as file:
            provenance = list(csv.DictReader(file))
        origins = [row["Origin"] for row in provenance if row["Toxicity"] == "1"]
        assert origins.count("real_labeled") == 1
        assert origins.count("synthetic_controlled") == 3
        assert len({row["Comment_SHA256"] for row in provenance}) == 10
    finally:
        workspace.close()
