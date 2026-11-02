from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dmx_crawler.cli import build_parser
from dmx_crawler.models import Product
from dmx_crawler.sentiment import (
    DEFAULT_BATCH_SIZE,
    LABEL_NAMES,
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    VALID_LABELS,
    GeminiSentimentLabeler,
    SentimentWorkspace,
)
from dmx_crawler.storage import Storage
from dmx_crawler.toxicity import LabelInput, ToxicityWorkspace, run_labeling
from test_toxicity import add_review


class FakeSentimentLabeler:
    model = "fake-sentiment"
    valid_labels = VALID_LABELS

    def __init__(self):
        self.calls: list[list[LabelInput]] = []

    def label(self, records):
        self.calls.append(list(records))
        return {
            item.source_review_id: (
                "dissatisfied" if "hỏng" in item.text else
                "satisfied" if "tốt" in item.text else "neutral"
            )
            for item in records
        }


def make_source(path: Path, texts: list[str]) -> Product:
    storage = Storage(path)
    product = Product("p1", "https://example.test/p1", "Demo", "tu-lanh")
    try:
        storage.save_product(product)
        for index, text in enumerate(texts):
            add_review(storage, product, f"r{index:04d}", text)
    finally:
        storage.close()
    return product


def test_sentiment_resumes_batches_of_100_and_exports_three_labels(tmp_path):
    source = tmp_path / "raw.sqlite3"
    texts = [f"Máy tốt {i}" for i in range(203)]
    texts.extend(["Máy hỏng rồi", "Còn hàng không?", "  MÁY TỐT 0  ", "!!!"])
    make_source(source, texts)
    path = tmp_path / "sentiment.sqlite3"
    prepared = SentimentWorkspace.prepare(source, path)
    assert prepared["pending_unique"] == 205
    workspace = SentimentWorkspace(path)
    labeler = FakeSentimentLabeler()
    try:
        first = run_labeling(workspace, labeler, max_batches=1)
        assert first["pending_unique"] == 105
        assert [len(batch) for batch in labeler.calls] == [100]
    finally:
        workspace.close()

    workspace = SentimentWorkspace(path)
    try:
        result = run_labeling(workspace, labeler)
        assert [len(batch) for batch in labeler.calls] == [100, 100, 5]
        assert result["labels"] == {
            "satisfied": 204, "dissatisfied": 1, "neutral": 1,
        }
        assert result["labeled_rows"] == 206
        assert result["pending_unique"] == 0
        assert result["failed_unique"] == 0
        assert {
            row[0] for row in workspace.conn.execute(
                "SELECT DISTINCT prompt_version FROM records WHERE label IS NOT NULL"
            )
        } == {PROMPT_VERSION}
        paths = workspace.export(tmp_path / "export")
        with paths["training_csv"].open(encoding="utf-8-sig", newline="") as file:
            training = list(csv.DictReader(file))
        assert len(training) == 205
        assert {row["label"] for row in training} == VALID_LABELS
        assert paths["training_csv"].name == "training_sentiment_unique.csv"
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assert metadata["label_names"] == LABEL_NAMES
        assert metadata["default_batch_size"] == 100
    finally:
        workspace.close()


def test_sentiment_failed_batch_recovery_and_incremental_sync(tmp_path):
    source = tmp_path / "raw.sqlite3"
    product = make_source(source, ["Máy tốt", "Máy hỏng"])
    path = tmp_path / "sentiment.sqlite3"
    SentimentWorkspace.prepare(source, path)
    workspace = SentimentWorkspace(path)
    try:
        items = workspace.next_batch(100, 3)
        batch_id, _ = workspace.start_batch(items, "fake", PROMPT_VERSION)
        with pytest.raises(ValueError, match="Invalid labels"):
            workspace.finish_batch(
                batch_id, {item.source_review_id: "toxic" for item in items},
                "fake", PROMPT_VERSION,
            )
        workspace.fail_batch(batch_id, "invalid response")
        items = workspace.next_batch(100, 3)
        workspace.start_batch(items, "fake", PROMPT_VERSION)
    finally:
        workspace.close()

    workspace = SentimentWorkspace(path)
    try:
        result = run_labeling(workspace, FakeSentimentLabeler())
        assert result["pending_unique"] == 0
        assert result["batches"] == {"failed": 1, "interrupted": 1, "succeeded": 1}
        storage = Storage(source)
        try:
            add_review(storage, product, "new-duplicate", "MÁY TỐT")
            add_review(storage, product, "new-question", "Máy nặng bao nhiêu?")
        finally:
            storage.close()
        synced = workspace.sync_from_source(source)
        assert synced["sync"]["reused_labels"] == 1
        assert synced["pending_unique"] == 1
        assert workspace.sync_from_source(source)["sync"]["added_rows"] == 0
    finally:
        workspace.close()


def test_sentiment_rejects_toxicity_workspace_and_reverse(tmp_path):
    path = tmp_path / "toxicity.sqlite3"
    workspace = ToxicityWorkspace(path)
    workspace.close()
    with pytest.raises(ValueError, match="separate workspace"):
        SentimentWorkspace(path)
    path = tmp_path / "sentiment.sqlite3"
    workspace = SentimentWorkspace(path)
    workspace.close()
    with pytest.raises(ValueError, match="separate workspace"):
        ToxicityWorkspace(path)


def test_sentiment_api_uses_one_request_with_three_label_schema():
    labeler = GeminiSentimentLabeler.__new__(GeminiSentimentLabeler)
    records = [LabelInput(str(i), "Máy tốt") for i in range(DEFAULT_BATCH_SIZE)]
    generated = [{"i": i, "label": "satisfied"} for i in reversed(range(100))]
    generate = Mock(return_value=SimpleNamespace(parsed=generated))
    labeler.model = "fake"
    labeler._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    labeler._types = SimpleNamespace(
        SafetySetting=lambda **kwargs: kwargs,
        GenerateContentConfig=lambda **kwargs: kwargs,
    )
    assert set(labeler.label(records).values()) == {"satisfied"}
    generate.assert_called_once()
    config = generate.call_args.kwargs["config"]
    assert config["system_instruction"] == SYSTEM_INSTRUCTION
    assert config["response_schema"]["minItems"] == 100
    assert set(config["response_schema"]["items"]["properties"]["label"]["enum"]) == VALID_LABELS
    payload = json.loads(generate.call_args.kwargs["contents"].split("\n", 1)[1])
    assert len(payload) == 100
    assert set(payload[0]) == {"i", "text"}
    with pytest.raises(ValueError, match="Invalid label"):
        labeler._validate_response([{"i": 0, "label": "non_toxic"}], records[:1])
    with pytest.raises(ValueError, match="Duplicate result index"):
        labeler._validate_response(
            [{"i": 0, "label": "neutral"}, {"i": 0, "label": "satisfied"}], records[:2]
        )


def test_sentiment_cli_defaults_are_separate_and_batch_is_100():
    args = build_parser().parse_args(["label-sentiment"])
    assert args.batch_size == 100
    assert str(args.workspace).replace("\\", "/") == "data/processed/sentiment/sentiment.sqlite3"
    assert build_parser().parse_args(["label-toxicity"]).workspace != args.workspace
