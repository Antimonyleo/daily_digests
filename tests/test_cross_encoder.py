"""Cross-encoder plumbing tests; no downloads or real model inference."""
import json
import socket
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dailydigest.rank.cross_encoder import LocalCrossEncoder, pair_input_diagnostics


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx/model.onnx").write_bytes(b"fake-model")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 512}))
    state = SimpleNamespace(chunks=[], feeds=[], shape=None, nonfinite=False)

    def no_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden")
    monkeypatch.setattr(socket, "create_connection", no_network)

    class Tokenizer:
        @staticmethod
        def from_file(path):
            state.tokenizer_path = path
            return Tokenizer()

        def token_to_id(self, token):
            return 0 if token == "[PAD]" else None

        def enable_truncation(self, **kwargs):
            state.truncation = kwargs

        def enable_padding(self, **kwargs):
            state.padding = kwargs

        def encode_batch(self, chunk):
            state.chunks.append(chunk)
            if chunk[0][0] == "oversize":
                raise ValueError("Sequence to truncate too short")
            return [SimpleNamespace(ids=[int(query), 0], attention_mask=[1, 0],
                                    type_ids=[0, 1]) for query, _ in chunk]

    class Session:
        def __init__(self, path, **kwargs):
            state.session_kwargs = kwargs

        def get_inputs(self):
            return [SimpleNamespace(name=name) for name in
                    ("input_ids", "attention_mask", "token_type_ids")]

        def get_outputs(self):
            return [SimpleNamespace(name="logits")]

        def run(self, names, feed):
            state.feeds.append(feed)
            if state.shape is not None:
                return [np.zeros(state.shape)]
            result = feed["input_ids"][:, :1].astype(float) - 2
            if state.nonfinite:
                result[0, 0] = np.nan
            return [result]

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=Session))
    return tmp_path, state


def test_pair_order_raw_logits_batches_padding_and_query_preservation(fake_model):
    path, state = fake_model
    scorer = LocalCrossEncoder(path, max_length=128, batch_size=2, threads=3)
    pairs = [("1", "first"), ("2", "second"), ("3", "third")]
    np.testing.assert_array_equal(scorer.score_pairs(pairs), [-1., 0., 1.])
    assert state.chunks == [pairs[:2], pairs[2:]]
    assert state.truncation == {"max_length": 128, "strategy": "only_second"}
    assert state.padding == {"pad_id": 0, "pad_token": "[PAD]"}
    assert state.session_kwargs["providers"] == ["CPUExecutionProvider"]
    assert state.session_kwargs["sess_options"].intra_op_num_threads == 3
    for feed in state.feeds:
        assert all(array.dtype == np.int64 for array in feed.values())
    assert scorer.provenance["score_kind"] == "raw_logit"


def test_empty_batch_does_not_infer(fake_model):
    path, state = fake_model
    result = LocalCrossEncoder(path).score_pairs([])
    assert result.shape == (0,)
    assert not state.chunks and not state.feeds


@pytest.mark.parametrize("pairs", [[("", "article")], [("query", " ")],
                                    [("1",)], [(3, "article")], ["ab"]])
def test_invalid_pairs_fail_before_inference(fake_model, pairs):
    path, state = fake_model
    with pytest.raises(ValueError, match="Each pair"):
        LocalCrossEncoder(path).score_pairs(pairs)
    assert not state.feeds


def test_oversized_query_is_rejected_not_truncated(fake_model):
    path, _ = fake_model
    with pytest.raises(ValueError, match="query may exceed"):
        LocalCrossEncoder(path).score_pairs([("oversize", "article")])


@pytest.mark.parametrize("shape", [(1, 2), (), (2, 1)])
def test_multioutput_or_wrong_batch_rejected(fake_model, shape):
    path, state = fake_model
    state.shape = shape
    with pytest.raises(ValueError, match="one logit per pair"):
        LocalCrossEncoder(path).score_pairs([("1", "article")])


def test_nonfinite_logits_rejected(fake_model):
    path, state = fake_model
    state.nonfinite = True
    with pytest.raises(ValueError, match="non-finite"):
        LocalCrossEncoder(path).score_pairs([("1", "article")])


@pytest.mark.parametrize("name", ["onnx/model.onnx", "tokenizer.json", "config.json"])
def test_missing_assets_fail_loudly_without_download(fake_model, name):
    path, _ = fake_model
    (path / name).unlink()
    with pytest.raises(FileNotFoundError, match="Required local"):
        LocalCrossEncoder(path)


@pytest.mark.parametrize("kwargs", [{"max_length": 0}, {"batch_size": -1},
                                     {"threads": True}, {"max_length": 513}])
def test_invalid_settings_rejected(fake_model, kwargs):
    path, _ = fake_model
    with pytest.raises(ValueError):
        LocalCrossEncoder(path, **kwargs)


def test_provenance_stable_sensitive_to_assets_and_settings(fake_model):
    path, _ = fake_model
    original = LocalCrossEncoder(path).provenance
    assert LocalCrossEncoder(path).provenance == original
    assert LocalCrossEncoder(path, max_length=128).provenance != original
    (path / "tokenizer.json").write_text('{"changed":true}')
    changed = LocalCrossEncoder(path).provenance
    assert changed["fingerprint"] != original["fingerprint"]
    assert changed["assets_sha256"]["tokenizer.json"] != original["assets_sha256"]["tokenizer.json"]


def test_retention_diagnostics_report_late_evidence_without_changing_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
                                            "query": 3, "background": 4, "engineered": 5, "protein": 6}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    tokenizer.enable_truncation(max_length=8, strategy="only_second")
    before = tokenizer.to_str()
    article = "background background background background engineered protein"
    report = pair_input_diagnostics(tokenizer, [("query", article)], max_length=8)[0]
    assert report["query_tokens"] == report["retained_query_tokens"] == 1
    assert report["document_tokens"] == 6
    assert report["document_token_budget"] == report["retained_document_tokens"] == 4
    assert report["document_truncated"] is True
    assert report["retained_document_prefix"] == "background background background background"
    assert "engineered" not in report["retained_document_prefix"]
    assert tokenizer.to_str() == before
    short = pair_input_diagnostics(tokenizer, [("query", "engineered protein")], max_length=8)[0]
    assert short["document_truncated"] is False
    assert short["retained_document_prefix"] == "engineered protein"


def test_retention_diagnostics_refuse_query_truncation():
    from tokenizers import Tokenizer, models, pre_tokenizers
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "q": 1, "doc": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    with pytest.raises(ValueError, match="query may exceed"):
        pair_input_diagnostics(tokenizer, [("q q q q q", "doc")], max_length=3)
