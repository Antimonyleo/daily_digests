"""Explicit-path, CPU-only ONNX cross-encoder for offline ranking experiments.

This module never downloads a model. Scores are raw model logits, not calibrated
probabilities or evidence that a paper belongs to a reader's research scope.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def pair_input_diagnostics(tokenizer, pairs: list[tuple[str, str]], *, max_length: int) -> list[dict]:
    """Inspect exact tokenizer retention locally without inference or mutation.

    Returned prefixes are supplied article text, not extracted contributions or
    relevance judgments. A clone removes padding/truncation for original counts;
    another applies the same query-preserving policy used by this encoder.
    """
    from tokenizers import Tokenizer

    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    _validate_pairs(pairs)
    raw = Tokenizer.from_str(tokenizer.to_str())
    raw.no_padding()
    raw.no_truncation()
    bounded = Tokenizer.from_str(raw.to_str())
    bounded.enable_truncation(max_length=max_length, strategy="only_second")
    reports = []
    for query, article in pairs:
        full = raw.encode(query, article)
        try:
            retained = bounded.encode(query, article)
        except Exception as exc:
            raise ValueError("Pair tokenization failed; query may exceed token capacity") from exc
        query_tokens = sum(sequence == 0 for sequence in full.sequence_ids)
        document_tokens = sum(sequence == 1 for sequence in full.sequence_ids)
        retained_query = sum(sequence == 0 for sequence in retained.sequence_ids)
        retained_document = sum(sequence == 1 for sequence in retained.sequence_ids)
        if query_tokens != retained_query:
            raise ValueError("Diagnostics must not truncate the query")
        end = max((offset[1] for sequence, offset in zip(retained.sequence_ids, retained.offsets, strict=True)
                   if sequence == 1), default=0)
        reports.append({
            "query_tokens": query_tokens, "document_tokens": document_tokens,
            "retained_query_tokens": retained_query, "retained_document_tokens": retained_document,
            "pair_tokens_before": len(full.ids), "pair_tokens_after": len(retained.ids),
            "max_length": max_length,
            "document_token_budget": max(0, max_length - (len(full.ids) - document_tokens)),
            "document_truncated": retained_document < document_tokens,
            "retained_document_end_char": end, "retained_document_prefix": article[:end],
            "meaning": "Tokenizer retention only; omitted text can contain the central contribution",
        })
    return reports


def _validate_pairs(pairs):
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2 or not all(
                isinstance(value, str) and value.strip() for value in pair):
            raise ValueError("Each pair must contain a nonempty query and article")


class LocalCrossEncoder:
    """Score ordered (query, article) pairs using pre-provisioned local assets."""

    def __init__(self, model_dir: str | Path, max_length: int = 512,
                 batch_size: int = 16, threads: int = 2):
        for name, value in (("max_length", max_length), ("batch_size", batch_size),
                            ("threads", threads)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        root = Path(model_dir).expanduser().resolve(strict=True)
        model = root / "onnx" / "model.onnx"
        tokenizer_path = root / "tokenizer.json"
        config_path = root / "config.json"
        for path in (model, tokenizer_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required local cross-encoder asset: {path}")
        config = json.loads(config_path.read_text())
        positional_limit = config.get("max_position_embeddings")
        if positional_limit is not None and max_length > int(positional_limit):
            raise ValueError("max_length exceeds model positional capacity")

        # Lazy imports keep normal ranking independent of this experimental path.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        pad_id = self._tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise ValueError("Tokenizer must define [PAD]")
        self._tokenizer.enable_truncation(max_length=max_length, strategy="only_second")
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self._inputs = {item.name for item in self._session.get_inputs()}
        if not {"input_ids", "attention_mask"} <= self._inputs or not self._inputs <= {
                "input_ids", "attention_mask", "token_type_ids"}:
            raise ValueError(f"Unsupported cross-encoder inputs: {sorted(self._inputs)}")
        outputs = self._session.get_outputs()
        if len(outputs) != 1:
            raise ValueError("Cross-encoder must expose one scalar-logit output")
        self._output = outputs[0].name
        self.batch_size = batch_size
        assets = {}
        # Include external ONNX tensor files and tokenizer/config companions.
        for path in sorted(root.rglob("*")):
            if path.is_file():
                with path.open("rb") as handle:
                    assets[path.relative_to(root).as_posix()] = hashlib.file_digest(
                        handle, "sha256").hexdigest()
        details = {
            "assets_sha256": assets, "max_length": max_length,
            "truncation": "only_second", "padding": "longest_in_batch",
            "batch_size": batch_size, "threads": threads,
            "providers": ["CPUExecutionProvider"], "score_kind": "raw_logit",
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "tokenizers_version": importlib.metadata.version("tokenizers"),
        }
        self.provenance = {**details, "fingerprint": hashlib.sha256(
            json.dumps(details, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}

    def input_diagnostics(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Explicit offline inspection; never changes scores or model provenance."""
        return pair_input_diagnostics(self._tokenizer, pairs, max_length=self.provenance["max_length"])

    def score_records(self, records: list[dict], *, check_cancel=None) -> list[dict]:
        """Score title/abstract windows without losing queries or late conclusions.

        Max window logits are relative match signals. Incomplete coverage must
        never be interpreted as evidence that the paper is off-interest.
        """
        from .semantic_windows import prepare_semantic_records

        if not hasattr(self, "_window_tokenizer"):
            from tokenizers import Tokenizer

            self._window_tokenizer = Tokenizer.from_str(self._tokenizer.to_str())
            self._window_tokenizer.no_truncation()
            self._window_tokenizer.no_padding()
        prepared = prepare_semantic_records(self._window_tokenizer, records)
        result = [{**record, "logit": float("-inf")} for record in prepared["records"]]
        windows = prepared["windows"]
        for start in range(0, len(windows), self.batch_size):
            if check_cancel is not None:
                check_cancel()
            batch = windows[start:start + self.batch_size]
            width = max(len(row["input_ids"]) for row in batch)
            feed = {name: np.asarray([
                row[name] + [0] * (width - len(row[name])) for row in batch
            ], dtype=np.int64) for name in self._inputs}
            logits = np.asarray(self._session.run([self._output], feed)[0], dtype=np.float64)
            if logits.shape not in ((len(batch),), (len(batch), 1)) or not np.isfinite(logits).all():
                raise ValueError("Semantic encoder must return one finite logit per window")
            for row, value in zip(batch, logits.reshape(-1), strict=True):
                summary = result[row["recordIndex"]]
                summary["logit"] = max(summary["logit"], float(value))
        if check_cancel is not None:
            check_cancel()
        return result

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Return one finite raw logit per pair; never silently truncate queries."""
        _validate_pairs(pairs)
        if not pairs:
            return np.empty(0, dtype=np.float64)
        batches = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start + self.batch_size]
            try:
                encoded = self._tokenizer.encode_batch(chunk)
            except Exception as exc:
                raise ValueError("Pair tokenization failed; query may exceed token capacity") from exc
            feed = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64),
            }
            outputs = self._session.run([self._output], {
                name: feed[name] for name in self._inputs})
            logits = np.asarray(outputs[0], dtype=np.float64)
            if logits.shape not in ((len(chunk),), (len(chunk), 1)):
                raise ValueError(f"Expected one logit per pair; received {logits.shape}")
            if not np.isfinite(logits).all():
                raise ValueError("Cross-encoder returned non-finite logits")
            batches.append(logits.reshape(-1))
        return np.concatenate(batches)
