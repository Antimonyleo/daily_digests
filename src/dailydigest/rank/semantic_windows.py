"""Bounded BERT pair windows shared by the native and browser topic matcher.

The browser implementation in semantic.mjs follows this token protocol exactly.
Coverage describes retained text, not confidence or research validity.
"""
from __future__ import annotations


WINDOW_PROTOCOL = "topic-pair-windows-v2"


def prepare_semantic_records(tokenizer, records: list[dict]) -> dict:
    windows, summaries = [], []
    for index, record in enumerate(records):
        if (not isinstance(record, dict) or not isinstance(record.get("query"), str)
                or not record["query"].strip()
                or any(not isinstance(record.get(key), str) for key in ("title", "abstract"))):
            raise ValueError("Semantic records need a query, title and abstract")
        query = tokenizer.encode(record["query"], add_special_tokens=False).ids
        title = tokenizer.encode(record["title"], add_special_tokens=False).ids
        abstract = tokenizer.encode(record["abstract"], add_special_tokens=False).ids
        if not query:
            raise ValueError("Semantic query must contain 1–192 tokens; shorten its description")
        query_truncated = len(query) > 192
        if query_truncated:
            topic = record.get("topic", "")
            if not isinstance(topic, str):
                raise ValueError("Semantic topic must be a string")
            topic_tokens = tokenizer.encode(topic, add_special_tokens=False).ids[:96]
            query = topic_tokens + query[:192 - len(topic_tokens)]
        complete = len(title) <= 128 and not query_truncated
        title = title[:128]
        budget = 512 - len(query) - len(title) - 3
        overlap = min(32, budget // 4)
        starts, start = [], 0
        while True:
            starts.append(start)
            if start + budget >= len(abstract):
                break
            start += budget - overlap
        if len(starts) > 3:
            starts = [starts[0], starts[len(starts) // 2], starts[-1]]
            complete = False
        summary = {"windows": len(starts), "complete": complete}
        if query_truncated:
            summary["query_truncated"] = True
        summaries.append(summary)
        for start in starts:
            head = [101, *query, 102]
            tail = [*title, *abstract[start:start + budget], 102]
            ids = head + tail
            windows.append({"recordIndex": index, "input_ids": ids,
                            "attention_mask": [1] * len(ids),
                            "token_type_ids": [0] * len(head) + [1] * len(tail)})
    return {"windows": windows, "records": summaries}
