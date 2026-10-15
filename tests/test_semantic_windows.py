"""Exact query-preserving windows for the native and browser topic matcher."""
from types import SimpleNamespace

import numpy as np
import pytest


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return SimpleNamespace(ids=[int(word) for word in text.split()] if text else [])


def text(count, start=10):
    return ' '.join(str(i) for i in range(start, start + count))


def test_short_input_preserves_query_title_abstract_and_segments():
    from dailydigest.rank.semantic_windows import prepare_semantic_records
    result = prepare_semantic_records(Tokenizer(), [{'query': '11 12', 'title': '21', 'abstract': '31 32'}])
    assert result['records'] == [{'windows': 1, 'complete': True}]
    assert result['windows'] == [{'recordIndex': 0, 'input_ids': [101, 11, 12, 102, 21, 31, 32, 102],
        'attention_mask': [1] * 8, 'token_type_ids': [0] * 4 + [1] * 4}]


def test_long_abstract_keeps_late_contribution_and_reports_missing_middle():
    from dailydigest.rank.semantic_windows import prepare_semantic_records
    record = {'query': '11', 'title': '21', 'abstract': text(2400, 1000)}
    result = prepare_semantic_records(Tokenizer(), [record])
    assert result['records'] == [{'windows': 3, 'complete': False}]
    assert all(len(w['input_ids']) <= 512 for w in result['windows'])
    assert result['windows'][0]['input_ids'][4] == 1000
    assert result['windows'][-1]['input_ids'][-2] == 3399
    assert all(w['input_ids'][:4] == [101, 11, 102, 21] for w in result['windows'])


def test_two_window_abstract_is_complete_and_title_cut_is_explicit():
    from dailydigest.rank.semantic_windows import prepare_semantic_records
    records = [{'query': '11', 'title': '21', 'abstract': text(700)},
               {'query': '11', 'title': text(129), 'abstract': '31'}]
    result = prepare_semantic_records(Tokenizer(), records)
    assert result['records'] == [{'windows': 2, 'complete': True}, {'windows': 1, 'complete': False}]


def test_overlong_query_is_bounded_with_explicit_partial_coverage():
    from dailydigest.rank.semantic_windows import prepare_semantic_records
    result = prepare_semantic_records(Tokenizer(), [
        {'query': text(300), 'topic': '501 502', 'title': '21', 'abstract': '31'}])
    assert result['records'] == [{'windows': 1, 'complete': False, 'query_truncated': True}]
    assert result['windows'][0]['input_ids'][1:5] == [501, 502, 10, 11]
    assert result['windows'][0]['input_ids'][193] == 102
    fallback = prepare_semantic_records(Tokenizer(), [
        {'query': text(193), 'title': '21', 'abstract': '31'}])
    assert fallback['records'][0]['query_truncated']
    assert fallback['windows'][0]['input_ids'][1:193] == list(range(10, 202))


def test_long_topic_is_bounded_to_leave_room_for_description():
    from dailydigest.rank.semantic_windows import prepare_semantic_records
    result = prepare_semantic_records(Tokenizer(), [
        {'query': text(300, 1000), 'topic': text(120, 2000), 'title': '21', 'abstract': '31'}])
    query = result['windows'][0]['input_ids'][1:193]
    assert query[:96] == list(range(2000, 2096))
    assert query[96:] == list(range(1000, 1096))


def test_description_within_form_character_limit_can_be_scored_partially():
    from dailydigest.rank.semantic_windows import prepare_semantic_records
    description = '1 ' * 299
    assert len(description) == 598
    result = prepare_semantic_records(Tokenizer(), [
        {'query': description, 'topic': '501', 'title': '21', 'abstract': '31'}])
    assert result['records'] == [{'windows': 1, 'complete': False, 'query_truncated': True}]
    assert result['windows'][0]['input_ids'][1:193] == [501] + [1] * 191


def test_nonempty_query_that_tokenizes_to_nothing_is_rejected():
    from dailydigest.rank.semantic_windows import prepare_semantic_records
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: SimpleNamespace(ids=[]))
    with pytest.raises(ValueError, match='1–192'):
        prepare_semantic_records(tokenizer, [{'query': '\u200b', 'title': '21', 'abstract': '31'}])


def test_native_window_scores_use_max_and_keep_record_identity():
    from dailydigest.rank.cross_encoder import LocalCrossEncoder
    encoder = LocalCrossEncoder.__new__(LocalCrossEncoder)
    encoder._tokenizer = Tokenizer()
    encoder._window_tokenizer = Tokenizer()
    encoder.batch_size = 4
    encoder._inputs = {'input_ids', 'attention_mask', 'token_type_ids'}
    encoder._output = 'logits'
    encoder._session = SimpleNamespace(run=lambda name, feed: [np.asarray(
        [[float(row[4])] for row in feed['input_ids']])])
    records = [{'query': '11', 'title': '21', 'abstract': text(700, 1000)},
               {'query': '12', 'title': '22', 'abstract': '31'}]
    result = encoder.score_records(records)
    assert result[0]['windows'] == 2 and result[0]['complete']
    assert result[0]['logit'] > 1000
    assert result[1] == {'windows': 1, 'complete': True, 'logit': 31.0}
