"""Display evidence and reader utility, without changing recommendation policy."""
from __future__ import annotations

import re
from typing import Any


def match_evidence(row: Any, topic: str, aliases=()) -> dict[str, str] | None:
    """Quote a literal whole-phrase occurrence; semantic fit alone is not a quote."""
    for term in [topic, *list(aliases)[:3]]:
        if not isinstance(term, str) or not term.strip() or len(term) > 160:
            continue
        pattern = r'(?<!\w)' + r'\s+'.join(re.escape(word) for word in term.split()) + r'(?!\w)'
        for text in (getattr(row, 'title', '') or '', getattr(row, 'abstract', '') or ''):
            match = re.search(pattern, text, re.I)
            if match:
                start = max(0, match.start() - 45)
                end = min(len(text), start + 240)
                return {'topic': topic, 'term': text[match.start():match.end()], 'quote': text[start:end]}
    return None


def usefulness_summary(entries: list[dict]) -> dict:
    """A descriptive count, not causal evidence that a ranker improved."""
    grades = [row.get('current_grade') for row in entries]
    rated = [grade for grade in grades if type(grade) is int and grade in {10, 40, 70, 100}]
    useful = sum(grade >= 70 for grade in rated)
    return {'selected': len(grades), 'rated': len(rated), 'useful': useful,
            'unrated': len(grades) - len(rated),
            'basis': 'Current explicit ratings only; unrated picks are unknown. Ratings can change after reading.'}


def recent_usefulness(limit: int = 14) -> list[dict]:
    """Describe recent research slates using current ratings, with no inference."""
    from sqlalchemy import select
    from .store import DigestItemRow, DigestRow, ItemRow, VoteRow, init_db, session_scope

    init_db()
    with session_scope() as session:
        digests = list(session.scalars(select(DigestRow.id).order_by(DigestRow.id.desc()).limit(max(1, min(limit, 30)))))
        rows = session.execute(select(DigestItemRow.digest_id, DigestItemRow.item_id, VoteRow.value, VoteRow.grade)
            .join(ItemRow, ItemRow.id == DigestItemRow.item_id)
            .outerjoin(VoteRow, VoteRow.item_id == DigestItemRow.item_id)
            .where(DigestItemRow.digest_id.in_(digests), ItemRow.section == 'research')
            .order_by(DigestItemRow.digest_id.desc(), DigestItemRow.item_id, VoteRow.created_at.desc(), VoteRow.id.desc())).all()
    by_digest = {day: [] for day in digests}
    seen = set()
    for day, item_id, value, grade in rows:
        if (day, item_id) in seen:
            continue
        seen.add((day, item_id))
        if grade is None and value in {-1, 1}:
            grade = 70 if value == 1 else 10
        by_digest[day].append({'current_grade': grade})
    return [{'digest_id': day, **usefulness_summary(entries)} for day, entries in by_digest.items()]
