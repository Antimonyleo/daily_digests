"""Literal topic matching shared by ranking and reader explanations."""
import re


def term_pattern(term: str) -> str:
    return r'(?<!\w)' + r'\s+'.join(re.escape(word) for word in term.split()) + r'(?!\w)'


def contains_term(text: str, term: str) -> bool:
    return bool(term.strip() and re.search(term_pattern(term), text, re.IGNORECASE))
