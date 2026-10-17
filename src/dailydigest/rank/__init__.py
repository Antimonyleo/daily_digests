"""Ranking entry points, loaded on first use rather than when displaying a page."""
from importlib import import_module

__all__ = ['embed_texts', 'build_profile_vector', 'score_items', 'pick_top_per_section']


def __getattr__(name):
    modules = {'embed_texts': 'embed', 'build_profile_vector': 'profile',
               'score_items': 'ranker', 'pick_top_per_section': 'ranker'}
    if name not in modules:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'.{modules[name]}', __name__), name)
    globals()[name] = value
    return value
