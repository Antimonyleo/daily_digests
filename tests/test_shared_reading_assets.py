"""Cached and Python desks must use the same authored Pip implementation."""
import json

import pytest

from tests.test_browser_build import build


def test_shared_pip_markup_is_release_versioned():
    source = "fetch(new URL('./pip.html', import.meta.url));"
    assert "./pip.html?build=new-release" in build.stamp_urls(source, 'new-release')


@pytest.mark.parametrize('source', ['', 'export const SHARED_READING_STYLES = null;\n' * 2])
def test_missing_or_ambiguous_style_marker_stops_build(tmp_path, source):
    (tmp_path / 'reading-view.mjs').write_text(source)
    with pytest.raises(ValueError, match='exactly once'):
        build.embed_shared_reading_styles(tmp_path)


def test_public_pip_assets_come_from_shared_templates_not_reader_state(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    templates = root / 'templates'
    templates.mkdir(parents=True)
    (templates / '_pip.js.j2').write_text('function mountPip(root, options) { return () => {}; }\n')
    (templates / '_design_tokens.css.j2').write_text(':root { --bg: mint; }\n')
    (templates / '_reading_surface.css.j2').write_text('body { background: var(--bg); }\n')
    (templates / '_pip.css.j2').write_text('.tea-pet { color: var(--accent); }\n')
    (templates / '_pip.html.j2').write_text('<span id="tea-note-text">{{ daily_note }}</span>')
    (root / 'data').mkdir()
    (root / 'data/profile.yaml').write_text('name: PRIVATE_READER_SENTINEL')
    (root / '.env').write_text('SECRET_SENTINEL=never-public')
    output = tmp_path / 'site'
    output.mkdir()
    (output / 'reading-view.mjs').write_text('export const SHARED_READING_STYLES = null;\n')
    monkeypatch.setattr(build, 'ROOT', root)
    build.embed_shared_reading_styles(output)
    build.publish_shared_reading_assets(output)
    assert (output / 'reading-surface.css').read_bytes() == (templates / '_reading_surface.css.j2').read_bytes()
    assert (output / 'pip.css').read_bytes() == (templates / '_pip.css.j2').read_bytes()
    module = (output / 'pip.mjs').read_text()
    assert module.startswith((templates / '_pip.js.j2').read_text())
    assert 'export { mountPip };' in module
    assert (output / 'pip.html').read_text() == '<span id="tea-note-text"></span>'
    reading_source = (output / 'reading-view.mjs').read_text()
    embedded = json.loads(reading_source.removeprefix('export const SHARED_READING_STYLES = ').strip().removesuffix(';'))
    assert embedded == ':root { --bg: mint; }\n\nbody { background: var(--bg); }\n'
    assert all('SENTINEL' not in path.read_text() for path in output.iterdir())
