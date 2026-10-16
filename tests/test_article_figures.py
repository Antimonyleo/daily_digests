"""Publisher-supplied article previews are optional, safe, and accurately labelled."""
import feedparser
import pytest

from dailydigest.ingest.rss import _entry_metadata


def test_graphical_abstract_survives_real_feed_parsing():
    feed = feedparser.parse('''<rss version="2.0"><channel><title>Research</title><item>
<title>A molecular method</title><link>https://publisher.test/article/1</link>
<description><![CDATA[<p>Graphical abstract</p><img src="https://publisher.test/toc/1.png" alt="Graphical abstract: a reaction scheme"/><p>A useful method.</p>]]></description>
</item></channel></rss>''')
    figure = _entry_metadata(feed.entries[0])['article_figure']
    assert figure['url'] == 'https://publisher.test/toc/1.png'
    assert figure['kind'] == 'graphical_abstract'
    assert figure['alt'] == 'Graphical abstract: a reaction scheme'
    assert figure['provenance'] == 'feed_html'


def test_feed_thumbnail_is_not_called_a_graphical_abstract():
    entry = {'media_thumbnail': [{'url': 'https://publisher.test/thumb.png'}]}
    assert _entry_metadata(entry)['article_figure']['kind'] == 'article_preview'


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'data:image/png;base64,abcd',
                               'http://publisher.test/a.png', '//publisher.test/a.png',
                               'https://user:secret@publisher.test/a.png', 'https://localhost/a.png'])
def test_unsafe_or_ambiguous_images_are_ignored(url):
    assert 'article_figure' not in _entry_metadata({'media_thumbnail': [{'url': url}]})


def test_no_image_is_invented_from_article_or_journal_url():
    assert 'article_figure' not in _entry_metadata({'link': 'https://publisher.test/paper/1'})


def test_imported_figure_metadata_is_revalidated_and_bounded():
    from dailydigest.article_figure import display_figure
    assert display_figure({'article_figure': {'url': 'javascript:alert(1)', 'kind': 'graphical_abstract'}}) is None
    assert display_figure({'article_figure': {'url': 'https://publisher.test/a.png', 'kind': 'cover'}}) is None
    value = display_figure({'article_figure': {'url': 'https://publisher.test/a.png',
                                             'kind': 'graphical_abstract', 'alt': 'x' * 1000}})
    assert len(value['alt']) == 500


def test_figure_template_defers_network_and_escapes_publisher_text():
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'), autoescape=True)
    value = {'url': 'https://publisher.test/a.png', 'label': 'Graphical abstract', 'alt': '<script>bad()</script>'}
    html = env.get_template('_article_figure.html.j2').render(item={'title': 'A paper', 'url': 'https://publisher.test/paper', 'figure': value})
    assert 'data-src="https://publisher.test/a.png"' in html
    assert ' src="' not in html
    assert 'crossorigin="anonymous"' in html
    assert 'referrerpolicy="no-referrer"' in html
    assert '<script>bad()' not in html
    assert env.get_template('_article_figure.html.j2').render(item={'figure': None}).strip() == ''


def test_tiny_tracking_pixels_and_journal_logos_are_not_previews():
    for html in ['<img src="https://publisher.test/p.png" width="1">',
                 '<img src="https://publisher.test/logo.png" alt="Journal logo">']:
        assert 'article_figure' not in _entry_metadata({'summary': html})


def test_publisher_figure_survives_store_digest_and_saved_routes(tmp_path, monkeypatch):
    from dailydigest import store, web
    from dailydigest.models import Item
    from tests.test_web import _request
    profile = tmp_path / 'profile.yaml'
    profile.write_text('name: Fixture\nbio: Molecular methods.\nkeywords: [chemistry]\n')
    monkeypatch.setattr(web, '_get_profile_path', lambda: profile)
    monkeypatch.setattr(web, '_digest_id', lambda: '2026-10-16')
    figure = _entry_metadata({'summary': '<img src="https://publisher.test/toc.png" alt="Graphical abstract: a scheme">'})
    store.upsert_items([Item(source='Nature',section='research',external_id='image-fixture',
                            title='A useful molecular method',url='https://publisher.test/paper',metadata=figure)])
    with store.session_scope() as session:
        item = session.query(store.ItemRow).one()
        item_id = int(item.id)
        item.summary = 'A molecular method and its limitations.'
    store.write_digest('2026-10-16', [('R1', item_id, .9)])
    store.set_bookmark(item_id, True)
    digest = web.index(_request('GET', '/')).body.decode()
    saved = web.saved_items(_request('GET', '/saved'), q='', page=1).body.decode()
    for html in (digest, saved):
        assert 'data-src="https://publisher.test/toc.png"' in html
        assert 'Graphical abstract' in html
        assert 'A useful molecular method' in html
    # An image-only metadata update must not discard existing scientific text.
    store.upsert_items([Item(source='Nature',section='research',external_id='image-fixture',
                            title='A useful molecular method',url='https://publisher.test/paper',metadata={})])
    assert store.search_bookmarks()[0].metadata['article_figure'] == figure['article_figure']


@pytest.mark.parametrize('entry', [
    {'summary': '<img src="https://publisher.test/static/journal-logo.png">'},
    {'media_thumbnail': [{'url': 'https://publisher.test/covers/issue.png'}]},
])
def test_unlabelled_logos_and_issue_covers_are_not_article_previews(entry):
    assert 'article_figure' not in _entry_metadata(entry)
