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


# --- Explicit graphical-abstract evidence --------------------------------------
# Wiley eTOC fixtures below reuse the image URL, alt text and caption text stored
# for these records in the 2026-10-23 public snapshot (browser/.build-cache). Raw
# feed HTML is not retained, so the markup around those values is the minimal
# content:encoded shape implied by the stored abstract (image, then caption).
WILEY_DECLARED = [
    ('https://onlinelibrary.wiley.com/cms/asset/38e2d4cf-2911-4d45-9735-dd1df61b7b06/adma75449-gra-0001-m.png',
     'Mechanoactive Biomaterials for Cell–Matrix Mechanotransduction',
     'The graphical abstract shows how individual triggers (force, light, or temperature) and combined '
     'optothermal or receptor‐targeted actuation induce material responses.'),
    ('https://onlinelibrary.wiley.com/cms/asset/92ae942a-2193-47ad-9fc5-9b78c6ed99e7/smll76381-gra-0001-m.png',
     'From Fiber to Electrode: Structure, Property and Advanced Characterization of Textile Carbon Electrodes',
     'The table of contents graphic summarizes electrode materials and characterization strategies for '
     'iron‐chromium redox flow batteries.'),
]


def _wiley_entry(url, alt, caption, abstract='ABSTRACT A longer abstract follows the caption.'):
    feed = feedparser.parse(f'''<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel><title>Wiley</title><item><title>{alt}</title><link>https://onlinelibrary.wiley.com/doi/10.1002/x?af=R</link>
<description>{caption[:80]}</description>
<content:encoded><![CDATA[<div><p><img src="{url}" alt="{alt}"/></p><p>{caption}</p> {abstract}</div>]]></content:encoded>
</item></channel></rss>''')
    return feed.entries[0]


@pytest.mark.parametrize('url,alt,caption', WILEY_DECLARED)
def test_caption_that_names_the_sole_image_marks_a_graphical_abstract(url, alt, caption):
    from dailydigest.article_figure import display_figure
    metadata = _entry_metadata(_wiley_entry(url, alt, caption))
    figure = metadata['article_figure']
    assert figure == {'url': url, 'kind': 'graphical_abstract', 'alt': alt,
                      'provenance': 'feed_html', 'label_evidence': 'following_caption'}
    assert display_figure(metadata)['label'] == 'Graphical abstract'


def test_undeclared_wiley_asset_stays_an_article_image_despite_its_file_name():
    from dailydigest.article_figure import display_figure
    url = 'https://advanced.onlinelibrary.wiley.com/cms/asset/a0584762-019c-46d8-9314-fbb17060e97c/adma75394-gra-0001-m.png'
    metadata = _entry_metadata(_wiley_entry(url, 'Targeting Endoplasmic Reticulum Stress',
                                            'An ER‐regulating heterojunction converts pathological oxidative cues.'))
    assert metadata['article_figure']['kind'] == 'article_preview'
    assert 'label_evidence' not in metadata['article_figure']
    assert display_figure(metadata)['label'] == 'Article image'


@pytest.mark.parametrize('caption', [
    'As displayed in Graphical Abstract image.tif, a molecular‐design strategy tailors droplet dynamics.',
    'Graphical abstracts are increasingly common in materials journals.',
    'A heterojunction is shown. The graphical abstract was created with an image tool.',
])
def test_incidental_mentions_do_not_relabel_an_image(caption):
    url = 'https://onlinelibrary.wiley.com/cms/asset/x/advs76684-gra-0001-m.png'
    assert _entry_metadata(_wiley_entry(url, 'A droplet paper', caption))['article_figure']['kind'] == 'article_preview'


def test_caption_cannot_choose_between_several_images():
    html = ('<p><img src="https://publisher.test/fig1.png" alt="Figure 1"></p>'
            '<p>The graphical abstract summarizes the method.</p>'
            '<p><img src="https://publisher.test/fig2.png" alt="Figure 2"></p>')
    figure = _entry_metadata({'summary': html})['article_figure']
    assert figure['url'] == 'https://publisher.test/fig1.png'
    assert figure['kind'] == 'article_preview'


def test_label_block_directly_before_an_image_marks_only_that_image():
    html = ('<p><img src="https://publisher.test/fig1.png" alt="Figure 1"></p>'
            '<h3>Graphical abstract:</h3><img src="https://publisher.test/ga.png" alt="Scheme">')
    figure = _entry_metadata({'summary': html})['article_figure']
    assert figure['url'] == 'https://publisher.test/ga.png'
    assert figure['label_evidence'] == 'preceding_label'
    separated = '<h3>Graphical abstract</h3><p>Some prose first.</p><img src="https://publisher.test/a.png" alt="Scheme">'
    assert _entry_metadata({'summary': separated})['article_figure']['kind'] == 'article_preview'


def test_same_image_in_content_and_summary_is_still_the_sole_image():
    url, alt, caption = WILEY_DECLARED[0]
    body = f'<p><img src="{url}" alt="{alt}"></p><p>{caption}</p>'
    entry = {'content': [{'value': body}], 'summary': body}
    assert _entry_metadata(entry)['article_figure']['kind'] == 'graphical_abstract'


def test_preprint_graphical_abstract_placeholder_is_not_resolved_to_a_guessed_url():
    # medRxiv/bioRxiv abstracts declare a graphical abstract but give only a
    # relative FIGDIR placeholder; resolving it would mean guessing a URL.
    summary = ('A placental mechanism contributing to sPTB. Graphical Abstract O_FIG O_LINKSMALLFIG WIDTH=200 '
               'HEIGHT=113 SRC="FIGDIR/small/26362823v1_ufig1.gif" ALT="Figure 1"> View larger version (25K): '
               '<img src="FIGDIR/small/26362823v1_ufig1.gif" alt="Figure 1"> C_FIG')
    assert 'article_figure' not in _entry_metadata({'summary': summary})


def test_media_thumbnail_is_labelled_article_image_for_readers():
    from dailydigest.article_figure import display_figure
    metadata = _entry_metadata({'media_thumbnail': [{'url': 'https://ichef.example.org/news/1.jpg'}]})
    assert display_figure(metadata)['label'] == 'Article image'


def test_figure_entry_names_the_declared_label_and_credits_the_feed():
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader
    from dailydigest.article_figure import display_figure
    env = Environment(loader=FileSystemLoader(Path(__file__).parents[1] / 'templates'), autoescape=True)
    template = env.get_template('_article_figure.html.j2')
    for kind, label in [('article_preview', 'Article image'), ('graphical_abstract', 'Graphical abstract')]:
        figure = display_figure({'article_figure': {'url': 'https://publisher.test/a.png', 'kind': kind}})
        html = template.render(item={'title': 'A paper', 'url': 'https://publisher.test/paper',
                                     'source': 'Small (Wiley)', 'figure': figure})
        # The cached reading view reads the label as the summary text before " · ".
        assert f'<summary><span class="figure-label">{label}</span> · ' in html
        assert f'data-figure-kind="{kind}"' in html
        assert f'{label} supplied in the Small (Wiley) feed' in html
        assert 'class="figure-full" href="https://publisher.test/a.png"' in html
        assert 'class="figure-unavailable" role="status" hidden' in html
    other = 'Graphical abstract' if label == 'Article image' else 'Article image'
    assert other not in html
