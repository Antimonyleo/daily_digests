"""Optional publisher-feed graphics; presentation metadata never affects ranking."""
from __future__ import annotations

import ipaddress
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

_GRAPHIC = re.compile(r'graphical[\s_-]+abstract|\btoc[\s_-]+(?:figure|graphic|image)|table of contents (?:figure|graphic|image)', re.I)
_NOISE = re.compile(r'\b(?:logo|icon|tracker|tracking|beacon|journal cover)\b', re.I)
_URL_NOISE = re.compile(r'(?:^|[/_.-])(?:logos?|icons?|tracking|tracker|beacon|covers?)(?:$|[/_.-])', re.I)


def image_url(value: object) -> str:
    """Accept public HTTPS image URLs without credentials or local hosts."""
    if not isinstance(value, str) or len(value) > 8000 or any(ord(c) < 33 for c in value):
        return ''
    try:
        url = urlsplit(value)
        host = (url.hostname or '').lower().rstrip('.')
        if (url.scheme != 'https' or not host or '.' not in host or url.username or url.password
                or host.endswith(('.localhost', '.local', '.internal')) or url.port not in {None, 443}
                or host.rsplit('.', 1)[-1].isdigit() or '\\' in value or '%' in host
                or _URL_NOISE.search(url.path)):
            return ''
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return ''
    except ValueError:
        return ''
    return value


def display_figure(metadata: dict | None) -> dict | None:
    """Revalidate imported metadata before rendering it in a reader's browser."""
    value = (metadata or {}).get('article_figure')
    if not isinstance(value, dict) or not image_url(value.get('url')):
        return None
    kind = value.get('kind')
    if kind not in {'graphical_abstract', 'article_preview'}:
        return None
    return {'url': value['url'], 'kind': kind,
            'alt': str(value.get('alt') or '')[:500],
            'label': 'Graphical abstract' if kind == 'graphical_abstract' else 'Article preview'}


class _Images(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag != 'img' or len(self.images) >= 8:
            return
        attrs = dict(attrs)
        alt = attrs.get('alt') or attrs.get('title') or ''
        if _NOISE.search(alt):
            return
        for dimension in ('width', 'height'):
            value = str(attrs.get(dimension) or '').removesuffix('px')
            if value.isdigit() and int(value) <= 48:
                return
        url = image_url(attrs.get('src'))
        if url:
            self.images.append({'url': url,
                                'kind': 'graphical_abstract' if _GRAPHIC.search(alt) else 'article_preview',
                                'alt': alt[:500], 'provenance': 'feed_html'})


def feed_figure(entry) -> dict | None:
    """Keep supplied images only; do not scrape publishers or guess image URLs."""
    parser = _Images()
    values = [value.get('value', '') for value in (entry.get('content') or []) if isinstance(value, dict)]
    values.append(entry.get('summary') or entry.get('description') or '')
    for value in values:
        if isinstance(value, str):
            parser.feed(value[:64000])
    if parser.images:
        return next((image for image in parser.images if image['kind'] == 'graphical_abstract'), parser.images[0])
    for field in ('media_content', 'media_thumbnail'):
        for value in entry.get(field, [])[:8]:
            if not isinstance(value, dict):
                continue
            if field == 'media_content' and value.get('medium') != 'image' and not str(value.get('type', '')).startswith('image/'):
                continue
            url = image_url(value.get('url'))
            if url:
                return {'url': url, 'kind': 'article_preview', 'alt': '', 'provenance': field}
    return None
