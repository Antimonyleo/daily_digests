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
            'label': 'Graphical abstract' if kind == 'graphical_abstract' else 'Article image'}


# A label-only text block placed directly before an image ("Graphical abstract").
_LABEL_ONLY = re.compile(r'^(?:' + _GRAPHIC.pattern + r')\s*:?$', re.I)
# The text directly after an item's only image names it ("The graphical abstract
# shows ...", "Table of contents graphic: ..."). Wiley's eTOC captions do this.
_DECLARES = re.compile(r'^(?:(?:the|this)\s+)?(?:' + _GRAPHIC.pattern + r')\b', re.I)


class _Images(HTMLParser):
    """Collect feed images with only the text that explicitly frames each one."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images = []
        self._previous = ''      # last text block seen before the next <img>
        self._following = None   # text after the latest accepted image

    def handle_data(self, data):
        text = ' '.join(data.split())
        if not text:
            return
        self._previous = text
        if self._following is not None and len(self._following) < 200:
            self._following = (self._following + ' ' + text).strip()

    def handle_starttag(self, tag, attrs):
        if tag != 'img':
            return
        labelled = bool(_LABEL_ONLY.match(self._previous))
        self._previous = ''
        self._close_following()
        if len(self.images) >= 8:
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
        if url and all(image['url'] != url for image in self.images):
            evidence = 'alt_text' if _GRAPHIC.search(alt) else 'preceding_label' if labelled else ''
            image = {'url': url, 'kind': 'graphical_abstract' if evidence else 'article_preview',
                     'alt': alt[:500], 'provenance': 'feed_html'}
            if evidence:
                image['label_evidence'] = evidence
            self.images.append(image)
            self._following = ''

    def _close_following(self):
        if self._following is not None and self.images:
            self.images[-1]['_following'] = self._following
        self._following = None

    def result(self) -> list[dict]:
        self._close_following()
        images = [dict(image) for image in self.images]
        # Text after an image only identifies it when it is the sole image.
        sole = images[0] if len(images) == 1 else None
        if sole and sole['kind'] != 'graphical_abstract' and _DECLARES.match(sole.get('_following', '')):
            sole.update(kind='graphical_abstract', label_evidence='following_caption')
        for image in images:
            image.pop('_following', None)
        return images


def feed_figure(entry) -> dict | None:
    """Keep supplied images only; do not scrape publishers or guess image URLs.

    An image is a graphical abstract only when the feed itself says so: in its
    alt/title text, in a label-only block directly before it, or in the text
    directly after an item's only image. Anything else is an article preview.
    """
    images = []
    values = [value.get('value', '') for value in (entry.get('content') or []) if isinstance(value, dict)]
    values.append(entry.get('summary') or entry.get('description') or '')
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        parser = _Images()
        parser.feed(value[:64000])
        parser.close()
        for image in parser.result():
            known = next((seen for seen in images if seen['url'] == image['url']), None)
            if known is None:
                images.append(image)
            elif image['kind'] == 'graphical_abstract' and known['kind'] != 'graphical_abstract':
                known.update(image)
    images = images[:8]
    if images:
        return next((image for image in images if image['kind'] == 'graphical_abstract'), images[0])
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
