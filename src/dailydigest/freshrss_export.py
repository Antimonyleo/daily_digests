"""
Export DailyDigest sources to FreshRSS-compatible OPML format.

This module reads config/sources.yaml and generates an OPML 2.0 file
that can be imported directly into FreshRSS. Non-RSS sources (arxiv, openalex,
etc.) are listed in a comment since FreshRSS can only consume RSS feeds.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from .config import load_sources


def export_opml(out_path: str = "data/sources.opml") -> int:
    """
    Generate OPML file from sources.yaml.

    Args:
        out_path: Path where the OPML file will be written.

    Returns:
        Total count of RSS sources exported.
    """
    sources = load_sources()

    # Build OPML document
    opml = ET.Element("opml")
    opml.set("version", "2.0")

    # Header
    head = ET.SubElement(opml, "head")
    title = ET.SubElement(head, "title")
    title.text = "DailyDigest Sources"
    date_created = ET.SubElement(head, "dateCreated")
    date_created.text = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Body: group by section
    body = ET.Element("body")
    opml.append(body)

    rss_count = 0
    skipped_sources = []

    # Group sources by section
    sections = {}
    for source in sources:
        section = source.section or "research"
        if section not in sections:
            sections[section] = []
        sections[section].append(source)

    # Iterate sections in a consistent order
    for section in sorted(sections.keys()):
        section_outline = ET.SubElement(body, "outline")
        section_outline.set("text", section.capitalize())
        section_outline.set("title", section.capitalize())

        for source in sections[section]:
            # Only include RSS sources; skip special kinds
            if source.kind == "rss":
                if source.url:
                    feed_outline = ET.SubElement(section_outline, "outline")
                    feed_outline.set("type", "rss")
                    feed_outline.set("text", source.name)
                    feed_outline.set("title", source.name)
                    feed_outline.set("xmlUrl", source.url)
                    rss_count += 1
            else:
                # Track non-RSS sources for the info message
                skipped_sources.append(f"{source.name} ({source.kind})")

    # Write file
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(opml)
    ET.indent(tree, space="  ")
    tree.write(out_file, encoding="utf-8", xml_declaration=True)

    # Log skipped sources if any
    if skipped_sources:
        print(f"\nNote: {len(skipped_sources)} non-RSS sources were skipped")
        print("(FreshRSS can only consume RSS feeds directly)")
        print("Skipped sources:")
        for name in skipped_sources:
            print(f"  - {name}")

    return rss_count


if __name__ == "__main__":
    count = export_opml()
    print(f"\nGenerated data/sources.opml with {count} RSS feeds.")
    print("Import this file in FreshRSS: Settings → Import/Export → Import")
