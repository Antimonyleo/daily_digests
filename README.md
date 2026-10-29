# DailyDigest

A private research reading desk: discover relevant papers and news, save what matters, and improve future recommendations with your ratings.

**[Open DailyDigest in your browser](https://antimonyleo.github.io/daily_digests/)** — no installation, paid AI subscription, or API key required. Start with a few research topics, then brew a digest sized for the time you have.

- Personalized papers and optional funding, events, industry, and news.
- Local extractive summaries, recommendation explanations, and saved reading.
- Your interests, ratings, and history stay on your device. Optional online summary providers receive selected article metadata and a short profile only when configured.

## Choose how to use it

| Option | Best for |
| --- | --- |
| [Hosted website](https://antimonyleo.github.io/daily_digests/) | Getting started without installing software; use current desktop Chrome |
| [Local app](docs/user-guide.md#install-the-local-app) | Running DailyDigest on your computer, including email and scheduling |
| [Optional Companion](docs/companion-guide.md) | Adding scheduling, email, or local provider credentials while continuing to use the website |

The website downloads its runtime on first use. Keep its tab open while brewing. Browser data belongs to that browser profile; **More → Backup & transfer** protects it before you move or clear storage. Full brewing on Safari, Firefox, and physical phones is not yet verified.

## Documentation

- [User guide: setup, everyday use, updates, and troubleshooting](docs/user-guide.md)
- [Companion: benefits, setup, and automatic brewing](docs/companion-guide.md)
- [Local backup and recovery](docs/local-recovery.md)
- [Email domain setup](docs/domain-setup.md)
- [Browser architecture, build, and verification](browser/README.md)
- [Developer guide](docs/development.md)

## Development

```bash
uv sync --frozen --group dev
uv run pytest
uv run ruff check --select F,B src tests
```

Keep local-only defaults and use fixtures for external services in tests. Do not commit personal `data/` or `.env` files. See [LICENSE](LICENSE).
