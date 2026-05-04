# FreshRSS Sidecar Setup Guide

**What is this?** A self-hosted Docker stack that runs FreshRSS (a web-based RSS reader) and RSSHub (a route proxy for sites without RSS) alongside DailyDigest. All items ingested by DailyDigest are browsable here as a fallback when you want to dig past the digest's top 20.

**Why optional?** The email digest works without this. The sidecar is for users who want a full-text search UI and want to keep everything.

**Monthly cost:** ~$4–6 on shared-CPU VPS (Hetzner CX11, Fly.io free tier, or similar). Uses ~100MB RAM at rest.

---

## Prerequisites

- **Docker & Docker Compose** installed (v20.10+ and v1.29+ respectively)
- **A domain you control** with DNS management access (e.g., `yourdomain.com`)
- **Ports 80 and 443 open** on your VPS (for HTTP/HTTPS traffic)
- **DNS A record** pointing to your server's IP for two subdomains
  - e.g., `freshrss.yourdomain.com` → `1.2.3.4`
  - e.g., `rsshub.yourdomain.com` → `1.2.3.4`

---

## Step 1: Prepare Configuration

Copy the environment template and edit it:

```bash
cd /path/to/dailydigest/infra
cp .env.example .env
```

Edit `infra/.env`:

```bash
FRESHRSS_DOMAIN=freshrss.yourdomain.com
RSSHUB_DOMAIN=rsshub.yourdomain.com
ACME_EMAIL=your-email@example.com
TZ=America/New_York
CRON_MIN=30
RSSHUB_BASIC_AUTH_USER=admin
RSSHUB_BASIC_AUTH_HASH=
```

For `RSSHUB_BASIC_AUTH_HASH`, generate a bcrypt password hash:

```bash
docker run caddy:2-alpine caddy hash-password --plaintext "your-secure-password"
```

This outputs something like `$2a$14$abc...xyz`. Copy the entire hash into `.env`:

```bash
RSSHUB_BASIC_AUTH_HASH=$2a$14$...
```

---

## Step 2: Launch the Stack

```bash
cd infra
docker compose up -d
```

Check logs:

```bash
docker compose logs -f freshrss caddy
```

Wait for FreshRSS to be healthy (the healthcheck should pass in ~40s):

```bash
docker compose ps
```

Look for `freshrss` with status `Up (healthy)`.

---

## Step 3: FreshRSS First-Run Setup

Visit `https://freshrss.yourdomain.com` in your browser.

1. Accept the first-run wizard.
2. Choose **SQLite** as the database (already configured).
3. Create an admin user and password.
4. Choose language (English recommended).
5. Finish and log in.

---

## Step 4: Import Your Sources

Generate an OPML file from DailyDigest:

```bash
cd /path/to/dailydigest
uv run python -m dailydigest.freshrss_export
```

This creates `data/sources.opml` with all RSS feeds from `config/sources.yaml`.

In FreshRSS:
1. Go to **Settings** (top-right)
2. Select **Import/Export**
3. Click **Upload** and choose `data/sources.opml`
4. FreshRSS imports all feeds and begins refreshing them

The first refresh may take 2–5 minutes depending on feed count. Check the **Health** page for feed status.

---

## Step 5 (Optional): Add Custom RSSHub Routes

RSSHub is useful for sites that don't offer RSS (paywalled journals, LinkedIn, etc.). It translates web pages into RSS feeds.

Example: Add a route for a journal without native RSS:

```
https://rsshub.yourdomain.com/journals/nature-nbt
```

When prompted for credentials, use:
- **Username:** value from `RSSHUB_BASIC_AUTH_USER` in `.env`
- **Password:** the plaintext password you used to generate the hash

Then add this URL as a new feed in FreshRSS.

(See [RSSHub docs](https://docs.rsshub.app/) for available routes.)

---

## Step 6 (Optional): Daily Digest Still Works

The email digest runs independently via GitHub Actions (or wherever you have it set up). The sidecar does NOT interfere with or replace the email pipeline — both run in parallel.

To keep them in sync, regenerate the OPML whenever you update `config/sources.yaml`:

```bash
uv run python -m dailydigest.freshrss_export
# Then re-import in FreshRSS (or FreshRSS auto-detects OPML URL if you point it there)
```

---

## Troubleshooting

### "Certificate error" / "Too many redirects"

**Cause:** Caddy hasn't issued a cert yet, or DNS isn't pointing to your server.

**Fix:**
1. Check DNS: `nslookup freshrss.yourdomain.com`
2. Check firewall: `curl -v http://freshrss.yourdomain.com:80` (should redirect to HTTPS)
3. Wait 30s for Caddy's ACME process, then refresh browser

Caddy logs:
```bash
docker compose logs caddy | grep -i "cert\|error\|tls"
```

### "Connection refused" when accessing FreshRSS

**Cause:** Container crashed or port conflict.

**Fix:**
```bash
docker compose ps              # check status
docker compose logs freshrss   # see errors
docker compose restart freshrss
```

### RSSHub returning 429 (Too Many Requests)

**Cause:** Rate limiting from the target site (legitimate).

**Fix:**
- Spread feed refreshes by increasing `CRON_MIN` (e.g., to 15 for every 15min).
- Reduce the number of RSSHub routes you subscribe to.
- Wait a few hours before retry.

### "Can't reach FreshRSS" from inside compose

This shouldn't happen, but if RSSHub or other services can't access FreshRSS:

```bash
docker network inspect infra_internal
docker compose exec rsshub ping freshrss
```

---

## Maintenance

### Backup Your Data

FreshRSS stores data in the `freshrss_data` volume. To back up:

```bash
docker run --rm -v infra_freshrss_data:/data \
  -v /tmp:/backup \
  alpine tar czf /backup/freshrss-backup.tar.gz -C /data .
```

### Restart All Services

```bash
docker compose down
docker compose up -d
```

### Update Images

```bash
docker compose pull
docker compose up -d
```

### Clean Up Old Data

If storage runs low, you can prune FreshRSS old entries in the UI (Settings → Data Management).

---

## Why OPML Import Instead of API Push?

FreshRSS's Greader API (the modern programmatic interface) **supports subscribing to feeds** but **does NOT allow inserting items directly** — items must come from real feeds. Our pragma: read `config/sources.yaml`, export the RSS URLs as OPML, import once, and let FreshRSS pull them natively. This sidesteps API complexity and guarantees day-one functionality.

---

## Cost & Scaling

| Component | Cost/month (Hetzner CX11) | Notes |
|---|---|---|
| VPS | ~$3–4 | Shared CPU; for personal use |
| Domain | $5–10 | Varies; .com often ~$10 |
| Emails (from main digest) | $0–3 | Separate, via Resend |
| **Total** | **~$8–17** | Fully self-hosted |

If you prefer no VPS: use **Fly.io free tier** (~1 shared CPU, 3 shared-3gb RAM apps). FreshRSS + RSSHub fit easily; Caddy provides HTTPS via Let's Encrypt at no extra cost.

---

## Important Caveat

This sidecar is **100% optional**. The email digest works without it. Use the sidecar if:
- You want to keep all ~900 ingested items searchable (not just top 20)
- You prefer a web UI for browsing
- You enjoy self-hosting

Skip it if email digest alone meets your needs.

---

## Questions?

Refer back to this guide or check:
- [FreshRSS docs](https://www.freshrss.org/)
- [RSSHub docs](https://docs.rsshub.app/)
- [Caddy docs](https://caddyserver.com/docs/)
