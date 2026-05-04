# Custom Sender Domain Setup (Resend + DKIM/SPF/DMARC)

By default DailyDigest sends from `onboarding@resend.dev`, Resend's shared
sandbox address. This lands in spam roughly 30% of the time because the
sending domain has no relationship to your inbox.

Verifying a custom domain you own takes about 20 minutes and permanently
fixes deliverability.

---

## Why bother?

| Without custom domain | With custom domain |
|---|---|
| Shared `onboarding@resend.dev` | Your own `digest@yourdomain.com` |
| No DKIM alignment — spam filters suspicious | DKIM + SPF aligned — passes most spam filters |
| Gmail often routes to Promotions / Spam | Typically reaches Primary or Inbox |
| No DMARC policy possible | Full DMARC enforcement available |

---

## Prerequisites

- A domain you own (any registrar: Namecheap, Cloudflare, GoDaddy, Route 53, etc.)
- Access to the DNS control panel for that domain
- A Resend account (free tier: 3,000 emails/month, 100/day)

---

## Step 1 — Add the domain in Resend

1. Log in at [resend.com](https://resend.com) and open **Domains**.
2. Click **Add Domain**, enter your domain (e.g., `yourdomain.com`).
3. Select the region closest to you (US East / EU West).
4. Resend shows 4 DNS records to add:
   - **1 SPF record** (TXT on the root or a subdomain)
   - **2 DKIM records** (TXT, long public-key values)
   - **1 DMARC record** (TXT, `v=DMARC1; p=none; ...`)

Leave this page open — you will need the values in Step 2.

---

## Step 2 — Add the DNS records

In your DNS provider's control panel add each record Resend displayed:

### SPF

```
Type:    TXT
Name:    @   (or the subdomain Resend specified, e.g. "send")
Value:   v=spf1 include:amazonses.com ~all
TTL:     3600
```

> Resend routes through AWS SES under the hood, so the SPF include is
> `amazonses.com`. Use the exact value Resend shows — do not copy the
> example above if it differs.

### DKIM (two records)

```
Type:    TXT
Name:    resend._domainkey   (or whatever Resend shows)
Value:   p=<very long base64 string>
TTL:     3600
```

Add both DKIM records exactly as shown. DKIM keys are long — copy/paste
from the Resend dashboard to avoid typos.

### DMARC (Resend default)

```
Type:    TXT
Name:    _dmarc
Value:   v=DMARC1; p=none; rua=mailto:postmaster@yourdomain.com
TTL:     3600
```

`p=none` is monitor-only and will not block any mail. You can tighten it
later (see Step 4).

Propagation is usually 5–15 minutes; allow up to 48 hours for slow registrars.

---

## Step 3 — Verify in Resend and update `.env`

1. Return to **Domains** in the Resend dashboard and click **Verify**.
2. The status next to each record turns green when detected.
3. Once all records are verified, update your `.env` (or GitHub secret):

```
DIGEST_FROM=digest@yourdomain.com
```

On the next `dd run-all` the email will originate from your domain.

---

## Step 4 — Tighten DMARC (recommended, after a week of monitoring)

Start with `p=none` (Step 2) and let digest emails flow for 7–14 days.
Check the DMARC aggregate reports landing at `postmaster@yourdomain.com`.
When all sends show `pass` for both SPF and DKIM, upgrade the policy:

```
v=DMARC1; p=quarantine; rua=mailto:postmaster@yourdomain.com
```

`quarantine` moves failing mail to spam instead of delivering it. This
prevents spoofing of your domain name.

Prefer `quarantine` over `reject` unless you are certain all legitimate
mail from your domain is covered, because `reject` causes some forwarded
mail (e.g., via alumni or mailing-list servers) to be silently dropped.

---

## Troubleshooting

### Records not verifying after 30 minutes

- Confirm there are no extra spaces or quotes added by your registrar's UI.
- Some registrars auto-append the domain to the record name (e.g., entering
  `resend._domainkey` becomes `resend._domainkey.yourdomain.com`). Check with
  `dig TXT resend._domainkey.yourdomain.com`.
- Cloudflare: make sure the records are **not proxied** (DNS-only, grey cloud).

### DKIM body-length tag warning

Some tools flag DKIM records with a `l=` body-length tag. Resend does not
add this tag, so you should not see it. If you imported a record from
another provider that includes `l=`, remove the `l=` parameter — it can
cause verification failures when email clients rewrite the message body.

### Emails still landing in spam after verification

- Check that SPF and DKIM both show `pass` in Gmail's "Show original" header view.
- Verify DMARC alignment: the `From:` domain must match the SPF/DKIM domain.
  If `DIGEST_FROM=digest@yourdomain.com` but SPF is on a different subdomain,
  alignment fails.
- Warm up the sending domain by sending a few emails to an address you
  control and marking them "Not Spam" before sending to your real inbox.

---

## Cost

| Component | Cost |
|---|---|
| Domain registration | $10–15/year (varies by TLD) |
| DNS hosting | Free with registrar, or free on Cloudflare |
| Resend (up to 3,000 emails/month) | Free |
| **Total for DailyDigest use case** | **$0 beyond domain registration** |
