# DailyDigest Companion: what it is and how to set it up

Setup instructions updated 2026-10-28. The dated verification notes below describe the scope of earlier testing.

## In one paragraph

The DailyDigest website runs entirely inside your browser. A browser page cannot use
your Claude or Codex login, send email, or run while the tab is closed. The
**Companion** is a small program you run on your own computer that does those three
things for the page. It is optional. Reading, brewing, voting and saving work without
it, and it never changes which papers are picked.

## Do you need it?

| You want | Needs the Companion? |
|---|---|
| Open the site, brew, read, vote, save | No |
| One-line "why read this" summaries written by your Claude or Codex subscription | Yes |
| The digest emailed to you, and email replies counted as feedback | Yes |
| A digest brewed every morning with the tab closed | Yes |
| Moving your reading desk to the Companion computer and back | Yes (optional) |

If you only want Claude or Codex summaries, email, or scheduling, there is a simpler
route: install the **local DailyDigest app** (`bash scripts/install.sh`, then open
http://127.0.0.1:8765). It has those features built in and needs no Companion. Use
the Companion only if you want to keep using the website.

## How it works

```
Browser tab (website)  --->  Companion on your computer  --->  claude / codex / email
   https://antimonyleo.github.io      http://127.0.0.1:8768          your own login
```

1. You start the Companion on your computer. It listens on `127.0.0.1` only, so
   nothing outside your computer can reach it.
2. You paste its address and a private token into the website once. The browser
   remembers them.
3. During a brew, if you selected a Claude or Codex summary provider, the website
   sends the picked papers (titles and abstracts) and a short interest profile to the
   Companion. The Companion runs your installed command in a temporary folder with a
   time limit, then returns the summaries. Claude runs with tools disabled. Codex uses
   a read-only sandbox and disables selected features; that does not disable every tool.
4. If the command fails, items fall back to the built-in summary, and the page records
   which summary each item got.

Only the website origin (scheme, host, and port) you allow can talk to it, and every request must carry
the token. Provider logins stay on the Companion computer.

## Setup

1. **Install DailyDigest on the computer that has your Claude or Codex login.**
   Download the repository ZIP (or clone it), then install `uv`:
   https://docs.astral.sh/uv/getting-started/installation/ . In the project folder run
   `uv sync`.
2. **Check the login you want to use.**
   Run `claude` or `codex` once and sign in. The Companion only offers a provider if
   the command is on your PATH.
3. **Start the Companion** in the project folder and leave the window open:

   ```
   uv run dd companion --allowed-origin https://antimonyleo.github.io
   ```

   It prints the address (`http://127.0.0.1:8768`) and the location of the token file.
   The token file is reused on later starts. Default location: `~/.config/dailydigest/companion-token`
   on Linux and macOS, `%LOCALAPPDATA%\dailydigest\companion-token` on Windows.
4. **Connect the website.** Open the site, then **Data & compute → Companion**.
   Paste `http://127.0.0.1:8768` and the token (open the token file and copy its
   single line), then press **Connect**. The status line should read
   "Connected… Available: summaries, …".
5. **Choose the summary provider.** Open **Settings**, pick *Claude* or *Codex* as the
   summary provider, save, and brew.

## Automatic morning brewing

In the website’s Companion panel, select **Prepare a command for automatic morning brewing**. It adds `--state-dir ./data/companion-state --schedule` to the command for this website. Run it from the DailyDigest directory and leave it running.

1. Connect with the displayed address and private token.
2. Select **Send reading desk to Companion**. Scheduling waits for this explicit upload.
3. Keep that computer on, awake, and online at the copied desk’s configured digest hour (8 am by default), in its saved timezone. `DIGEST_HOUR` is an integer from 0 to 23. This is not an operating-system wake-up service.
4. Use **Get Companion’s latest copy** to retrieve the result. This replaces the browser desk, with a recovery copy retained; newer browser ratings are not merged. Export a backup before changing copies.

The directory is a separate copy of the reading desk, not automatic synchronization. Send it again after changing topics or settings. Selecting the checkbox only prepares a command: it does not install, start, or schedule anything by itself.

Email is separate: configure `RESEND_API_KEY`, `DIGEST_FROM`, and `DIGEST_TO` in the app’s `.env` (see [email setup](domain-setup.md)). Importing email reply ratings also requires mailbox configuration. Optional providers must be explicitly configured on the Companion computer; scheduling does not choose one for you.

## Stop and disconnect

Press Ctrl+C in the terminal to stop the helper and scheduling. **Disconnect** in the website forgets the connection but does not stop the process. To revoke a token, stop the helper, remove the token file whose path it printed, restart, and reconnect with the newly generated token. Preserve backups before deleting any reading-desk copies.

## Troubleshooting

| What you see | Likely cause and fix |
|---|---|
| "Failed to fetch" when connecting | Recent Chrome blocks a public website from reaching a program on your own computer unless you allow it. Allow "local network access" for the site if Chrome asks, or in the site settings (lock icon, Site settings). Also confirm the Companion window is still running. |
| "Invalid companion token" | The token pasted does not match the file. Copy the single line from the token file again. |
| Connects, but no Claude or Codex option | The command is not on the PATH of the window that started the Companion, or you are not signed in. Run `claude --version` or `codex --version` in that window. |
| Summaries look like the basic ones | The provider failed and the page fell back. Check the Companion window for an error and your plan's usage limit. |
| Port already in use | Start with `--port 8769` and use that address in the page. |

## Privacy

When you use a Claude or Codex summary, the titles and abstracts of the picked papers
and a short profile are sent to that provider through your own login. Your token never
leaves your computer except to the website you connect it to. Connect only to a
Companion you started yourself.

## What was verified and what was not

Verified (2026-10-16, Chrome 152 headless, local Companion):
- The Companion starts, rejects requests without the token (401) and from the wrong
  website (403), and reports `claude_cli` and `codex_cli` when both are installed.
- With local-network permission granted, the live site connected and showed
  "Available: summaries, state transfer".
- Without that permission, the connection failed with only "Failed to fetch".

Unverified: whether a headed Chrome shows its permission prompt for this request
(it comes from a background worker), Firefox and Safari, Windows, a real summary
through Claude or Codex, email, inbound mail and the scheduler.
