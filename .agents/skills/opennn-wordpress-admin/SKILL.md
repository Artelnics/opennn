---
name: opennn-wordpress-admin
description: Connect to, inspect, create, and safely edit the OpenNN, Neural Designer, and Artelnics WordPress sites via the REST API. Use when Codex needs to verify WordPress access for opennn.net, neuraldesigner.com, or artelnics.com; load the shared desktop .env credentials; read or update blog posts, pages, titles, excerpts/descriptions, categories, raw HTML content, benchmark entries, or Elementor-backed pages without storing credentials.
---

# OpenNN / Neural Designer / Artelnics WordPress Admin

## Core Rules

- Never store WordPress usernames, passwords, application passwords, cookies, or tokens in repo files, skill files, logs meant for sharing, or generated artifacts.
- Use environment variables for credentials:
  - `OPENNN_WP_SITE`, `OPENNN_WP_USER`, `OPENNN_WP_APP_PASSWORD`
  - `NEURALDESIGNER_WP_SITE`, `NEURALDESIGNER_WP_USER`, `NEURALDESIGNER_WP_APP_PASSWORD`
  - `ARTELNICS_WP_SITE`, `ARTELNICS_WP_USER`, `ARTELNICS_WP_APP_PASSWORD`
- On this machine, credentials live in three separate per-site `KEY=value` `.env` files, not one shared block file:
  - OpenNN: `C:\Artelnics\Líneas de negocio\OpenNN\Web\.env`
  - Neural Designer: `C:\Artelnics\Líneas de negocio\Neural Designer\Web\.env`
  - Artelnics: `C:\Artelnics\Líneas de negocio\Consultoría\Web\.env`
  **Known gap:** `scripts/wp_access.py`'s `--env` flag was written for one shared multi-site block file (see `read_block_env`) — it has not been verified against these three separate `KEY=value` files. Confirm the parser handles this format (or load the vars directly) before trusting it.
- Read with `context=edit` when credentials are available; it exposes `title.raw`, `excerpt.raw`, `content.raw`, status, slug, categories, and editable metadata.
- Before any write, save the current editable JSON response under `.backups/` (repo root) with a timestamp.
- Do not modify the live site unless the user explicitly asks for a write. For "try", "check access", or "review", only perform authenticated GET requests.
- If the response is HTML such as "One moment, please..." instead of JSON, stop WordPress writes and report the WAF/anti-bot block.
- On OpenNN benchmark posts, keep the WordPress category `Platforms`.
- Do not update Elementor-backed pages by writing only `content.raw`; use Elementor data only if it is actually available and backed up. Otherwise provide snippets for manual insertion.

## Workflow

1. Locate credentials without printing passwords. Point at the relevant per-site `.env` (see paths above):

```bash
python3 .agents/skills/opennn-wordpress-admin/scripts/wp_access.py \
  --env "C:\Artelnics\Líneas de negocio\OpenNN\Web\.env" check
```

2. If credentials are already exported, confirm one site manually:

```bash
SITE="${OPENNN_WP_SITE:-https://www.opennn.net}"
curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  "$SITE/wp-json/wp/v2/users/me?context=edit"
```

3. Use the correct REST route style:
   - OpenNN: `/wp-json/wp/v2/...`
   - Neural Designer: `?rest_route=/wp/v2/...`
   - Artelnics: `?rest_route=/wp/v2/...`

4. Discover available post types and categories before assuming endpoint names.

5. Read target content with `context=edit`, then inspect `title.raw`, `excerpt.raw`, `content.raw`, `status`, `slug`, `categories`, and `link`.

6. For writes, back up first, then send the smallest scoped update possible.

7. Verify after writing with another authenticated GET and a public GET when appropriate.

## Common Targets

- OpenNN site: `https://www.opennn.net`, REST style `wp-json`
- Neural Designer site: `https://www.neuraldesigner.com`, REST style `rest_route`
- Artelnics site: `https://www.artelnics.com`, REST style `rest_route`
- OpenNN blog custom post type usually uses endpoint: `blog`; confirm post types on each site before assuming custom post type names elsewhere.
- WordPress pages use the `pages` endpoint.
- Categories use the `categories` endpoint.
- Public benchmarks page: `https://www.opennn.net/benchmarks/`
- Benchmark blogs should belong to category `Platforms`.

## Read The Command Reference

For exact cURL patterns, JSON payload examples, backup commands, category handling, blog creation, and Elementor cautions, read `references/rest-commands.md` before doing live WordPress work.

## Helper Script

Use `scripts/wp_access.py` for safe auth checks and endpoint discovery. It reads either exported env vars or an `--env` file and prints only non-secret status (see the known gap above for the multi-site block-file assumption).

```bash
python3 .agents/skills/opennn-wordpress-admin/scripts/wp_access.py --env "C:\Artelnics\Líneas de negocio\OpenNN\Web\.env" check
python3 .agents/skills/opennn-wordpress-admin/scripts/wp_access.py --env "C:\Artelnics\Líneas de negocio\OpenNN\Web\.env" types --site all
```

## Safe Editing Checklist

- Verify the user requested a live edit.
- Fetch the editable object with `context=edit`.
- Save a timestamped backup JSON in `.backups/` (repo root).
- Locate the category ID for `Platforms` if editing benchmark blogs.
- Prepare a minimal JSON payload.
- Use `POST` for updates and creation through the correct REST endpoint.
- Re-fetch the object and compare the fields changed.
- Never paste credentials into final answers.
