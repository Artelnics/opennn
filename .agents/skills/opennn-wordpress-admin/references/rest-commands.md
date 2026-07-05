# Artelnics / OpenNN WordPress REST Commands

Use these commands from any shell with `curl` and `python3`. They intentionally rely on environment variables or per-site `.env` files so credentials do not land in skill files, repos, or generated artifacts.

On this machine there are three separate per-site `KEY=value` `.env` files (`Líneas de negocio\{OpenNN,Neural Designer,Consultoría}\Web\.env`), not one shared block file — see the "Known gap" note in `../SKILL.md` before relying on the block-file example below.

## Environment

```bash
export OPENNN_WP_SITE="https://www.opennn.net"
export OPENNN_WP_USER="..."
export OPENNN_WP_APP_PASSWORD="..."

export NEURALDESIGNER_WP_SITE="https://www.neuraldesigner.com"
export NEURALDESIGNER_WP_USER="..."
export NEURALDESIGNER_WP_APP_PASSWORD="..."

export ARTELNICS_WP_SITE="https://www.artelnics.com"
export ARTELNICS_WP_USER="..."
export ARTELNICS_WP_APP_PASSWORD="..."
```

Do not commit, copy, or write the password into skill files. WordPress application passwords may contain spaces; keep the value quoted when exporting it.

The shared desktop `.env` used by Artelnics machines may be a block file:

```text
NeuralDesigner
User ...
Pass ...

Artelnics
User ...
Pass ...

OpenNN
User ...
Pass ...
```

Do not print it. Use the helper:

```bash
python3 .agents/skills/opennn-wordpress-admin/scripts/wp_access.py --env "C:\Artelnics\Líneas de negocio\OpenNN\Web\.env" check
```

Expected route styles:

| Site | Base URL | REST style |
| --- | --- | --- |
| OpenNN | `https://www.opennn.net` | `/wp-json/wp/v2/...` |
| Neural Designer | `https://www.neuraldesigner.com` | `/?rest_route=/wp/v2/...` |
| Artelnics | `https://www.artelnics.com` | `/?rest_route=/wp/v2/...` |

## Basic Access Checks

```bash
SITE="${OPENNN_WP_SITE:-https://www.opennn.net}"

curl -fsS "$SITE/wp-json/" | python3 -m json.tool | head

curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  "$SITE/wp-json/wp/v2/users/me?context=edit" | python3 -m json.tool
```

For Neural Designer:

```bash
SITE="${NEURALDESIGNER_WP_SITE:-https://www.neuraldesigner.com}"
curl -fsS -u "$NEURALDESIGNER_WP_USER:$NEURALDESIGNER_WP_APP_PASSWORD" \
  "$SITE/?rest_route=/wp/v2/users/me&context=edit" | python3 -m json.tool
```

For Artelnics:

```bash
SITE="${ARTELNICS_WP_SITE:-https://www.artelnics.com}"
curl -fsS -u "$ARTELNICS_WP_USER:$ARTELNICS_WP_APP_PASSWORD" \
  "$SITE/?rest_route=/wp/v2/users/me&context=edit" | python3 -m json.tool
```

If JSON parsing fails and the response is an HTML challenge page, stop. Do not retry write requests through a WAF block.

## Discover Post Types And Categories

```bash
curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  "$SITE/wp-json/wp/v2/types?context=edit" | python3 -m json.tool

curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  "$SITE/wp-json/wp/v2/categories?search=Platforms&per_page=20&context=edit" \
  | python3 -m json.tool
```

The OpenNN blog custom post type has historically used the REST endpoint `blog`. Confirm it exists in `/types` before writing.

For Neural Designer or Artelnics, replace the URL with the `rest_route` style:

```bash
SITE="${ARTELNICS_WP_SITE:-https://www.artelnics.com}"
curl -fsS -u "$ARTELNICS_WP_USER:$ARTELNICS_WP_APP_PASSWORD" \
  "$SITE/?rest_route=/wp/v2/types&context=edit" | python3 -m json.tool
```

## List And Search Blogs

```bash
# Recent OpenNN blogs.
curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  "$SITE/wp-json/wp/v2/blog?per_page=20&context=edit&_fields=id,date,modified,status,slug,link,title,excerpt,categories" \
  | python3 -m json.tool

# Search by title text.
QUERY="Transformer inference"
curl -G -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  --data-urlencode "search=$QUERY" \
  --data-urlencode "context=edit" \
  --data-urlencode "per_page=20" \
  --data-urlencode "_fields=id,status,slug,link,title,excerpt,categories,modified" \
  "$SITE/wp-json/wp/v2/blog" | python3 -m json.tool

# Read one blog completely.
ID=1234
curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  "$SITE/wp-json/wp/v2/blog/$ID?context=edit" | python3 -m json.tool
```

For Neural Designer and Artelnics, first discover whether posts live under `posts`, `blog`, or another custom post type:

```bash
python3 .agents/skills/opennn-wordpress-admin/scripts/wp_access.py --env "C:\Artelnics\Líneas de negocio\Neural Designer\Web\.env" types --site neuraldesigner
python3 .agents/skills/opennn-wordpress-admin/scripts/wp_access.py --env "C:\Artelnics\Líneas de negocio\Consultoría\Web\.env" types --site artelnics
```

Then use `?rest_route=/wp/v2/<endpoint>` for reads and writes.

Editable fields usually appear as:

- `title.raw`: title shown in the WordPress editor.
- `excerpt.raw`: short description / excerpt.
- `content.raw`: raw HTML or block content.
- `status`: `draft`, `publish`, etc.
- `slug`: URL slug.
- `categories`: numeric category IDs.
- `link`: public URL.

## Backup Before Writing

```bash
ID=1234
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="C:/Artelnics/opennn/.backups/wordpress-blog-${ID}-${STAMP}.json"

curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  "$SITE/wp-json/wp/v2/blog/$ID?context=edit" \
  -o "$BACKUP"

python3 -m json.tool "$BACKUP" >/dev/null
echo "$BACKUP"
```

## Update A Blog Safely

Prefer a small JSON payload containing only fields that must change.

```bash
ID=1234
cat > /tmp/opennn-blog-update.json <<'JSON'
{
  "title": "New title",
  "excerpt": "Short description.",
  "content": "<p>Full HTML content here.</p>",
  "status": "publish"
}
JSON

curl -fsS -X POST -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/opennn-blog-update.json \
  "$SITE/wp-json/wp/v2/blog/$ID?context=edit" | python3 -m json.tool
```

After updating:

```bash
curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  "$SITE/wp-json/wp/v2/blog/$ID?context=edit&_fields=id,status,slug,link,title,excerpt,categories,modified" \
  | python3 -m json.tool
```

## Add Or Preserve The Platforms Category

```bash
PLATFORMS_ID="$(
  curl -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
    "$SITE/wp-json/wp/v2/categories?search=Platforms&per_page=20&context=edit" \
  | python3 -c 'import json,sys; print(next((c["id"] for c in json.load(sys.stdin) if c.get("name","").lower()=="platforms"), ""))'
)"
echo "$PLATFORMS_ID"
```

When updating a benchmark blog, include the current category IDs plus `PLATFORMS_ID`; do not replace unrelated categories accidentally.

`Platforms` is required for OpenNN benchmark blogs. Do not assume the same category exists or is required on Neural Designer or Artelnics unless the user says so.

## Create A New Blog

```bash
PLATFORMS_ID=123
cat > /tmp/opennn-new-blog.json <<JSON
{
  "title": "Benchmark title",
  "slug": "benchmark-title-slug",
  "excerpt": "Short card/blog description.",
  "content": "<p>Full HTML content here.</p>",
  "status": "draft",
  "categories": [$PLATFORMS_ID]
}
JSON

curl -fsS -X POST -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/opennn-new-blog.json \
  "$SITE/wp-json/wp/v2/blog?context=edit" | python3 -m json.tool
```

Create as `draft` unless the user explicitly asks to publish. Switch to `publish` only after reviewing the public preview/content.

## Read And Handle Elementor Pages

```bash
curl -G -fsS -u "$OPENNN_WP_USER:$OPENNN_WP_APP_PASSWORD" \
  --data-urlencode "slug=benchmarks" \
  --data-urlencode "context=edit" \
  "$SITE/wp-json/wp/v2/pages" | python3 -m json.tool
```

Important:

- OpenNN's visible `/benchmarks/` page is Elementor-backed.
- Neural Designer and Artelnics may also use Elementor-backed pages.
- `content.raw` may be only a fallback representation.
- Do not overwrite page content unless the actual Elementor data is accessible and backed up.
- If the safe Elementor path is unavailable, provide card HTML snippets for manual insertion instead of writing to the page.

## Public Verification

```bash
URL="https://www.opennn.net/blog/example/"
curl -fsS "$URL" | head
```

For visual/layout checks, use a browser or screenshot tool when available. REST verification proves stored fields, not necessarily rendered Elementor layout.
