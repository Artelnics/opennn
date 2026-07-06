---
name: opennn-benchmark-sync
description: Audit, render, create, and update OpenNN benchmark WordPress blogs from docs/benchmarks Markdown. Use when Codex needs to compare benchmark .md files against opennn.net blog posts, identify missing or stale benchmark blogs, publish/update OpenNN WordPress custom post type blog entries from Markdown with the Platforms category, generate benchmark-card snippets for https://www.opennn.net/benchmarks/, or reason about safe Elementor-backed benchmark-page updates.
---

# OpenNN Benchmark Sync

## Source Of Truth

- Treat `docs/benchmarks/*.md` as the source of truth.
- Treat every root benchmark Markdown file as one WordPress `blog` post, except internal files such as `README.md` and `INVESTOR_AUDIT_AND_PLAN.md`.
- Keep website blog posts synchronized to the Markdown structure and data.
- Assign every benchmark blog post to the WordPress category `Platforms`.
- Keep the Markdown `#` heading as the canonical WordPress post title, but do not render it inside the post body. The theme/Platforms hero already prints the visible title.
- Do not render or publish editorial opening paragraphs that start with `Benchmark note` or `Status:`.
- Do not render inline code styling inside table cells or headers. Framework/tool names such as `torch.compile`, `tf.function`, or library filenames should appear as normal table text so they remain readable on dark table headers.
- Every published benchmark blog should have a corresponding entry on `https://www.opennn.net/benchmarks/` with title, compact chart/metrics, and a "Learn more" link.

## Safety Rules

- Never store WordPress credentials in skill files, generated files, or repo files.
- Use `OPENNN_WP_USER` and `OPENNN_WP_APP_PASSWORD`, or pass credentials only for the current command.
- Before updating an existing WordPress blog, read it with `context=edit` and save a backup under `.backups/` (repo root).
- Do not update the Elementor-backed Benchmarks page by writing only `content.raw`; that is usually a search/fallback representation, not the visible Elementor layout.
- Only update `/benchmarks/` when the real Elementor data is accessible and backed up, or when the user explicitly accepts a manual/alternative page update.
- If WordPress returns an HTML "One moment, please..." page instead of JSON, stop remote writes and report the temporary anti-bot/WAF block.

## Quick Commands

From the OpenNN repository root:

```bash
python3 .agents/skills/opennn-benchmark-sync/scripts/benchmark_sync.py audit \
  --repo . \
  --site https://www.opennn.net \
  --user "$OPENNN_WP_USER" \
  --password "$OPENNN_WP_APP_PASSWORD"
```

Render one Markdown file to WordPress-safe HTML:

```bash
python3 .agents/skills/opennn-benchmark-sync/scripts/benchmark_sync.py render \
  --repo . \
  --doc docs/benchmarks/transformer-inference-gpu-opennn-vs-pytorch.md \
  --output .backups/transformer-inference.html
```

Create or update one WordPress `blog` post from its Markdown:

```bash
python3 .agents/skills/opennn-benchmark-sync/scripts/benchmark_sync.py publish-blog \
  --repo . \
  --doc docs/benchmarks/transformer-inference-gpu-opennn-vs-pytorch.md \
  --site https://www.opennn.net \
  --user "$OPENNN_WP_USER" \
  --password "$OPENNN_WP_APP_PASSWORD" \
  --status publish
```

Create or update every benchmark blog from `docs/benchmarks`:

```bash
python3 .agents/skills/opennn-benchmark-sync/scripts/benchmark_sync.py publish-all \
  --repo . \
  --site https://www.opennn.net \
  --user "$OPENNN_WP_USER" \
  --password "$OPENNN_WP_APP_PASSWORD" \
  --status publish
```

Ensure every existing benchmark blog has the `Platforms` category:

```bash
python3 .agents/skills/opennn-benchmark-sync/scripts/benchmark_sync.py ensure-category \
  --repo . \
  --site https://www.opennn.net \
  --user "$OPENNN_WP_USER" \
  --password "$OPENNN_WP_APP_PASSWORD"
```

Generate card snippets for missing Benchmarks-page entries:

```bash
python3 .agents/skills/opennn-benchmark-sync/scripts/benchmark_sync.py cards \
  --repo . \
  --output .backups/opennn_benchmark_cards.html
```

## Workflow

1. Run `audit` first. It reports:
   - `.md` files without a WordPress blog.
   - Existing blogs whose rendered Markdown HTML differs from the editable WordPress raw content.
   - Existing benchmark blogs that are missing the `Platforms` category.
   - WordPress blogs that do not have a matching root `.md`.
   - Whether `/benchmarks/` can be safely edited automatically.
2. Run `ensure-category` when existing benchmark blogs are missing `Platforms`.
3. For missing or stale blogs, run `publish-blog` per file, or use its output as a checklist.
4. Verify each changed blog with `context=edit` and public link checks.
5. For `/benchmarks/`, use `cards` to produce title/chart/link snippets, then update Elementor only through a safe Elementor-aware path.

## Script Notes

The bundled script uses conservative Markdown support tailored to the benchmark docs: headings, paragraphs, emphasis, inline code, links, lists with wrapped continuations, fenced code, blockquotes, and pipe tables. It emits an HTML fragment with the OpenNN blog style block, not a full HTML document.

Before rendering, it removes the first Markdown `#` heading from the body output and drops opening editorial paragraphs beginning with `Benchmark note` or `Status:`. It also renders table cells without `<code>` tags, even when the Markdown table cell uses backticks. Keep those out of publishable Markdown tables when possible.

Known historical slugs are built in because several existing WordPress posts do not use the exact Markdown filename as their slug. New benchmark posts default to the Markdown filename stem.
