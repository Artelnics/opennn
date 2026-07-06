---
name: opennn-blog-writer
description: Create, revise, style, publish, and export OpenNN website/blog articles, especially benchmark comparison posts, using OpenNN's Elementor/Outfit visual style and WordPress REST API when credentials are available. Use when Codex needs to write or edit an OpenNN web article, generate WordPress/Elementor-safe HTML, add benchmark tables/throughput visuals/references, publish/update a blog post directly with the Platforms category for benchmark posts, or export an article to DOCX/HTML in Documents.
---

# OpenNN Blog Writer

## Workflow

1. Gather benchmark facts first: dataset, model, hardware, software versions, commands, timing scope, number of runs, metrics, and caveats.
2. Frame OpenNN favorably but defensibly. Prefer "OpenNN reaches this performance natively" over claims that can be dismissed as comparing against unoptimized competitors.
3. Use the current OpenNN web style from `references/opennn_blog_style.md`.
4. For WordPress/Elementor output, avoid fragile inline SVGs. Use HTML tables or `div` bar charts with inline styles.
5. Add anchors to section headings and use linked contents when the user wants navigation.
6. Add external links in references with `target="_blank" rel="noopener"`.
7. If publishing/editing WordPress directly, use the WordPress workflow below.
8. If the user asks for a Word document, create HTML first and convert with `scripts/html_to_docx.py`.

## WordPress Direct Editing

Use this when the user asks to edit/publish an OpenNN post directly and provides a WordPress post URL, post ID, or slug plus valid credentials/application password.

Safety rules:

- Never store WordPress credentials in this skill or in generated files.
- Prefer an Application Password. If credentials are provided in chat, use them only for the current operation.
- Before changing a post, read it with `context=edit` and save a backup to `.backups/opennn_blog_<id>_backup_<timestamp>.html` (repo root).
- OpenNN blog articles use the custom post type `blog`. If `/wp-json/wp/v2/posts/<id>` returns 404, try `/wp-json/wp/v2/blog/<id>`.
- OpenNN benchmark articles published as `blog` posts must use the WordPress category `Platforms`. Preserve any existing categories and add `Platforms` if it is missing.
- For admin URLs like `wp-admin/post.php?post=841&action=edit`, extract `post=841`.
- Publish/update only when the user explicitly asks to edit, upload, publish, or leave the post updated. For "look/check/connect" requests, read only.
- After every update, read the post again and verify title, slug, content length, and several key phrases/numbers.
- Save a local copy of the final HTML in `.backups/`.

WordPress-safe HTML rules:

- Upload an HTML fragment, not a full document. Do not include `<!doctype html>`, `<html>`, `<head>`, or `<body>` unless the user specifically needs a full standalone file.
- Do not include the article title as an `<h1>` in the post body. The OpenNN blog theme already prints the WordPress title in the hero.
- Do not publish internal/editorial opening notes such as `Benchmark note for opennn.net/benchmarks`, `Last updated ...`, or `Status: ...`.
- Do not use inline code styling inside table cells or headers. Names such as `torch.compile`, `tf.function`, model filenames, and library names should be readable as normal table text, especially on dark header cells.
- A top-level `<style>` block is acceptable and has worked on the OpenNN blog editor.
- Elementor can break inline SVG coordinate layouts. Prefer tables and CSS `div` bar charts.
- Keep references compact: one `<li>` per reference.

Recommended script:

```bash
python3 .agents/skills/opennn-blog-writer/scripts/wp_blog_post.py get \
  --site https://www.opennn.net \
  --user "$OPENNN_WP_USER" \
  --password "$OPENNN_WP_APP_PASSWORD" \
  --id 841

python3 .agents/skills/opennn-blog-writer/scripts/wp_blog_post.py update \
  --site https://www.opennn.net \
  --user "$OPENNN_WP_USER" \
  --password "$OPENNN_WP_APP_PASSWORD" \
  --id 841 \
  --title "CPU Training Speed on HIGGS: OpenNN vs PyTorch" \
  --slug higgs-cpu-opennn-pytorch \
  --html .backups/article.html
```

If env vars are not available, pass the user/application password directly as arguments for that one command only.

## Article Structure

Use this default structure for benchmark posts:

- Lead paragraph with the strongest objective OpenNN message
- Short framing/caveat note
- Contents
- Introduction
- Benchmark application
- Reference computer
- Methodology
- Results
- Discussion
- Conclusions
- References

For benchmark comparisons, include:

- Same metric rows for all frameworks.
- Both predictive quality and speed.
- Clear timing scope: training loop only vs preprocessing included.
- Whether the result is one representative run or an average.
- Whether epoch 0 is included.
- Whether competitor optimizations such as CUDA Graphs or `torch.compile` were tested.
- A caveat when differences are small enough that repeated-run variance could matter.

## Exporting

Use the bundled converter:

```bash
python3 .agents/skills/opennn-blog-writer/scripts/html_to_docx.py \
  .backups/article.html \
  .backups/article.docx
```

The script uses LibreOffice headless through an ODT intermediate, which was more reliable than direct HTML to DOCX conversion.

## Style Reference

Read `references/opennn_blog_style.md` when generating HTML, CSS, benchmark bars, references, or WordPress-ready snippets.
