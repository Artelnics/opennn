# OpenNN Blog Style Reference

Use this reference for OpenNN website/blog articles intended for WordPress/Elementor.

## Brand Style

- Font: `Outfit`, fallback `Arial, sans-serif`.
- Body text: `18px`, `line-height: 24px`, `font-weight: 300`, black.
- Main OpenNN dark blue: `#001233`.
- OpenNN accent blue: `#6EC1E4`.
- Secondary gray: `#54595F`.
- PyTorch orange: `#EE4C2C`; secondary PyTorch/reference orange: `#F97316`.
- Prefer clean tables and simple inline styles over fragile SVGs.

## CSS Block

```html
<style>
html {
    scroll-behavior: smooth;
}

body {
    font-family: "Outfit", Arial, sans-serif;
    font-size: 18px;
    font-weight: 300;
    line-height: 24px;
    color: #000000;
}

h1, h2, h3 {
    font-family: "Outfit", Arial, sans-serif;
    color: #001233;
    margin-top: 0;
}

h1 {
    font-size: 42px;
    font-weight: 600;
    line-height: 48px;
    margin-bottom: 22px;
}

h2 {
    font-size: 30px;
    font-weight: 600;
    line-height: 36px;
    margin-top: 44px;
    margin-bottom: 18px;
}

h2[id] {
    scroll-margin-top: 90px;
}

h3 {
    font-size: 22px;
    font-weight: 500;
    line-height: 28px;
    margin-top: 28px;
    margin-bottom: 12px;
}

p {
    margin: 0 0 18px 0;
}

.lead {
    font-size: 22px;
    font-weight: 400;
    line-height: 30px;
    color: #001233;
    margin-bottom: 26px;
}

ul, ol {
    margin: 0 0 22px 24px;
    padding: 0;
}

li {
    margin-bottom: 8px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 24px 0 32px 0;
    font-size: 16px;
    line-height: 22px;
}

th {
    background: #001233;
    color: #ffffff;
    font-weight: 500;
    text-align: left;
    padding: 12px 14px;
    border: 1px solid #001233;
}

td {
    padding: 12px 14px;
    border: 1px solid #e5eaf0;
    vertical-align: top;
}

tr:nth-child(even) td {
    background: #f8fbfd;
}

.note {
    background: #f4fbff;
    border-left: 4px solid #6EC1E4;
    padding: 14px 18px;
    margin: 22px 0;
}

.warning {
    background: #f8fbfd;
    border-left: 4px solid #001233;
    padding: 14px 18px;
    margin: 22px 0;
}

code {
    font-family: Consolas, Monaco, monospace;
    font-size: 0.92em;
    background: #f1f5f9;
    padding: 2px 4px;
}

th code,
td code {
    font-family: inherit;
    font-size: inherit;
    background: transparent;
    padding: 0;
    color: inherit;
}

.small {
    font-size: 14px;
    line-height: 20px;
    color: #54595F;
}

.references-list {
    margin: 0 0 22px 24px;
    padding: 0;
}

.references-list li {
    margin-bottom: 10px;
    line-height: 24px;
}

.references-list a {
    color: #2f84b8;
    text-decoration: underline;
}

@media (max-width: 767px) {
    body {
        font-size: 16px;
        line-height: 23px;
    }

    h1 {
        font-size: 34px;
        line-height: 40px;
    }

    h2 {
        font-size: 26px;
        line-height: 32px;
    }

    table {
        font-size: 14px;
        line-height: 20px;
    }
}
</style>
```

## Contents With Anchors

```html
<h2>Contents</h2>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#benchmark-application">Benchmark application</a></li>
  <li><a href="#reference-computer">Reference computer</a></li>
  <li><a href="#methodology">Methodology</a></li>
  <li><a href="#results">Results</a></li>
  <li><a href="#discussion">Discussion</a></li>
  <li><a href="#conclusions">Conclusions</a></li>
  <li><a href="#references">References</a></li>
</ul>

<h2 id="introduction">Introduction</h2>
```

## WordPress-Safe Throughput Chart

Use this instead of inline SVG. Elementor often breaks SVG coordinate layout.

## WordPress Body Rules

- Do not include a duplicate `<h1>` inside the post body. The OpenNN theme already shows the WordPress title in the hero.
- Do not include internal benchmark maintenance text in the public body, including paragraphs that begin with `Benchmark note` or `Status:`.
- Do not style text as inline code inside table cells or headers; it becomes hard to read in dark table headers.
- Start the body with the strongest public-facing lead paragraph or the first useful section.

```html
<div style="max-width:860px;margin:32px 0 40px 0;font-family:'Outfit',Arial,sans-serif;color:#001233;">
  <div style="font-size:26px;font-weight:600;line-height:32px;margin-bottom:4px;">
    Training throughput on HIGGS
  </div>

  <div style="font-size:14px;font-weight:300;line-height:20px;color:#54595F;margin-bottom:26px;">
    Samples per second · RTX 4080 · batch size 100 · epochs 0-20
    <span style="float:right;color:#6EC1E4;font-weight:500;">HIGHER IS BETTER -&gt;</span>
  </div>

  <div style="display:grid;grid-template-columns:210px 1fr 70px;gap:12px;align-items:center;margin-bottom:18px;">
    <div style="font-size:16px;font-weight:500;">OpenNN</div>
    <div style="height:34px;background:#001233;border-radius:4px;width:100%;"></div>
    <div style="font-size:16px;font-weight:600;">604k</div>
  </div>

  <div style="display:grid;grid-template-columns:210px 1fr 70px;gap:12px;align-items:center;margin-bottom:18px;">
    <div style="font-size:16px;font-weight:500;">PyTorch · CUDA Graphs</div>
    <div style="height:34px;background:#EE4C2C;border-radius:4px;width:93%;"></div>
    <div style="font-size:16px;font-weight:600;">563k</div>
  </div>

  <div style="display:grid;grid-template-columns:210px 1fr 70px;gap:12px;align-items:center;margin-bottom:24px;">
    <div style="font-size:16px;font-weight:500;">PyTorch · reference</div>
    <div style="height:34px;background:#F97316;border-radius:4px;width:25%;"></div>
    <div style="font-size:16px;font-weight:600;">153k</div>
  </div>

  <div style="font-size:13px;font-weight:300;line-height:20px;color:#54595F;">
    OpenNN reaches CUDA Graph-class throughput natively, without graph-capture code.<br>
    Measured on a single representative run. Training loop only; CSV preprocessing excluded.
  </div>
</div>
```

## References

Keep each reference in one compact `li`. Elementor may break multiline nested inline markup.

```html
<h2 id="references">References</h2>

<ul class="references-list">
  <li><a href="https://archive.ics.uci.edu/dataset/280/higgs" target="_blank" rel="noopener">HIGGS dataset, UCI Machine Learning Repository</a>.</li>
  <li>P. Baldi, P. Sadowski and D. Whiteson, <a href="https://www.nature.com/articles/ncomms5308" target="_blank" rel="noopener">Searching for exotic particles in high-energy physics with deep learning</a>, Nature Communications, 2014.</li>
  <li><a href="https://pytorch.org/" target="_blank" rel="noopener">PyTorch</a>.</li>
</ul>
```

## Benchmark Framing Rules

- Lead with the strongest true OpenNN advantage.
- When comparing against a framework like PyTorch, state whether the competitor is a reference implementation or optimized.
- If an optimized competitor reaches similar speed, frame OpenNN as providing that speed natively without extra engineering.
- Avoid absolute claims like "OpenNN is faster than PyTorch" unless all tested PyTorch variants support that exact claim.
- Prefer "one representative run" unless repeated runs and variance are available.
- Include caveats in a way that improves credibility rather than weakening the article.
