"""Render application comparisons from the raw artifact, without fixed numbers."""


def write_table(out, kind, rows):
    if kind == "startup":
        lines = [
            "# Faster startup",
            "",
            "Local diagnostic measurements; validate on the reference computer.",
            "",
            "| Backend | Model | Precision | Cache | OpenNN (ms) | PyTorch Python (ms) | Ratio |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['backend']} | {r['family']} | {r['precision']} | {r['cache']} | "
                f"{r['opennn_ms']:.2f} | {r['pytorch_python_ms']:.2f} | {r['opennn_percent_of_pytorch']:.2f}% |"
            )
        lines += [
            "",
            "OpenNN startup time ÷ PyTorch startup time × 100. Lower is better.",
            "Time runs from process creation to the first completed prediction in host memory.",
            "Fresh processes; warmed filesystem cache. Reused/empty refers to application tuning caches.",
            f"{sum(r['variation_over_3_percent'] for r in rows)} of {len(rows)} comparisons exceed the 3% timing variation limit.",
            f"{sum(not r['variation_assessed'] for r in rows)} comparisons have insufficient repetitions to assess variation.",
        ]
    else:
        lines = [
            "# Smaller deployment",
            "",
            "Local diagnostic measurements; decimal MB.",
            "",
            "| Backend | Model | Precision | OpenNN (MB) | PyTorch Python (MB) | Ratio |",
            "|---|---|---|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['backend']} | {r['family']} | {r['precision']} | {r['opennn_MB']:.2f} | "
                f"{r['pytorch_python_MB']:.2f} | {r['opennn_percent_of_pytorch']:.2f}% |"
            )
        lines += [
            "",
            "OpenNN deployment size ÷ PyTorch deployment size × 100. Lower is better.",
            "OpenNN: application plus exercised native dependencies. PyTorch: application plus its full standard Python installation.",
            "The raw artifact also records observed runtime files for both engines. Neither is a proven minimal portable bundle.",
            "Excludes OS/driver files, model weights and generated caches. Zero Python packages does not mean zero native dependencies.",
        ]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
