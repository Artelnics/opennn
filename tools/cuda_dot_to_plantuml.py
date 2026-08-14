#!/usr/bin/env python3
"""Convert CUDA's cudaGraphDebugDotPrint output to exact-topology PlantUML."""

from __future__ import annotations

import argparse
import html
import re
from collections import Counter
from pathlib import Path


NODE_RE = re.compile(
    r'^"(?P<name>graph_\d+_node_(?P<index>\d+))"\[(?P<attrs>.*?)'
    r'label="(?P<label>.*?)"\];$',
    re.MULTILINE | re.DOTALL,
)
EDGE_RE = re.compile(
    r'^"(?P<src>graph_\d+_node_(?P<src_index>\d+))"\s*->\s*'
    r'"(?P<dst>graph_\d+_node_(?P<dst_index>\d+))"'
    r'\s*\[headlabel=(?P<headlabel>\d+)\];$',
    re.MULTILINE,
)

NODE_TYPES = (
    "KERNEL",
    "MEMCPY",
    "MEMSET",
    "MEM_ALLOC",
    "MEM_FREE",
    "EVENT_RECORD",
    "EVENT_WAIT",
    "HOST",
    "EMPTY",
    "CHILD_GRAPH",
)

COLORS = {
    "KERNEL": "#EAF3F9",
    "MEMCPY": "#FFF1D6",
    "MEMSET": "#E8F6EC",
    "MEM_ALLOC": "#F2EAF9",
    "MEM_FREE": "#F8E8EE",
    "EVENT_RECORD": "#DDEEFF",
    "EVENT_WAIT": "#DDEEFF",
    "HOST": "#F4F4F4",
    "EMPTY": "#FFFFFF",
    "CHILD_GRAPH": "#E9E9FF",
    "UNKNOWN": "#FFFFFF",
}


def node_type(label: str) -> str:
    for candidate in NODE_TYPES:
        if re.search(rf"(?:^|[{{\n]){re.escape(candidate)}(?:$|[|\n])", label):
            return candidate
    return "UNKNOWN"


def topology_id(label: str) -> str:
    match = re.search(r"topoId:\s*(\d+)", label)
    return match.group(1) if match else "?"


def kernel_signature(label: str) -> str:
    match = re.search(r"\(topoId:\s*\d+\)\s*\|\s*(.*?)\}\s*\n", label)
    return match.group(1).strip() if match else "kernel"


def compact_details(kind: str, label: str) -> list[str]:
    if kind == "KERNEL":
        return [kernel_signature(label)]
    if kind == "MEMCPY":
        transfer = re.search(r"\{kind\s*\|\s*([^}]+)\}", label)
        extent = re.search(
            r"\{Extent\s*\|\s*\{\{Width\s*\|\s*(\d+)\}\s*\|\s*"
            r"\{Height\s*\|\s*(\d+)\}\s*\|\s*\{Depth\s*\|\s*(\d+)\}",
            label,
        )
        lines = [transfer.group(1).strip()] if transfer else []
        if extent:
            lines.append(f"extent={extent.group(1)}x{extent.group(2)}x{extent.group(3)}")
        return lines
    if kind == "MEMSET":
        values = re.search(
            r"\|\s*\{\d+\s*\(topoId:.*?\|\s*(0x[0-9A-Fa-f]+)\s*\|\s*"
            r"(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\}",
            label,
        )
        if values:
            return [
                f"dptr={values.group(1)} value={values.group(2)}",
                f"element={values.group(3)} width={values.group(4)} height={values.group(5)}",
            ]
    if kind == "MEM_ALLOC":
        values = re.search(r"\{bytesize\s*\|\s*dptr\}\s*\|\s*\{(\d+)\s*\|\s*(0x[0-9A-Fa-f]+)\}", label)
        if values:
            return [f"bytes={values.group(1)}", f"dptr={values.group(2)}"]
    if kind == "MEM_FREE":
        value = re.search(r"\{dptr\}\s*\|\s*\{(0x[0-9A-Fa-f]+)\}", label)
        if value:
            return [f"dptr={value.group(1)}"]

    flattened = re.sub(r"[{}|]", " ", label)
    flattened = re.sub(r"\s+", " ", flattened).strip()
    return [flattened]


def full_details(label: str) -> list[str]:
    lines = []
    for part in re.split(r"[|\n]+", label):
        cleaned = re.sub(r"[{}]+", " ", part)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def puml_text(value: str) -> str:
    value = (
        value.replace(r"\<", "<")
        .replace(r"\>", ">")
        .replace(r"\{", "{")
        .replace(r"\}", "}")
    )
    escaped = html.escape(value, quote=False)
    escaped = escaped.replace("\\", "\\\\").replace('"', '\\"')
    return escaped


def find_groups(nodes: list[dict[str, object]]) -> list[tuple[str, int, int]]:
    embedding_nodes = [
        int(node["index"])
        for node in nodes
        if node["kind"] == "KERNEL" and "embedding_forward_kernel" in str(node["label"])
    ]
    entries: list[int] = []
    previous = -2
    for index in embedding_nodes:
        if index != previous + 1:
            entries.append(index)
        previous = index
    if len(entries) < 2:
        return [("CUDA graph", int(nodes[0]["index"]), int(nodes[-1]["index"]))]

    groups: list[tuple[str, int, int]] = []
    if entries[0] > int(nodes[0]["index"]):
        groups.append(("Captured H2D staging", int(nodes[0]["index"]), entries[0] - 1))
    for position, start in enumerate(entries):
        end = entries[position + 1] - 1 if position + 1 < len(entries) else int(nodes[-1]["index"])
        groups.append((f"Captured training step {position + 1}", start, end))
    return groups


def parse_dot(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = path.read_text(encoding="utf-8")
    nodes = []
    for match in NODE_RE.finditer(source):
        label = match.group("label")
        nodes.append(
            {
                "name": match.group("name"),
                "index": int(match.group("index")),
                "label": label,
                "kind": node_type(label),
                "topology_id": topology_id(label),
            }
        )
    edges = [
        {
            "src": int(match.group("src_index")),
            "dst": int(match.group("dst_index")),
            "headlabel": match.group("headlabel"),
        }
        for match in EDGE_RE.finditer(source)
    ]
    if not nodes:
        raise ValueError(f"No CUDA nodes found in {path}")
    return nodes, edges


def write_puml(
    output: Path,
    source_name: str,
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    label_mode: str,
    title: str | None = None,
) -> None:
    groups = find_groups(nodes)
    lines = [
        "@startuml",
        "!pragma layout smetana",
        f"title {puml_text(title or f'OpenNN CUDA training graph — exact topology from {source_name}')}",
        "top to bottom direction",
        "skinparam backgroundColor #FFFFFF",
        "skinparam shadowing false",
        "skinparam roundcorner 6",
        "skinparam linetype ortho",
        "skinparam nodesep 8",
        "skinparam ranksep 12",
        "skinparam packageStyle rectangle",
        "skinparam packageBorderColor #1E5374",
        "skinparam packageBackgroundColor #F8FBFD",
        "skinparam ArrowColor #5F6368",
        "skinparam defaultFontName Monospaced",
        "skinparam defaultFontSize 9",
        "hide stereotype",
        "legend right",
        f"  Nodes: {len(nodes)}",
        f"  Dependencies: {len(edges)}",
        "  Every rectangle is one CUDA graph node.",
        "  Every arrow is one dependency emitted by CUDA.",
        "endlegend",
    ]

    group_starts = {start: (name, end) for name, start, end in groups}
    for node in nodes:
        index = int(node["index"])
        if index in group_starts:
            group_name, group_end = group_starts[index]
            lines.append(f"' {group_name}: CUDA nodes {index}..{group_end}")
        kind = str(node["kind"])
        details = full_details(str(node["label"])) if label_mode == "full" else compact_details(kind, str(node["label"]))
        label_lines = [
            f"CUDA node {index}",
            kind,
            f"topoId={node['topology_id']}",
            *details,
        ]
        label = "\\n".join(puml_text(line) for line in label_lines)
        lines.append(f'rectangle "{label}" as N{index} {COLORS.get(kind, COLORS["UNKNOWN"])}')

    for edge in edges:
        lines.append(f"N{edge['src']} --> N{edge['dst']} : {edge['headlabel']}")

    lines.append("@enduml")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--labels", choices=("compact", "full"), default="compact")
    parser.add_argument(
        "--split-dir",
        type=Path,
        help="Also write separately renderable induced subgraphs for H2D and every captured step.",
    )
    args = parser.parse_args()

    nodes, edges = parse_dot(args.dot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_puml(args.output, args.dot.name, nodes, edges, args.labels)

    if args.split_dir:
        args.split_dir.mkdir(parents=True, exist_ok=True)
        for part, (group_name, start, end) in enumerate(find_groups(nodes)):
            part_nodes = [node for node in nodes if start <= int(node["index"]) <= end]
            part_edges = [
                edge
                for edge in edges
                if start <= int(edge["src"]) <= end and start <= int(edge["dst"]) <= end
            ]
            slug = "h2d" if part == 0 else f"step-{part}"
            part_path = args.split_dir / f"opennn-transformer-training.{slug}.puml"
            write_puml(
                part_path,
                args.dot.name,
                part_nodes,
                part_edges,
                args.labels,
                title=f"{group_name} — exact induced CUDA subgraph (nodes {start}..{end})",
            )

    counts = Counter(str(node["kind"]) for node in nodes)
    print(f"nodes={len(nodes)} edges={len(edges)}")
    print("types=" + ",".join(f"{kind}:{count}" for kind, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
