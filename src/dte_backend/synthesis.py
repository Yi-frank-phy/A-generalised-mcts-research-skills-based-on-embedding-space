"""Synthesis report generation for DTE prototype."""

from __future__ import annotations

from .models import DTERunSpec, SearchNode


def synthesize_report(spec: DTERunSpec, nodes: list[SearchNode], max_nodes: int = 8) -> str:
    """Create a deterministic Markdown report from the current graph."""

    ranked = sorted(nodes, key=lambda n: (n.score if n.score is not None else n.confidence), reverse=True)
    lines = [
        "# DTE Prototype Report",
        "",
        "## Problem",
        spec.problem,
        "",
        "## Goal",
        spec.goal,
        "",
        "## Search Summary",
        f"- Total nodes: {len(nodes)}",
        f"- Frontier nodes: {sum(1 for n in nodes if n.status == 'frontier')}",
        f"- Closed nodes: {sum(1 for n in nodes if n.status == 'closed')}",
        "",
        "## Top Nodes",
    ]

    for i, node in enumerate(ranked[:max_nodes], 1):
        score = node.score if node.score is not None else node.confidence
        lines.extend(
            [
                f"### {i}. {node.claim}",
                f"- id: `{node.node_id}`",
                f"- status: `{node.status}`",
                f"- score: {score:.3f}",
                f"- ucb: {node.ucb_score:.3f}" if node.ucb_score is not None else "- ucb: n/a",
                f"- parents: {', '.join(node.parent_ids) if node.parent_ids else 'none'}",
                f"- rationale: {node.rationale or 'n/a'}",
                f"- risks: {', '.join(node.risks) if node.risks else 'none'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Protocol Note",
            "This prototype report is generated only after nodes pass through the DTE Judge/Evolution/Expansion loop. Executor adapters may produce candidate nodes, but final synthesis belongs to DTE.",
            "",
        ]
    )
    return "\n".join(lines)
