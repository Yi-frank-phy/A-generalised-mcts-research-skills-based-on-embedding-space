"""Synthesis report generation for DTE prototype."""

from __future__ import annotations

from .models import DTERunSpec, ForcedSynthesisRecord, SearchNode


def synthesize_report(
    spec: DTERunSpec,
    nodes: list[SearchNode],
    max_nodes: int = 8,
    forced_synthesis: ForcedSynthesisRecord | None = None,
) -> str:
    """Create a deterministic Markdown report from the current graph."""

    if forced_synthesis is not None and forced_synthesis.scope == "node_ids":
        selected = set(forced_synthesis.node_ids)
        report_nodes = [node for node in nodes if node.node_id in selected]
    else:
        report_nodes = nodes
    ranked = sorted(report_nodes, key=lambda n: (n.score if n.score is not None else n.confidence), reverse=True)
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

    if forced_synthesis is not None:
        lines.extend(
            [
                "## User-Interrupted Synthesis",
                f"- stop reason: `{forced_synthesis.stop_reason}`",
                f"- requested by: `{forced_synthesis.requested_by}`",
                f"- reason: {forced_synthesis.reason}",
                f"- scope: `{forced_synthesis.scope}`",
                "- selected nodes: "
                + (", ".join(f"`{node_id}`" for node_id in forced_synthesis.node_ids) or "all"),
                "- left unexplored frontier branches: "
                + (
                    ", ".join(f"`{node_id}`" for node_id in forced_synthesis.left_unexplored_node_ids)
                    or "none"
                ),
                "- control path: " + (f"`{forced_synthesis.control_path}`" if forced_synthesis.control_path else "n/a"),
                "",
                "This was an explicit user interruption honored by the backend at a safe boundary, not natural "
                "`entropy_plateau` convergence. Any listed frontier branches remain unresolved risk.",
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
