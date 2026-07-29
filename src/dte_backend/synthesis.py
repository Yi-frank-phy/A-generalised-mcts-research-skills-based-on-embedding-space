"""Synthesis report generation for DTE prototype."""

from __future__ import annotations

from .models import DTERunSpec, ForcedSynthesisRecord, SearchNode, SynthesisControlRequest
from .relation_models import ProvisionalSynthesisSelection


def select_provisional_synthesis_nodes(
    nodes: list[SearchNode],
    *,
    graph_revision: int,
    synthesis_request: SynthesisControlRequest | None = None,
    max_nodes: int = 8,
    required_coverage_ids: list[str] | None = None,
) -> ProvisionalSynthesisSelection:
    """Select material science scope first and compact presentation headlines second."""

    eligible = [
        node
        for node in nodes
        if node.status in {"frontier", "closed"} and node.node_type != "synthesis"
    ]
    if synthesis_request is not None and synthesis_request.scope == "node_ids":
        requested = set(synthesis_request.node_ids)
        eligible = [node for node in eligible if node.node_id in requested]
        reason = "operator-requested committed node scope"
    else:
        reason = "coverage-aware material scope with a compact headline projection"

    ranked = sorted(
        eligible,
        key=lambda node: (
            -(node.score if node.score is not None else node.confidence),
            node.node_id,
        ),
    )
    by_id = {node.node_id: node for node in eligible}
    required = sorted(set(required_coverage_ids or []))
    representatives: list[SearchNode] = []
    unresolved: list[str] = []
    for coverage_id in required:
        candidates = [node for node in ranked if coverage_id in node.coverage_ids]
        if not candidates:
            unresolved.append(coverage_id)
            continue
        representatives.append(candidates[0])

    material_ids = {node.node_id for node in representatives}
    support_ids: set[str] = set()
    stack = list(material_ids)
    while stack:
        node_id = stack.pop()
        node = by_id.get(node_id)
        if node is None:
            continue
        for parent_id in node.parent_ids:
            if parent_id in by_id and parent_id not in material_ids:
                material_ids.add(parent_id)
                support_ids.add(parent_id)
                stack.append(parent_id)

    required_set = set(required)
    for node in ranked:
        if node.node_type != "counterexample":
            continue
        if (
            set(node.parent_ids).intersection(material_ids)
            or set(node.coverage_ids).intersection(required_set)
        ):
            material_ids.add(node.node_id)

    # An empty required-coverage contract is the legacy path.  Preserve its
    # historical top-N material selection exactly; App-native shared/strict
    # runs always provide stable seed coverage obligations.
    if not required:
        material_ids.update(node.node_id for node in ranked[:max_nodes])

    headline_ids: list[str] = []
    for node in representatives:
        if node.node_id in material_ids and node.node_id not in headline_ids:
            headline_ids.append(node.node_id)
    for node in ranked:
        if (
            node.node_type == "counterexample"
            and node.node_id in material_ids
            and node.node_id not in headline_ids
        ):
            headline_ids.append(node.node_id)
    for node in ranked:
        if node.node_id in material_ids and node.node_id not in headline_ids:
            headline_ids.append(node.node_id)

    return ProvisionalSynthesisSelection(
        selected_node_ids=headline_ids[:max_nodes],
        material_scope_node_ids=sorted(material_ids),
        support_dependency_node_ids=sorted(support_ids),
        unresolved_coverage_ids=unresolved,
        selection_reason=reason,
        selection_revision=graph_revision,
    )


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
        section_title = (
            "User-Interrupted Synthesis"
            if forced_synthesis.requested_by == "user"
            else "Main-Agent-Requested Synthesis"
        )
        actor_description = (
            "an explicit user interruption"
            if forced_synthesis.requested_by == "user"
            else "an authorized main-agent controller command"
        )
        lines.extend(
            [
                f"## {section_title}",
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
                f"This was {actor_description} honored by the backend at a safe boundary, not natural "
                "`entropy_plateau` convergence or algorithmic convergence. Any listed frontier branches remain "
                "unresolved risk.",
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
