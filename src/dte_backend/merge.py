"""Merge skeleton for turning beam tree search into graph search.

This is intentionally conservative. It only implements deterministic equivalent
claim merging now, while leaving complementary/conflict merge types in the data
model for later stronger model-backed merge decisions.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .models import MergeProposal, SearchNode


def normalize_claim(text: str) -> str:
    """Normalize claim text for exact-equivalence merge proposals."""

    lowered = text.casefold().strip()
    return re.sub(r"\s+", " ", lowered)


def propose_equivalent_merges(nodes: list[SearchNode]) -> list[MergeProposal]:
    """Propose equivalent merges for frontier nodes with identical normalized claims."""

    groups: dict[str, list[SearchNode]] = defaultdict(list)
    for node in nodes:
        if node.status == "frontier" and node.node_type != "synthesis":
            groups[normalize_claim(node.claim)].append(node)

    proposals: list[MergeProposal] = []
    for claim_key, group in groups.items():
        if len(group) < 2:
            continue
        # Keep the highest score/confidence node as the representative.
        ranked = sorted(group, key=lambda n: (n.score if n.score is not None else n.confidence), reverse=True)
        keep = ranked[0]
        absorbed = ranked[1:]
        proposals.append(
            MergeProposal(
                merge_type="equivalent_merge",
                source_node_ids=[n.node_id for n in group],
                target_node_id=keep.node_id,
                rationale=f"Equivalent normalized claim: {claim_key!r}; keep highest-scoring representative.",
                merged_node=None,
                absorbed_node_ids=[n.node_id for n in absorbed],
            )
        )
    return proposals


def apply_equivalent_merges(nodes: list[SearchNode]) -> list[MergeProposal]:
    """Apply conservative equivalent merges in-place and return proposals."""

    proposals = propose_equivalent_merges(nodes)
    by_id = {node.node_id: node for node in nodes}
    for proposal in proposals:
        for absorbed_id in proposal.absorbed_node_ids:
            node = by_id.get(absorbed_id)
            if node is None:
                continue
            node.status = "merged"
            node.expansion_budget = 0
            node.risks.append(f"merged into {proposal.target_node_id} by equivalent_merge")
    return proposals
