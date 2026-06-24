"""Convert validated relation-oracle results into DTE proposals/tasks.

Relation oracles do not mutate the graph directly. They only produce observable
relation judgments. This module translates those judgments into typed backend
objects that the controller can inspect, persist, or later apply.
"""

from __future__ import annotations

from uuid import uuid4

from .models import MergeProposal, SearchNode
from .oracles import RelationOracleResult


def relation_result_to_outputs(
    relation: RelationOracleResult,
    nodes: list[SearchNode],
) -> tuple[MergeProposal | None, SearchNode | None]:
    """Convert a validated relation result into a proposal and optional task node.

    Returns:
        `(proposal, discriminator_node)`.
        - independent -> `(None, None)`
        - equivalent -> equivalent merge proposal
        - complementary -> complementary merge proposal with a merged SearchNode
        - conflict -> conflict merge proposal and, if supplied, a discriminator node
    """

    by_id = {node.node_id: node for node in nodes}
    source_nodes = [by_id[node_id] for node_id in relation.source_node_ids if node_id in by_id]
    if len(source_nodes) < 2:
        raise ValueError("relation result must refer to at least two known nodes")

    if relation.relation == "independent":
        return None, None

    if relation.relation == "equivalent":
        ranked = sorted(
            source_nodes,
            key=lambda node: (node.score if node.score is not None else node.confidence),
            reverse=True,
        )
        keep = ranked[0]
        absorbed = ranked[1:]
        return (
            MergeProposal(
                merge_type="equivalent_merge",
                source_node_ids=relation.source_node_ids,
                target_node_id=keep.node_id,
                rationale=relation.rationale,
                absorbed_node_ids=[node.node_id for node in absorbed],
            ),
            None,
        )

    if relation.relation == "complementary":
        merged = SearchNode(
            node_id=f"merge-{uuid4()}",
            node_type="merge",
            claim="Complementary synthesis candidate: " + " / ".join(node.claim for node in source_nodes[:3]),
            rationale=relation.rationale,
            assumptions=sorted({item for node in source_nodes for item in node.assumptions}),
            evidence=sorted({item for node in source_nodes for item in node.evidence}),
            risks=sorted({item for node in source_nodes for item in node.risks}),
            parent_ids=[node.node_id for node in source_nodes],
            confidence=max(node.confidence for node in source_nodes),
            status="frontier",
        )
        return (
            MergeProposal(
                merge_type="complementary_merge",
                source_node_ids=relation.source_node_ids,
                target_node_id=merged.node_id,
                rationale=relation.rationale,
                merged_node=merged,
            ),
            None,
        )

    if relation.relation == "conflict":
        proposal = MergeProposal(
            merge_type="conflict_merge",
            source_node_ids=relation.source_node_ids,
            target_node_id=None,
            rationale=relation.rationale,
        )
        discriminator = None
        if relation.discriminator_question:
            discriminator = SearchNode(
                node_id=f"discriminator-{uuid4()}",
                node_type="counterexample",
                claim=relation.discriminator_question,
                rationale="Discriminator task generated from conflict relation oracle.",
                assumptions=sorted({item for node in source_nodes for item in node.assumptions}),
                risks=["generated from unresolved branch conflict"],
                parent_ids=[node.node_id for node in source_nodes],
                confidence=0.5,
                status="frontier",
            )
        return proposal, discriminator

    raise ValueError(f"unsupported relation: {relation.relation}")
