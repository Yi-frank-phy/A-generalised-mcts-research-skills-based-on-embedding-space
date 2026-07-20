"""Deterministic provenance-only epistemic read model.

The read path consumes authoritative committed facts and never repairs a run.
Scientific verification, artifact evaluation, and researcher learning remain
outside DTE authority.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

from .app_driver import AppRunState
from .epistemic_models import (
    EpistemicDataQualityV1,
    EpistemicDependencyGraphV1,
    EpistemicEdgeRecordV1,
    EpistemicIndependenceSummaryV1,
    EpistemicStatementRecordV1,
    ImportantPathHandoffV1,
    MergeEpistemicProjectionV1,
    NodeEpistemicSummaryV1,
    RelationEpistemicProjectionV1,
    SelectedClaimHandoffV1,
    SourceProvenanceSummaryV1,
    TerminalEpistemicHandoffV1,
)
from .episode_models import ExecutorEpisodeOutput
from .observability import (
    _load_state_read_only,
    build_run_observability_summary,
)


def _committed_attempts(state: AppRunState):
    for episode in state.episodes:
        if episode.committed_attempt_id is None:
            continue
        for attempt in episode.attempts:
            if (
                attempt.attempt_id == episode.committed_attempt_id
                and attempt.status == "committed"
                and attempt.committed_result is not None
            ):
                yield episode, attempt
                break


def _relation_projections(state: AppRunState) -> list[RelationEpistemicProjectionV1]:
    return [
        RelationEpistemicProjectionV1(
            relation_record_id=record.relation_record_id,
            candidate_id=record.candidate_id,
            left_node_id=record.left_node_id,
            right_node_id=record.right_node_id,
            relation_type=record.relation_type,
            confidence=record.confidence,
            rationale=record.rationale,
            evidence_refs=list(record.evidence_refs),
            material_to_synthesis=record.material_to_synthesis,
            materiality_assessment=record.materiality_assessment,
            disclosure_required=record.disclosure_required,
            conflict_summary=record.observation.conflict_summary,
            episode_id=record.episode_id,
            attempt_id=record.attempt_id,
        )
        for record in state.relation_ledger
    ]


def _merge_projections(state: AppRunState) -> list[MergeEpistemicProjectionV1]:
    return [
        MergeEpistemicProjectionV1(
            merge_application_id=item.merge_application_id,
            relation_record_id=item.relation_record_id,
            canonical_node_id=item.canonical_node_id,
            absorbed_node_ids=list(item.absorbed_node_ids),
            source_node_ids=list(item.source_node_ids),
        )
        for item in state.merge_applications
    ]


def _dependency_graph(state: AppRunState) -> EpistemicDependencyGraphV1:
    return EpistemicDependencyGraphV1(
        run_id=state.run_id,
        node_claim_refs=[f"node-claim:{node.node_id}" for node in state.nodes],
        statements=[item.model_copy(deep=True) for item in state.epistemic_ledger.statements],
        edges=[item.model_copy(deep=True) for item in state.epistemic_ledger.edges],
        path_dispositions=[
            item.model_copy(deep=True) for item in state.epistemic_ledger.path_dispositions
        ],
        relation_projections=_relation_projections(state),
        merge_projections=_merge_projections(state),
    )


def _selected_ids(state: AppRunState) -> list[str]:
    return (
        []
        if state.provisional_synthesis_selection is None
        else list(state.provisional_synthesis_selection.selected_node_ids)
    )


def _search_dispositions(state: AppRunState, node_id: str) -> list[str]:
    node = next(item for item in state.nodes if item.node_id == node_id)
    selected = set(_selected_ids(state))
    values: list[str] = []
    if state.provisional_synthesis_selection is not None:
        values.append("selected" if node_id in selected else "not_selected")
    if node.status == "merged":
        values.append("merged")
    elif node.status == "closed":
        values.append("closed")
    elif node.status == "frontier" and node_id not in selected:
        if state.terminal_record is not None and state.terminal_record.source in {
            "max_iterations",
            "max_search_nodes",
        }:
            values.append("out_of_budget")
        elif state.controller_action in {"ready_for_synthesis", "run_complete"}:
            values.append("not_explored")
    return list(dict.fromkeys(values))


def _ref_to_statement(
    ref: str,
    statements: dict[str, EpistemicStatementRecordV1],
) -> EpistemicStatementRecordV1 | None:
    if not ref.startswith("epistemic:"):
        return None
    return statements.get(ref.removeprefix("epistemic:"))


def _record_refs(record) -> list[str]:
    refs = list(record.basis_refs)
    if isinstance(record, EpistemicEdgeRecordV1):
        refs.extend((record.source_ref, record.target_ref))
    return refs


def _claim_dependencies(
    claim_ref: str,
    edges: list[EpistemicEdgeRecordV1],
) -> tuple[set[str], set[str], set[str]]:
    """Traverse explicit requires/qualifies edges only; never infer from text."""

    required: set[str] = set()
    conditional: set[str] = set()
    derived_from: set[str] = set()
    queue: deque[str] = deque([claim_ref])
    visited: set[str] = set()
    outgoing: dict[str, list[EpistemicEdgeRecordV1]] = defaultdict(list)
    incoming: dict[str, list[EpistemicEdgeRecordV1]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source_ref].append(edge)
        incoming[edge.target_ref].append(edge)
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for edge in outgoing.get(current, []):
            if edge.relation_type == "requires":
                if edge.target_ref not in required:
                    required.add(edge.target_ref)
                    queue.append(edge.target_ref)
            elif edge.relation_type == "derived_from":
                derived_from.add(edge.target_ref)
                queue.append(edge.target_ref)
        for edge in incoming.get(current, []):
            if edge.relation_type == "qualifies":
                conditional.add(edge.source_ref)
                queue.append(edge.source_ref)
    return required, conditional, derived_from


def _incoming_sources(
    targets: set[str],
    edges: list[EpistemicEdgeRecordV1],
    relations: set[str],
) -> set[str]:
    return {
        edge.source_ref
        for edge in edges
        if edge.target_ref in targets and edge.relation_type in relations
    }


def _selected_claim(
    state: AppRunState,
    node_id: str,
) -> SelectedClaimHandoffV1:
    node = next(item for item in state.nodes if item.node_id == node_id)
    claim_ref = f"node-claim:{node_id}"
    edges = state.epistemic_ledger.edges
    statements = {
        item.statement_id: item for item in state.epistemic_ledger.statements
    }
    required, conditional, derived_from = _claim_dependencies(claim_ref, edges)
    dependency_targets = {claim_ref, *required, *conditional, *derived_from}
    supporting = _incoming_sources(dependency_targets, edges, {"supports"})
    challenging = _incoming_sources(
        dependency_targets, edges, {"challenges", "contradicts"}
    )
    unresolved = {
        ref
        for ref in required
        if not _incoming_sources({ref}, edges, {"supports"})
    }
    disposition_records = [
        item
        for item in state.epistemic_ledger.path_dispositions
        if item.target_node_id == node_id
    ]
    counterexamples = {
        ref
        for item in disposition_records
        if item.epistemic_disposition == "counterexample_found"
        for ref in item.basis_refs
    }
    related_refs = {
        *required,
        *conditional,
        *derived_from,
        *supporting,
        *challenging,
    }
    related_statements = [
        statement
        for ref in related_refs
        if (statement := _ref_to_statement(ref, statements)) is not None
    ]
    related_edges = [
        edge
        for edge in edges
        if edge.source_ref in dependency_targets | related_refs
        or edge.target_ref in dependency_targets | related_refs
    ]
    referenced_artifacts = sorted(
        {
            ref.removeprefix("artifact:")
            for record in [*related_statements, *related_edges]
            for ref in _record_refs(record)
            if ref.startswith("artifact:")
        }
    )
    relations = [
        record
        for record in state.relation_ledger
        if node_id in {record.left_node_id, record.right_node_id}
    ]
    merges = [
        item
        for item in state.merge_applications
        if node_id == item.canonical_node_id
        or node_id in item.source_node_ids
        or node_id in item.absorbed_node_ids
    ]
    sources = sorted(
        {
            *(item.source_type for item in related_statements),
            *(item.source_type for item in related_edges),
            *(item.source_type for item in disposition_records),
            *("agent_reported" for _ in relations),
            *("backend_derived" for _ in merges),
        }
    )
    selection_reason = (
        "provisional synthesis selection"
        if state.provisional_synthesis_selection is None
        else state.provisional_synthesis_selection.selection_reason
    )
    claim_episode_id = None
    claim_attempt_id = None
    for episode, attempt in _committed_attempts(state):
        result = attempt.committed_result
        if episode.role != "executor" or result is None:
            continue
        output = result.structured_output
        if not isinstance(output, ExecutorEpisodeOutput):
            continue
        if any(item.node_id == node_id for item in output.nodes):
            claim_episode_id = episode.episode_id
            claim_attempt_id = attempt.attempt_id
            break
    if claim_episode_id is not None and "agent_reported" not in sources:
        sources = sorted([*sources, "agent_reported"])
    producing_episode_ids = {
        item.episode_id for item in [*related_statements, *related_edges]
    }
    producing_attempt_ids = {
        item.attempt_id for item in [*related_statements, *related_edges]
    }
    if claim_episode_id is not None:
        producing_episode_ids.add(claim_episode_id)
    if claim_attempt_id is not None:
        producing_attempt_ids.add(claim_attempt_id)
    return SelectedClaimHandoffV1(
        node_id=node_id,
        claim_ref=claim_ref,
        claim=node.claim,
        claim_origin=(
            "executor_episode" if claim_episode_id is not None else "initial_node"
        ),
        claim_source_type=(
            "agent_reported" if claim_episode_id is not None else None
        ),
        claim_producing_episode_id=claim_episode_id,
        claim_producing_attempt_id=claim_attempt_id,
        selection_reason=selection_reason,
        search_dispositions=_search_dispositions(state, node_id),
        epistemic_dispositions=list(
            dict.fromkeys(item.epistemic_disposition for item in disposition_records)
        ),
        epistemic_disposition_records=[
            item.model_copy(deep=True) for item in disposition_records
        ],
        required_assumption_refs=sorted(required),
        supporting_record_refs=sorted(supporting),
        challenging_record_refs=sorted(challenging),
        conditional_dependency_refs=sorted(conditional),
        derived_from_refs=sorted(derived_from),
        unresolved_dependency_refs=sorted(unresolved),
        counterexample_refs=sorted(counterexamples),
        producing_episode_ids=sorted(producing_episode_ids),
        producing_attempt_ids=sorted(producing_attempt_ids),
        referenced_artifacts=referenced_artifacts,
        relation_record_ids=[item.relation_record_id for item in relations],
        merge_application_ids=[item.merge_application_id for item in merges],
        source_types=sources,
    )


def _node_summaries(
    state: AppRunState,
    selected_claims: dict[str, SelectedClaimHandoffV1],
) -> list[NodeEpistemicSummaryV1]:
    statements_by_node: dict[str, list[EpistemicStatementRecordV1]] = defaultdict(list)
    dispositions_by_node = defaultdict(list)
    for statement in state.epistemic_ledger.statements:
        statements_by_node[statement.target_node_id].append(statement)
    for disposition in state.epistemic_ledger.path_dispositions:
        dispositions_by_node[disposition.target_node_id].append(disposition)
    summaries = []
    selected = set(selected_claims)
    for node in state.nodes:
        claim = selected_claims.get(node.node_id)
        relations = [
            item.relation_record_id
            for item in state.relation_ledger
            if node.node_id in {item.left_node_id, item.right_node_id}
        ]
        merges = [
            item.merge_application_id
            for item in state.merge_applications
            if node.node_id == item.canonical_node_id
            or node.node_id in item.source_node_ids
        ]
        summaries.append(
            NodeEpistemicSummaryV1(
                node_id=node.node_id,
                claim_ref=f"node-claim:{node.node_id}",
                claim=node.claim,
                selected_for_synthesis=node.node_id in selected,
                search_dispositions=_search_dispositions(state, node.node_id),
                epistemic_dispositions=list(
                    dict.fromkeys(
                        item.epistemic_disposition
                        for item in dispositions_by_node[node.node_id]
                    )
                ),
                epistemic_disposition_records=[
                    item.model_copy(deep=True)
                    for item in dispositions_by_node[node.node_id]
                ],
                statement_refs=[
                    f"epistemic:{item.statement_id}"
                    for item in statements_by_node[node.node_id]
                ],
                supporting_record_refs=[] if claim is None else claim.supporting_record_refs,
                challenging_record_refs=[] if claim is None else claim.challenging_record_refs,
                required_assumption_refs=[] if claim is None else claim.required_assumption_refs,
                derived_from_refs=[] if claim is None else claim.derived_from_refs,
                unresolved_dependency_refs=[] if claim is None else claim.unresolved_dependency_refs,
                relation_record_ids=relations,
                merge_application_ids=merges,
            )
        )
    return summaries


def _attempt_models(state: AppRunState):
    values = []
    for episode, attempt in _committed_attempts(state):
        result = attempt.committed_result
        assert result is not None
        if episode.role not in {"executor", "judge", "relation"}:
            continue
        values.append(
            (
                episode.role,
                episode.episode_id,
                attempt.attempt_id,
                result.runtime_diagnostics.model,
                result.runtime_diagnostics.runtime_profile,
            )
        )
    return values


def _independence_summary(
    state: AppRunState,
    selected: list[SelectedClaimHandoffV1],
) -> EpistemicIndependenceSummaryV1:
    attempts = _attempt_models(state)
    models = [item[3] for item in attempts]
    if not attempts or not any(models):
        model_status = "unavailable"
    elif all(models):
        model_status = "available"
    else:
        model_status = "partial"
    same_cross: int | None = None
    different_cross: int | None = None
    if model_status == "available":
        same_cross = 0
        different_cross = 0
        for index, left in enumerate(attempts):
            for right in attempts[index + 1 :]:
                if left[0] == right[0]:
                    continue
                if left[3] == right[3]:
                    same_cross += 1
                else:
                    different_cross += 1

    profiles = [item[4] for item in attempts]
    if not attempts or not any(profiles):
        profile_status = "unavailable"
    elif all(profiles):
        profile_status = "available"
    else:
        profile_status = "partial"
    same_profile: int | None = None
    different_profile: int | None = None
    if profile_status == "available":
        same_profile = 0
        different_profile = 0
        for index, left in enumerate(attempts):
            for right in attempts[index + 1 :]:
                if left[0] == right[0]:
                    continue
                if left[4] == right[4]:
                    same_profile += 1
                else:
                    different_profile += 1

    model_by_attempt = {
        (episode_id, attempt_id): model
        for _, episode_id, attempt_id, model, _ in attempts
    }
    statement_models = {
        f"epistemic:{item.statement_id}": model_by_attempt.get(
            (item.episode_id, item.attempt_id)
        )
        for item in state.epistemic_ledger.statements
    }
    node_models: dict[str, str | None] = {
        f"node-claim:{node.node_id}": None for node in state.nodes
    }
    for episode, attempt in _committed_attempts(state):
        result = attempt.committed_result
        if (
            episode.role != "executor"
            or result is None
            or not isinstance(result.structured_output, ExecutorEpisodeOutput)
        ):
            continue
        model = result.runtime_diagnostics.model
        for child in result.structured_output.nodes:
            node_models[f"node-claim:{child.node_id}"] = model
    ref_models = {**node_models, **statement_models}
    same_support_challenge: int | None = None
    different_support_challenge: int | None = None
    if model_status == "available":
        same_support_challenge = 0
        different_support_challenge = 0
        for edge in state.epistemic_ledger.edges:
            if edge.relation_type not in {"supports", "challenges", "contradicts"}:
                continue
            source_model = ref_models.get(edge.source_ref)
            target_model = ref_models.get(edge.target_ref)
            if source_model is None or target_model is None:
                continue
            if source_model == target_model:
                same_support_challenge += 1
            else:
                different_support_challenge += 1

    statements = {
        f"epistemic:{item.statement_id}": item
        for item in state.epistemic_ledger.statements
    }
    agent_only = external = without_support = self_referential = 0
    unresolved = 0
    for item in selected:
        supports = [statements[ref] for ref in item.supporting_record_refs if ref in statements]
        support_edges = [
            edge
            for edge in state.epistemic_ledger.edges
            if edge.relation_type == "supports"
            and edge.source_ref in item.supporting_record_refs
        ]
        if not item.supporting_record_refs:
            without_support += 1
        support_sources = {
            *(record.source_type for record in supports),
            *(edge.source_type for edge in support_edges),
        }
        if item.supporting_record_refs and support_sources == {"agent_reported"}:
            agent_only += 1
        if "external_artifact_backed" in support_sources:
            external += 1
        if item.unresolved_dependency_refs:
            unresolved += 1
        self_referential += sum(
            edge.relation_type == "supports"
            and edge.target_ref == item.claim_ref
            and edge.source_ref in statements
            and statements[edge.source_ref].episode_id == edge.episode_id
            for edge in state.epistemic_ledger.edges
        )

    flags: list[str] = []
    if model_status == "unavailable":
        flags.append("model_metadata_unavailable")
    elif model_status == "partial":
        flags.append("model_metadata_partial")
    if same_cross:
        flags.append("same_model_cross_role_correlation")
    if same_profile:
        flags.append("same_runtime_profile_cross_role_correlation")
    if agent_only:
        flags.append("selected_claims_supported_only_by_agent_reports")
    if without_support:
        flags.append("selected_claims_without_structured_support")
    if unresolved:
        flags.append("selected_claims_have_unresolved_assumptions")
    if self_referential:
        flags.append("self_referential_support")
    return EpistemicIndependenceSummaryV1(
        model_metadata_status=model_status,
        model_metadata_available=model_status == "available",
        same_model_cross_role_count=same_cross,
        different_model_cross_role_count=different_cross,
        runtime_profile_metadata_status=profile_status,
        same_runtime_profile_cross_role_count=same_profile,
        different_runtime_profile_cross_role_count=different_profile,
        same_model_support_challenge_count=same_support_challenge,
        different_model_support_challenge_count=different_support_challenge,
        selected_claim_count=len(selected),
        agent_only_supported_selected_claim_count=agent_only,
        external_artifact_backed_selected_claim_count=external,
        selected_claims_with_unresolved_assumptions=unresolved,
        selected_claims_without_structured_support_count=without_support,
        self_referential_support_count=self_referential,
        risk_flags=flags,
    )


def _statement_lists(
    state: AppRunState,
    selected: list[SelectedClaimHandoffV1],
) -> tuple[
    list[EpistemicStatementRecordV1],
    list[EpistemicStatementRecordV1],
    list[EpistemicStatementRecordV1],
    list[EpistemicStatementRecordV1],
    list[EpistemicStatementRecordV1],
    list[EpistemicStatementRecordV1],
]:
    statements = {
        f"epistemic:{item.statement_id}": item
        for item in state.epistemic_ledger.statements
    }
    assumption_refs = {
        ref for item in selected for ref in item.required_assumption_refs
    }
    support_refs = {ref for item in selected for ref in item.supporting_record_refs}
    challenge_refs = {ref for item in selected for ref in item.challenging_record_refs}
    conditional_refs = {
        ref for item in selected for ref in item.conditional_dependency_refs
    }
    unresolved_refs = {
        ref for item in selected for ref in item.unresolved_dependency_refs
    }
    counterexample_refs = {
        ref for item in selected for ref in item.counterexample_refs
    }

    def by_refs(refs: set[str]) -> list[EpistemicStatementRecordV1]:
        return [
            statement
            for ref in sorted(refs)
            if (statement := statements.get(ref)) is not None
        ]
    return (
        [
            item
            for item in by_refs(assumption_refs)
            if item.statement_type in {"assumption", "open_question"}
        ],
        by_refs(support_refs),
        by_refs(challenge_refs),
        by_refs(conditional_refs | assumption_refs),
        by_refs(unresolved_refs),
        by_refs(counterexample_refs),
    )


def _important_paths(
    state: AppRunState,
    selected_ids: set[str],
) -> list[ImportantPathHandoffV1]:
    result = []
    dispositions = defaultdict(list)
    for item in state.epistemic_ledger.path_dispositions:
        dispositions[item.target_node_id].append(item)
    for node in state.nodes:
        if node.node_id in selected_ids:
            continue
        records = dispositions[node.node_id]
        search = _search_dispositions(state, node.node_id)
        epistemic = list(
            dict.fromkeys(item.epistemic_disposition for item in records)
        )
        if not records and not any(
            value in search for value in {"merged", "closed", "out_of_budget", "not_explored"}
        ):
            continue
        result.append(
            ImportantPathHandoffV1(
                node_id=node.node_id,
                claim=node.claim,
                search_dispositions=search,
                epistemic_dispositions=epistemic,
                epistemic_disposition_records=[
                    item.model_copy(deep=True) for item in records
                ],
                basis_refs=sorted({ref for item in records for ref in item.basis_refs}),
                explanation=(
                    "; ".join(item.explanation for item in records)
                    if records
                    else "search lifecycle only; no structured epistemic challenge or contradiction recorded"
                ),
            )
        )
    return result


def build_terminal_epistemic_handoff(
    run_dir: str | Path,
) -> TerminalEpistemicHandoffV1:
    """Build the formal machine handoff from current persisted facts only."""

    directory = Path(run_dir)
    state, _, partial, validation_error, reconstructed, raw_state = (
        _load_state_read_only(directory)
    )
    # Reuse the existing deterministic operational read model as required by
    # the handoff contract.  Its value is not copied into a second fact source.
    operational_summary = build_run_observability_summary(directory)
    dependency_graph = _dependency_graph(state)
    selected = [_selected_claim(state, node_id) for node_id in _selected_ids(state)]
    selected_by_id = {item.node_id: item for item in selected}
    (
        key_assumptions,
        supporting,
        challenging,
        conditional,
        unresolved,
        counterexamples,
    ) = _statement_lists(state, selected)
    independence = _independence_summary(state, selected)
    artifact_refs = sorted(
        {
            ref.removeprefix("artifact:")
            for collection in (
                state.epistemic_ledger.statements,
                state.epistemic_ledger.edges,
                state.epistemic_ledger.path_dispositions,
            )
            for item in collection
            for ref in _record_refs(item)
            if ref.startswith("artifact:")
        }
    )
    missing_artifacts = [
        path for path in artifact_refs if not (directory / path).is_file()
    ]
    unresolved_references: list[str] = []
    structured_episodes = {
        (item.episode_id, item.attempt_id)
        for collection in (
            state.epistemic_ledger.statements,
            state.epistemic_ledger.edges,
            state.epistemic_ledger.path_dispositions,
        )
        for item in collection
    }
    ledger_present = "epistemic_ledger" in raw_state
    limitations: list[str] = []
    if not ledger_present:
        limitations.append(
            "legacy run has no authoritative epistemic ledger; dependency graph is unavailable"
        )
    if not state.epistemic_ledger.statements and not state.epistemic_ledger.edges:
        limitations.append(
            "no structured epistemic statements or dependency edges were committed"
        )
    if any(item.claim_origin == "initial_node" for item in selected):
        limitations.append(
            "initial node claims predate episode-level source metadata; their claim source is unavailable"
        )
    if missing_artifacts:
        limitations.append("one or more referenced run artifacts are currently missing")
    if unresolved_references:
        limitations.append(
            "one or more referenced researcher-learning records are currently missing"
        )
    if validation_error:
        limitations.append(f"persisted state validation warning: {validation_error}")
    if any(
        item.startswith("deprecated_human_confirmed_epistemic_records:")
        for item in reconstructed
    ):
        limitations.append(
            "legacy human-confirmation epistemic records were isolated from this "
            "read-only handoff; they are deprecated and carry no verification authority"
        )
    if any(
        item.startswith("deprecated_learning_epistemic_references:")
        for item in reconstructed
    ):
        limitations.append(
            "legacy learning: references were isolated from this read-only handoff; "
            "researcher learning is outside the current DTE contract"
        )
    data_status = (
        "unavailable"
        if not ledger_present
        else "partial"
        if partial or validation_error or missing_artifacts or unresolved_references
        else "available"
    )
    all_records = [
        *state.epistemic_ledger.statements,
        *state.epistemic_ledger.edges,
        *state.epistemic_ledger.path_dispositions,
    ]
    source_counts = {
        source: sum(item.source_type == source for item in all_records)
        for source in (
            "agent_reported",
            "external_artifact_backed",
            "backend_derived",
        )
    }
    source_counts["agent_reported"] += len(
        dependency_graph.relation_projections
    )
    source_counts["backend_derived"] += len(
        dependency_graph.merge_projections
    )
    material_conflicts = [
        item
        for item in dependency_graph.relation_projections
        if item.relation_type == "conflict" and item.material_to_synthesis
    ]
    disclosures = [
        item for item in dependency_graph.relation_projections if item.disclosure_required
    ]
    heuristics = [
        item
        for item in state.epistemic_ledger.statements
        if item.statement_type == "heuristic"
    ]
    failure_modes = [
        item
        for item in state.epistemic_ledger.statements
        if item.statement_type == "failure_mode"
    ]
    return TerminalEpistemicHandoffV1(
        run_id=state.run_id,
        terminal_action=(
            state.controller_action
            if state.controller_action in {"ready_for_synthesis", "run_complete"}
            else None
        ),
        terminal_reason=operational_summary.run.terminal_reason,
        terminal_source=operational_summary.run.terminal_source,
        selected_claims=selected,
        key_assumptions=key_assumptions,
        supporting_evidence=supporting,
        challenging_evidence=challenging,
        conditional_dependencies=conditional,
        unresolved_dependencies=unresolved,
        material_conflicts=material_conflicts,
        relation_disclosures=disclosures,
        counterexamples=counterexamples,
        counterexample_refs=sorted(
            {ref for item in selected for ref in item.counterexample_refs}
        ),
        important_abandoned_or_inconclusive_paths=_important_paths(
            state, set(selected_by_id)
        ),
        possible_transferable_heuristics=heuristics,
        transferable_failure_modes=failure_modes,
        source_provenance=SourceProvenanceSummaryV1(
            agent_reported_record_count=source_counts["agent_reported"],
            external_artifact_backed_record_count=source_counts[
                "external_artifact_backed"
            ],
            backend_derived_record_count=source_counts["backend_derived"],
        ),
        independence_summary=independence,
        node_summaries=_node_summaries(state, selected_by_id),
        dependency_graph=dependency_graph,
        data_quality=EpistemicDataQualityV1(
            epistemic_data_status=data_status,
            structured_contribution_episode_count=len(structured_episodes),
            missing_artifacts=missing_artifacts,
            unresolved_references=unresolved_references,
            inconsistent_but_recoverable_records=(
                [] if validation_error is None else [validation_error]
            ),
            model_metadata_status=independence.model_metadata_status,
            operational_observability_status=(
                operational_summary.run.observability_status
            ),
            operational_observability_limitations=list(
                operational_summary.data_quality.limitations
            ),
            limitations=limitations,
        ),
    )


def render_epistemic_text(handoff: TerminalEpistemicHandoffV1) -> str:
    """Render a compact human handoff without changing the JSON contract."""

    def source_label(source_type: str) -> str:
        return (
            "artifact_referenced"
            if source_type == "external_artifact_backed"
            else source_type
        )

    def statement_lines(items: list[EpistemicStatementRecordV1]) -> list[str]:
        return [f"- [{source_label(item.source_type)}] {item.text}" for item in items]

    lines = [
        f"Epistemic handoff for run {handoff.run_id}",
        "",
        "Provenance boundary",
        "- artifact_referenced means only that a record points to an external "
        "artifact; DTE does not check the artifact, its assumptions, its "
        "applicability, or the scientific claim.",
        "- DTE preserves provenance and uncertainty; scientific judgment remains "
        "outside backend authority.",
        "",
        "Current most credible conclusions",
    ]
    lines.extend(
        [
            f"- [{item.claim_source_type or 'source_unavailable'}] "
            f"{item.claim} ({item.claim_ref})"
            for item in handoff.selected_claims
        ]
        or ["- No provisional-selected node claim is available."]
    )
    sections = [
        ("Key dependency assumptions", handoff.key_assumptions),
        ("Main supporting evidence", handoff.supporting_evidence),
        ("Main challenging evidence", handoff.challenging_evidence),
        ("Most dangerous unresolved questions", handoff.unresolved_dependencies),
    ]
    for title, items in sections:
        lines.extend(["", title])
        lines.extend(statement_lines(items) or ["- None recorded."])
    lines.extend(["", "Explicit counterexamples"])
    counterexample_dispositions = [
        item
        for item in handoff.dependency_graph.path_dispositions
        if item.epistemic_disposition == "counterexample_found"
    ]
    lines.extend(
        [
            f"- [{source_label(item.source_type)}] {ref}: {item.explanation}"
            for item in counterexample_dispositions
            for ref in item.basis_refs
        ]
        or ["- None recorded."]
    )
    lines.extend(["", "Important paths not carried forward"])
    lines.extend(
        [
            f"- {item.claim}: search={','.join(item.search_dispositions) or 'none'}; "
            "epistemic="
            + (
                ",".join(
                    f"{record.epistemic_disposition}[{source_label(record.source_type)}]"
                    for record in item.epistemic_disposition_records
                )
                or "none"
            )
            for item in handoff.important_abandoned_or_inconclusive_paths
        ]
        or ["- None recorded."]
    )
    lines.extend(["", "Possible transferable judgments"])
    lines.extend(
        statement_lines(
            [
                *handoff.possible_transferable_heuristics,
                *handoff.transferable_failure_modes,
            ]
        )
        or ["- None reported by an episode for possible researcher use."]
    )
    lines.extend(["", "Correlated-error risk indicators"])
    lines.extend(
        [f"- {flag}" for flag in handoff.independence_summary.risk_flags]
        or ["- No indicator was emitted from available metadata."]
    )
    lines.extend(["", "Data-quality limitations"])
    lines.extend(
        [f"- {item}" for item in handoff.data_quality.limitations]
        or ["- None recorded."]
    )
    return "\n".join(lines) + "\n"
