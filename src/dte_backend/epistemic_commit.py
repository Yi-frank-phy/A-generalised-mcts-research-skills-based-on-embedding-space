"""Validation and stable identity resolution for episode epistemic facts.

This module is deliberately content-agnostic.  It validates authority,
identity, lifecycle provenance, and safe artifact references; it never decides
whether a scientific statement is true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .epistemic_models import (
    EpistemicContributionBundle,
    EpistemicEdgeRecordV1,
    EpistemicLedgerV1,
    EpistemicStatementRecordV1,
    PathDispositionRecordV1,
    stable_epistemic_id,
)
from .episode_models import EpisodeRequest, EpisodeResult


@dataclass(frozen=True)
class EpistemicReferenceContext:
    """Run-scoped identities which live outside :class:`EpisodeGraph`."""

    committed_episode_attempts: set[tuple[str, str]] = field(default_factory=set)
    artifact_paths: set[str] = field(default_factory=set)
    user_confirmed_learning_ids: set[str] = field(default_factory=set)


def _safe_artifact_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe epistemic artifact reference: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe epistemic artifact reference: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"unsafe epistemic artifact reference: {value!r}")
    return normalized


def _all_record_ids(ledger: EpistemicLedgerV1) -> set[str]:
    return {
        *(item.statement_id for item in ledger.statements),
        *(item.edge_id for item in ledger.edges),
        *(item.disposition_id for item in ledger.path_dispositions),
    }


def _external_basis_present(refs: list[str]) -> bool:
    return any(ref.startswith(("artifact:", "external:")) for ref in refs)


def prepare_epistemic_commit(
    *,
    graph,
    request: EpisodeRequest,
    result: EpisodeResult,
    bundle: EpistemicContributionBundle | None,
    authorized_node_ids: set[str],
    committed_at: str,
    context: EpistemicReferenceContext | None = None,
) -> EpistemicLedgerV1:
    """Return the next ledger or raise before any caller-visible mutation."""

    next_ledger = graph.epistemic_ledger.model_copy(deep=True)
    if bundle is None:
        return next_ledger
    if request.role not in {"executor", "judge"}:
        raise ValueError("epistemic contributions require an Executor or Judge episode")

    reference_context = context or EpistemicReferenceContext()
    committed_node_ids = {node.node_id for node in graph.nodes} | authorized_node_ids
    existing_record_ids = _all_record_ids(next_ledger)
    statement_ids = {
        statement.local_id: stable_epistemic_id(
            "epistmt",
            run_id=request.run_id,
            episode_id=request.episode_id,
            attempt_id=request.attempt_id,
            output_hash=result.output_hash,
            local_id=statement.local_id,
            record_type="statement",
        )
        for statement in bundle.statements
    }
    statement_target_nodes = {
        **{
            f"epistemic:{item.statement_id}": item.target_node_id
            for item in next_ledger.statements
        },
        **{
            f"epistemic:{statement_ids[item.local_id]}": item.target_node_id
            for item in bundle.statements
        },
    }

    def ref_target_node(ref: str) -> str | None:
        if ref.startswith("node-claim:"):
            return ref.removeprefix("node-claim:")
        return statement_target_nodes.get(ref)

    def resolve(ref: str) -> str:
        if ref.startswith("local-statement:"):
            local_id = ref.removeprefix("local-statement:")
            statement_id = statement_ids.get(local_id)
            if statement_id is None:
                raise ValueError(f"unknown local epistemic reference: {ref}")
            return f"epistemic:{statement_id}"
        if ref.startswith("node-claim:"):
            node_id = ref.removeprefix("node-claim:")
            if node_id not in committed_node_ids:
                raise ValueError(f"unknown epistemic reference: {ref}")
            if node_id not in authorized_node_ids:
                raise ValueError(f"epistemic node reference exceeds episode authority: {ref}")
            return ref
        if ref.startswith("epistemic:"):
            record_id = ref.removeprefix("epistemic:")
            if record_id not in existing_record_ids and record_id not in statement_ids.values():
                raise ValueError(f"unknown epistemic reference: {ref}")
            return ref
        if ref.startswith("relation:"):
            record_id = ref.removeprefix("relation:")
            if record_id not in {
                item.relation_record_id for item in graph.relation_ledger
            }:
                raise ValueError(f"unknown epistemic reference: {ref}")
            return ref
        if ref.startswith("merge:"):
            application_id = ref.removeprefix("merge:")
            if application_id not in {
                item.merge_application_id for item in graph.merge_applications
            }:
                raise ValueError(f"unknown epistemic reference: {ref}")
            return ref
        if ref.startswith("episode-result:"):
            identity = ref.removeprefix("episode-result:").split(":")
            if len(identity) != 2 or tuple(identity) not in {
                *reference_context.committed_episode_attempts,
                (request.episode_id, request.attempt_id),
            }:
                raise ValueError(f"unknown epistemic reference: {ref}")
            return ref
        if ref.startswith("artifact:"):
            path = _safe_artifact_path(ref.removeprefix("artifact:"))
            if path not in reference_context.artifact_paths:
                raise ValueError(f"unknown epistemic reference: {ref}")
            return f"artifact:{path}"
        if ref.startswith("external:"):
            if not ref.removeprefix("external:").strip():
                raise ValueError(f"unknown epistemic reference: {ref}")
            return ref
        if ref.startswith("learning:"):
            learning_id = ref.removeprefix("learning:")
            if learning_id not in reference_context.user_confirmed_learning_ids:
                raise ValueError(f"unknown epistemic reference: {ref}")
            return ref
        if ref == f"run:{request.run_id}":
            return ref
        raise ValueError(f"unknown epistemic reference: {ref}")

    def validate_source(source_type: str, refs: list[str]) -> None:
        if source_type not in {"agent_reported", "external_artifact_backed"}:
            raise ValueError(
                "episode epistemic source_type must be agent_reported or "
                "external_artifact_backed"
            )
        if source_type == "external_artifact_backed" and not _external_basis_present(refs):
            raise ValueError(
                "external_artifact_backed epistemic records require an artifact: "
                "or external: basis reference"
            )

    prepared_statements: list[EpistemicStatementRecordV1] = []
    for statement in bundle.statements:
        if statement.target_node_id not in authorized_node_ids:
            raise ValueError(
                "epistemic statement target exceeds episode authority: "
                f"{statement.target_node_id}"
            )
        resolved_basis = [resolve(ref) for ref in statement.basis_refs]
        validate_source(statement.source_type, resolved_basis)
        prepared_statements.append(
            EpistemicStatementRecordV1(
                statement_id=statement_ids[statement.local_id],
                local_id=statement.local_id,
                statement_type=statement.statement_type,
                text=statement.text,
                target_node_id=statement.target_node_id,
                source_type=statement.source_type,
                basis_refs=resolved_basis,
                run_id=request.run_id,
                episode_id=request.episode_id,
                attempt_id=request.attempt_id,
                role=request.role,
                output_hash=result.output_hash,
                committed_at=committed_at,
            )
        )

    prepared_edges: list[EpistemicEdgeRecordV1] = []
    for edge in bundle.edges:
        refs = [edge.source_ref, edge.target_ref, *edge.basis_refs]
        validate_source(edge.source_type, refs)
        resolved_source = resolve(edge.source_ref)
        resolved_target = resolve(edge.target_ref)
        target_node = ref_target_node(resolved_target)
        source_node = ref_target_node(resolved_source)
        if target_node is not None and target_node not in authorized_node_ids:
            raise ValueError(
                "epistemic edge target exceeds episode authority: "
                f"{edge.target_ref}"
            )
        if target_node is None and source_node not in authorized_node_ids:
            raise ValueError(
                "epistemic edge lacks an authorized node anchor"
            )
        prepared_edges.append(
            EpistemicEdgeRecordV1(
                edge_id=stable_epistemic_id(
                    "epiedge",
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    attempt_id=request.attempt_id,
                    output_hash=result.output_hash,
                    local_id=edge.local_id,
                    record_type="edge",
                ),
                local_id=edge.local_id,
                source_ref=resolved_source,
                target_ref=resolved_target,
                relation_type=edge.relation_type,
                source_type=edge.source_type,
                basis_refs=[resolve(ref) for ref in edge.basis_refs],
                explanation=edge.explanation,
                run_id=request.run_id,
                episode_id=request.episode_id,
                attempt_id=request.attempt_id,
                role=request.role,
                output_hash=result.output_hash,
                committed_at=committed_at,
            )
        )

    prepared_dispositions: list[PathDispositionRecordV1] = []
    for disposition in bundle.path_dispositions:
        if disposition.target_node_id not in authorized_node_ids:
            raise ValueError(
                "epistemic disposition target exceeds episode authority: "
                f"{disposition.target_node_id}"
            )
        resolved_basis = [resolve(ref) for ref in disposition.basis_refs]
        validate_source(disposition.source_type, resolved_basis)
        prepared_dispositions.append(
            PathDispositionRecordV1(
                disposition_id=stable_epistemic_id(
                    "epidisp",
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    attempt_id=request.attempt_id,
                    output_hash=result.output_hash,
                    local_id=disposition.local_id,
                    record_type="path_disposition",
                ),
                local_id=disposition.local_id,
                target_node_id=disposition.target_node_id,
                epistemic_disposition=disposition.epistemic_disposition,
                source_type=disposition.source_type,
                basis_refs=resolved_basis,
                explanation=disposition.explanation,
                run_id=request.run_id,
                episode_id=request.episode_id,
                attempt_id=request.attempt_id,
                role=request.role,
                output_hash=result.output_hash,
                committed_at=committed_at,
            )
        )

    new_ids = {
        *(item.statement_id for item in prepared_statements),
        *(item.edge_id for item in prepared_edges),
        *(item.disposition_id for item in prepared_dispositions),
    }
    duplicates = sorted(existing_record_ids & new_ids)
    if duplicates:
        raise ValueError(f"duplicate epistemic stable ID: {duplicates[0]}")

    next_ledger.statements.extend(prepared_statements)
    next_ledger.edges.extend(prepared_edges)
    next_ledger.path_dispositions.extend(prepared_dispositions)
    return EpistemicLedgerV1.model_validate(next_ledger.model_dump(mode="json"))
