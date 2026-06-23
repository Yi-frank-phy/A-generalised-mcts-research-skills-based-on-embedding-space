"""Codex-app-facing Markdown artifacts for DTE runs.

These files are the lightweight frontend: Codex app / main agent can show them
and summarize them instead of requiring a custom DTE dashboard.
"""

from __future__ import annotations

import json

from .runner import RunResult


def render_frontier_markdown(result: RunResult) -> str:
    frontier = [n for n in result.nodes if n.status == "frontier"]
    lines = ["# DTE Frontier", "", f"Frontier nodes: {len(frontier)}", ""]
    lines.append("| id | score | uncertainty | ucb | claim |")
    lines.append("|---|---:|---:|---:|---|")
    for node in frontier:
        score = "n/a" if node.score is None else f"{node.score:.3f}"
        uncertainty = "n/a" if node.uncertainty is None else f"{node.uncertainty:.3f}"
        ucb = "n/a" if node.ucb_score is None else f"{node.ucb_score:.3f}"
        claim = node.claim.replace("|", "\\|")
        lines.append(f"| `{node.node_id}` | {score} | {uncertainty} | {ucb} | {claim} |")
    return "\n".join(lines) + "\n"


def render_entropy_trace_markdown(result: RunResult) -> str:
    lines = ["# DTE Entropy / Temperature Trace", ""]
    lines.append("| iteration | entropy | delta | normalized temperature | stop reason |")
    lines.append("|---:|---:|---:|---:|---|")
    for trace in result.traces:
        state = trace.entropy_state
        if state is None:
            lines.append(f"| {trace.iteration} | n/a | n/a | n/a | |")
            continue
        delta = "n/a" if state.entropy_delta is None else f"{state.entropy_delta:.4f}"
        reason = state.stop_reason or ""
        lines.append(
            f"| {trace.iteration} | {state.spatial_entropy:.4f} | {delta} | "
            f"{state.normalized_temperature:.4f} | {reason} |"
        )
    return "\n".join(lines) + "\n"


def render_main_agent_status(result: RunResult) -> str:
    last = result.traces[-1] if result.traces else None
    lines = ["# DTE Main Agent Status", ""]
    lines.append(f"Problem: {result.spec.problem}")
    lines.append(f"Goal: {result.spec.goal}")
    lines.append("")
    lines.append(f"Embedding provider: `{result.spec.embedding_provider}`")
    lines.append(f"Embedding dimension: `{result.spec.embedding_dimension}`")
    lines.append(f"Total nodes: {len(result.nodes)}")
    lines.append(f"Frontier nodes: {sum(1 for n in result.nodes if n.status == 'frontier')}")
    lines.append(f"Closed nodes: {sum(1 for n in result.nodes if n.status == 'closed')}")
    lines.append(f"Merged nodes: {sum(1 for n in result.nodes if n.status == 'merged')}")
    if last and last.entropy_state:
        state = last.entropy_state
        lines.append("")
        lines.append("## Current search phase")
        lines.append(f"- spatial entropy: {state.spatial_entropy:.4f}")
        lines.append(f"- entropy delta: {'n/a' if state.entropy_delta is None else f'{state.entropy_delta:.4f}'}")
        lines.append(f"- normalized temperature: {state.normalized_temperature:.4f}")
        lines.append(f"- stop reason: {state.stop_reason or 'continue'}")
    lines.append("")
    lines.append("## Main-agent role")
    lines.append("The main agent should summarize frontier state, entropy/temperature, expansion allocation, and any human question in chat. It should not bypass DTE synthesis.")
    return "\n".join(lines) + "\n"


def render_human_questions_markdown(result: RunResult) -> str:
    questions = [trace.human_question for trace in result.traces if trace.human_question is not None]
    lines = ["# DTE Human Questions", ""]
    if not questions:
        lines.append("No human question was triggered in this run.")
        return "\n".join(lines) + "\n"
    for i, question in enumerate(questions, 1):
        lines.append(f"## {i}. {question.question_type}")
        lines.append(question.question)
        lines.append("")
        for option in question.options:
            lines.append(f"- {option}")
        lines.append("")
        lines.append(f"Context: {question.context}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_role_audit_markdown(result: RunResult) -> str:
    lines = ["# DTE Role Audit", ""]
    if not result.role_audit:
        lines.append("Initial nodes were supplied directly; role-isolated seed pipeline was not run.")
        return "\n".join(lines) + "\n"
    lines.append("This records the logical old-backend seed roles: decomposition, research, distillation, and strategy generation.")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.role_audit, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"
