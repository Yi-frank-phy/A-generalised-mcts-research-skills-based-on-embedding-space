# Architecture Decision: DTE-as-a-Skill/Backend

## Decision

Package the existing DTE architecture as a skill-backed backend for Codex-level extreme research.

Do not redesign the DTE architecture per task.

## Rationale

DTE's advantage is not ordinary multi-agent scaffolding. Its advantage is the information-theoretic/evolutionary selection layer:

```text
Judge value → density/novelty → UCB → Boltzmann expansion → synthesis
```

This layer requires evaluation calls. Skipping evaluation removes the main advantage. Therefore optimization should focus on batching, caching, and role output compression, not removing the search engine.

## Key distinction

```text
Logical role separation != physical model-call separation
```

Roles remain separate to reduce bias. Calls can be batched or macro-stepped to reduce IO.

## Kimi Agent Swarm lessons to absorb

Learn:

- isolated subagent contexts;
- structured summaries returned to orchestrator;
- parallel execution at the frontier;
- critical-path reduction;
- coach/player separation.

Do not copy:

- fully role-free orchestration;
- direct final answers from subagents;
- massive unbounded horizontal expansion.

## Final architecture

```text
DTE Skill
  ↓
Fixed role protocol
  ↓
Python math backend
  ↓
Executor adapters: Codex / Kimi / OpenClaw / LangChain
  ↓
structured node outputs
  ↓
DTE synthesis
```
