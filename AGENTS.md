# AGENTS.md — DTE Codex Skill Backend

> **Read first:** [`SPEC.md`](./SPEC.md) and [`ARCHITECTURE.md`](./ARCHITECTURE.md) define the current protocol and architecture. GitHub issue #2 tracks the active implementation plan.

This repository implements a fixed DTE research protocol. When acting as an agent in this repo, you must treat DTE as the controlling architecture, not as optional advice.

## Non-negotiable protocol invariants

1. **DTE is the only outer controller.** A model-facing root agent must not decide whether to run DTE, advance its state machine, allocate, stop, or commit synthesis. Final research conclusions must pass through the backend-controlled protocol: node generation → structured node output → Judge/scoring → allocation/expansion → synthesis.
2. **Preserve role separation.** Strategy generation, judging, execution, relation classification, and synthesis are logically separate roles, even if implemented in fewer physical model calls or subagents.
3. **Executor is not the final authority.** Codex/Kimi/OpenClaw may perform local research episodes, write code, run tests, or draft candidate reasoning, but must return structured SearchNode objects.
4. **No direct final answer from subagents.** A self-organized executor episode may produce evidence, counterexamples, candidate nodes, Judge outputs, or merge relation outputs, but the final answer must be created by DTE synthesis.
5. **UCB is not cost-aware by default.** Exploration is stabilized by UCB/uncertainty and hard budget caps. Do not silently change the objective to penalize cost unless the user explicitly chooses an experimental profile.
6. **Budget limits are hard.** Never increase `max_iterations`, `allocation_mass_per_iteration`, `max_children_per_iteration`, or backend model strength without explicit user instruction.
7. **Schema is source of truth.** Free-form Markdown or natural language cannot override the JSON/Pydantic run spec.
8. **Use max geometry by default.** For real embedding geometry, prefer `embedding_dimension=3072`; lower dimensions are debug/fallback profiles.
9. **Observation is not authority.** A model-facing agent may read and summarize checkpoints or recommend that the user interrupt, but it must not write a control request or treat artifact access as state-machine permission.

## Preferred implementation style

- Keep code simple and explicit.
- Avoid over-engineered abstractions.
- Prefer small Python modules over deep frameworks.
- Use JSON schemas and Pydantic models for all machine-facing boundaries.
- Add comments explaining shape, purpose, and assumptions.

## Role contracts

### StrategyGenerator
Produces multiple mutually distinct hypothesis/search nodes. It must not rank its own candidates.

### Judge Oracle
Scores candidates according to logical coherence, assumption strength, evidence, and constraint compliance. It may be implemented by a strong subagent. It returns observable scores/reasoning only; it does not provide hidden vectors and does not allocate budget.

### EvolutionController
Computes embeddings/density/uncertainty/UCB and allocates expansion budgets. It is deterministic or mostly deterministic Python code.

### Executor
Runs local research/coding/proof episodes. It must submit structured outputs, not direct final conclusions.

### Merge / Relation Oracle
Classifies pairs or sets of nodes as equivalent, complementary, conflicting, or independent. It may be implemented by a subagent and may propose discriminator questions. It does not directly synthesize the final answer.

### Compile
Compile is not a mandatory backend role. Codex/subagents may compile local context when useful, but compilation is a prompt-level compression operation, not a separate required DTE phase.

### Synthesis
Compresses graph state into a report after DTE-controlled selection. It must preserve uncertainty and failure modes.

## When editing this repo

Before changing architecture-level files, update `SPEC.md` and tests. Do not introduce a new framework dependency unless it replaces a real repeated pain point.
