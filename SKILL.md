---
name: dte-extreme-research
description: "Run the fixed Deep Think Evolving research protocol as a Codex skill/backend for high-depth mathematical, physical, academic, proof, derivation, or conceptual research. Invoke directly for slash-command style research tasks. The skill must enforce the DTE loop: DTERunSpec input, SearchNode frontier, real Judge oracle or explicit dry-run, EvolutionController embedding/KDE/entropy/UCB allocation, executor expansion, relation-oracle classification, validated artifacts, and final DTE synthesis."
---

# DTE Extreme Research Skill

This file is the primary runtime instruction for slash-command usage. A Codex agent invoking this skill should not require the user to read separate docs before the first run.

## Slash-command intent

Use this skill when the user invokes a DTE-style research command, asks for extreme-depth research, wants a Codex-backed research agent, or wants a structured mathematical/physical/academic exploration rather than a direct one-shot answer.

Typical invocation:

```text
/dte-extreme-research <research task>
```

On invocation, Codex should immediately run the DTE workflow below unless the user explicitly asks only for explanation or planning.

## Critical real-run rule

Mock adapters are smoke-test tools only. They are not research Judges, not research Relation Oracles, and not acceptable for real slash-command runs.

Forbidden in real research runs:

```bash
--judge-command "python examples/mock_judge_adapter.py"
--relation-command "python examples/mock_relation_adapter.py"
```

The mock adapters are blocked by default and only run when `DTE_ALLOW_MOCK_ADAPTER=1` is set by smoke tests. If a real Codex Judge/Relation/Executor subagent is unavailable, Codex must either:

1. run only a clearly labelled dry-run/smoke check; or
2. stop and tell the user that the real oracle layer is not available.

It must not present mock-oracle output as research judgment.

## One-screen execution protocol

1. Read this `SKILL.md`, then `AGENTS.md`, then `CODEX_NEXT_STEPS.md` only for current blockers.
2. If not installed, run `python -m pip install -e .[dev]`.
3. Run `python scripts/smoke_workflow.py` once before serious use or after repo changes. This is the only default place where mock adapters are allowed.
4. Convert the user task into a `DTERunSpec`.
5. Use `embedding_provider=gemini-embedding-2` and `embedding_dimension=3072` if `GEMINI_API_KEY` or `GOOGLE_API_KEY` is available. Otherwise use hash fallback only for dry-run/debug and label it as such.
6. Always provide `--cache-path .dte_cache/cache.json`.
7. For a real research run, use a real Codex Judge subagent command that follows `prompts/judge_oracle.md`. Do not use the mock Judge.
8. Summarize these artifacts after each run:
   - `main_agent_status.md`
   - `frontier.md`
   - `entropy_trace.md`
   - `relation_candidates.md`
   - `human_questions.md`
   - `report.md`
9. If `human_questions.md` asks the user a branch question, ask it in chat instead of guessing.
10. Final answer must come from DTE synthesis, not directly from any subagent.

## Smoke command

```bash
python scripts/smoke_workflow.py
```

This validates the protocol machinery. Smoke results are not research judgments.

## Real Gemini geometry

Use max geometry by default:

```json
"embedding_provider": "gemini-embedding-2",
"embedding_dimension": 3072
```

Manual check:

```bash
python scripts/gemini_smoke.py
```

Only run Gemini smoke when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set. Respect free-tier limits by embedding only node summaries and always using `--cache-path`.

## Prefix-cache rule for Codex backend LLM calls

LLM backend prefix caches require stable prompt prefixes. Do not optimize by making every subagent prompt a different shortest summary.

For every Judge, Executor, Relation, or future Synthesis subagent call, build the prompt in this exact order:

```text
prompts/DTE_STATIC_PREFIX.md
↓
role-specific prompt
↓
dynamic JSON input
```

Role-specific prompts:

```text
prompts/judge_oracle.md
prompts/executor_subagent.md
prompts/relation_oracle.md
```

Never put user task text, SearchNode content, retrieved documents, repo status, timestamps, git SHAs, file paths, stdout/stderr, tool logs, or scratchpad summaries before `prompts/DTE_STATIC_PREFIX.md`.

Use this helper when possible:

```python
from dte_backend.prompt_builder import build_cached_subagent_prompt
```

## Backend semantic cache rule

This is separate from LLM prefix cache. The backend caches embeddings and Judge scores using canonical context envelopes.

Preserve in SearchNodes:

- claim;
- explicit assumptions;
- evidence;
- counterexamples;
- risks;
- discriminator questions.

Do not include volatile material in semantic identity:

- parent ids;
- controller metrics;
- score;
- UCB;
- expansion budget;
- status;
- timestamps;
- logs;
- paths;
- transient summaries.

## Required DTE flow

1. Generate or ingest initial SearchNodes.
2. Validate all SearchNodes against schema.
3. Score frontier nodes through a real Judge Oracle, or explicitly declare dry-run mode.
4. EvolutionController computes embedding/KDE density, entropy, uncertainty, UCB, temperature, and expansion budgets.
5. Executor expands selected frontier nodes into structured child SearchNodes.
6. Validate executor output before adding it to the graph.
7. Render `relation_candidates.md`.
8. Relation Oracle classifies only selected candidate pairs, not every pair.
9. Convert validated relation output into `MergeProposal` or discriminator task through backend helpers.
10. Produce final synthesis through DTE synthesis.

## Oracle definitions

Oracle means a bounded external/subagent judgment function. It is not the controller.

- Judge Oracle: `SearchNodes -> score/reasoning/risks`.
- Relation Oracle: `SearchNodes -> equivalent|complementary|conflict|independent`.
- Executor: `ExpansionRequest -> child SearchNodes`.

Oracles must return machine-readable JSON and must pass validators.

## Relation oracle workflow

After each run, inspect `relation_candidates.md`. If non-empty, Codex may call a real Relation Oracle on the listed pairs. Do not use the mock Relation adapter for research judgment.

Convert validated relation result to machine artifacts:

```bash
python -m dte_backend relation-artifacts \
  --nodes <nodes.json> \
  --relation-output <relation_output.json> \
  --out-dir artifacts/relation
```

Outputs:

```text
relation_proposals.json
discriminator_tasks.json
```

Relation Oracle output must not directly mutate the graph.

## Hook status and what Codex must implement next

The repository already has guard logic:

```text
hooks/dte_guard.py
hooks/README.md
HOOK_WIRING_TODO.md
```

Available guard modes:

```bash
python hooks/dte_guard.py spec examples/run_spec.json
python hooks/dte_guard.py judge --nodes examples/frontier_nodes.json --output <judge_output.json>
python hooks/dte_guard.py relation --nodes examples/frontier_nodes.json --output <relation_output.json>
python hooks/dte_guard.py executor --parent <parent.json> --output <executor_output.json> --child-count <n>
```

If the Codex environment supports hooks, wire these guards into the hook system. Until hook wiring exists, the main agent must run guard commands manually at the same boundaries.

## Compile behavior

Compile is not a mandatory Distiller phase. Codex or subagents may compile their own local context when useful, but compilation must happen after the stable prompt prefix and must preserve assumptions, evidence, counterexamples, branch conflicts, and uncertainty.

Do not restore a mandatory Distiller role.

## Output contract

The final answer/report must include:

- answer or research report;
- selected search path;
- key rejected alternatives;
- assumptions;
- confidence levels;
- unresolved risks;
- reproducibility metadata: run spec, budget, embedding provider, Judge/Executor/Relation backends, cache path, and whether any dry-run fallback was used.

## Prohibited behavior

- Do not use mock adapters for real research judgment.
- Do not return a final answer directly from an executor episode.
- Do not silently skip Judge or EvolutionController.
- Do not replace role isolation with a single all-in-one agent.
- Do not modify UCB to be cost-aware by default.
- Do not rely on free-form Markdown as machine truth.
- Do not let executor adapters return synthesis nodes or pre-filled Judge/Evolution metrics.
- Do not treat Judge as an embedding model; Judge is a closed oracle that returns observable judgments only.
- Do not let relation-oracle output directly rewrite graph state before validation.
- Do not place dynamic content before `prompts/DTE_STATIC_PREFIX.md` in subagent prompts.
