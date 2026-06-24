---
name: dte-extreme-research
description: "Run the fixed Deep Think Evolving research protocol as a Codex skill/backend for high-depth mathematical, physical, academic, proof, derivation, or conceptual research. Invoke directly for slash-command style research tasks. The skill must enforce the DTE loop: DTERunSpec input, SearchNode frontier, Judge oracle, EvolutionController embedding/KDE/entropy/UCB allocation, executor expansion, relation-oracle classification, validated artifacts, and final DTE synthesis."
---

# DTE Extreme Research Skill

This file is the primary runtime instruction for slash-command usage. A Codex agent invoking this skill should not require the user to read separate docs before the first run.

## Slash-command intent

Use this skill when the user invokes a DTE-style research command, asks for extreme-depth research, wants a Codex-backed research agent, or wants a structured mathematical/physical/academic exploration rather than a direct one-shot answer.

Typical invocation:

```text
/dte-extreme-research <research task>
```

or:

```text
Use the dte-extreme-research skill on this task: <research task>
```

On invocation, Codex should immediately run the DTE workflow below unless the user explicitly asks only for explanation or planning.

## One-screen execution protocol

1. Read this `SKILL.md`, then `AGENTS.md`, then `CODEX_NEXT_STEPS.md` only for current blockers.
2. If not installed, run `python -m pip install -e .[dev]`.
3. Run `python scripts/smoke_workflow.py` once before serious use or after repo changes.
4. Convert the user task into a `DTERunSpec`.
5. Use `embedding_provider=gemini-embedding-2` and `embedding_dimension=3072` if `GEMINI_API_KEY` or `GOOGLE_API_KEY` is available. Otherwise use hash fallback only for dry-run/debug.
6. Always provide `--cache-path .dte_cache/cache.json`.
7. Run DTE through the backend. Prefer Judge oracle if available:

```bash
python -m dte_backend run \
  --spec examples/run_spec.json \
  --out-dir artifacts/session \
  --cache-path .dte_cache/cache.json \
  --judge-command "python examples/mock_judge_adapter.py"
```

8. Replace mock adapters with real Codex subagent calls when the environment supports them, but preserve the same JSON contracts.
9. Summarize these artifacts after each run:
   - `main_agent_status.md`
   - `frontier.md`
   - `entropy_trace.md`
   - `relation_candidates.md`
   - `human_questions.md`
   - `report.md`
10. If `human_questions.md` asks the user a branch question, ask it in chat instead of guessing.
11. Final answer must come from DTE synthesis, not directly from any subagent.

## Minimal smoke command

```bash
python scripts/smoke_workflow.py
```

This verifies spec guard, Judge oracle, Relation oracle, relation artifact conversion, DTE run with Judge command, and required artifact generation.

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

Never put these before `prompts/DTE_STATIC_PREFIX.md`:

- user task text;
- SearchNode content;
- retrieved documents;
- repo status;
- timestamps;
- git SHAs;
- file paths;
- stdout/stderr;
- tool logs;
- scratchpad summaries.

The helper for correct assembly is:

```python
from dte_backend.prompt_builder import build_cached_subagent_prompt
```

Use it when possible. If constructing prompts manually, preserve the same order.

## Backend semantic cache rule

This is separate from LLM prefix cache.

The backend caches embeddings and Judge scores using canonical context envelopes. The point is to avoid cache misses from unstable subagent formatting.

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

Implementation files:

```text
src/dte_backend/context_envelope.py
src/dte_backend/cache.py
src/dte_backend/file_cache.py
src/dte_backend/prompt_builder.py
```

## Required DTE flow

1. Generate or ingest initial SearchNodes.
2. Validate all SearchNodes against schema.
3. Score frontier nodes through a Judge Oracle. Judge returns observable score/reasoning/risks only.
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

After each run, inspect `relation_candidates.md`. If non-empty, Codex may call Relation Oracle on the listed pairs.

Standalone call:

```bash
python -m dte_backend relation-oracle \
  --nodes examples/frontier_nodes.json \
  --relation-command "python examples/mock_relation_adapter.py"
```

Convert relation result to machine artifacts:

```bash
python -m dte_backend relation-artifacts \
  --nodes examples/frontier_nodes.json \
  --relation-output examples/relation_result.json \
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
```

Available guard modes:

```bash
python hooks/dte_guard.py spec examples/run_spec.json
python hooks/dte_guard.py judge --nodes examples/frontier_nodes.json --output <judge_output.json>
python hooks/dte_guard.py relation --nodes examples/frontier_nodes.json --output <relation_output.json>
python hooks/dte_guard.py executor --parent <parent.json> --output <executor_output.json> --child-count <n>
```

If the Codex environment supports hooks, the next Codex implementation task is:

```text
Wire hooks/dte_guard.py into the Codex hook system so that:
1. spec guard runs before DTE backend execution;
2. Judge output guard runs after every Judge Oracle call;
3. Relation output guard runs after every Relation Oracle call;
4. Executor output guard runs after every Executor call;
5. failed guard stops the workflow before artifacts or graph state are consumed.
```

Until that hook wiring exists, the main agent must run guard commands manually at the same boundaries.

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
- reproducibility metadata: run spec, budget, embedding provider, Judge/Executor/Relation backends, cache path.

## Prohibited behavior

- Do not return a final answer directly from an executor episode.
- Do not silently skip Judge or EvolutionController.
- Do not replace role isolation with a single all-in-one agent.
- Do not modify UCB to be cost-aware by default.
- Do not rely on free-form Markdown as machine truth.
- Do not let executor adapters return synthesis nodes or pre-filled Judge/Evolution metrics.
- Do not treat Judge as an embedding model; Judge is a closed oracle that returns observable judgments only.
- Do not let relation-oracle output directly rewrite graph state before validation.
- Do not place dynamic content before `prompts/DTE_STATIC_PREFIX.md` in subagent prompts.
