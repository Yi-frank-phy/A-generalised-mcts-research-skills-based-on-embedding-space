---
name: dte-extreme-research
description: "Run the fixed Deep Think Evolving research protocol as a Codex skill/backend for high-depth mathematical, physical, academic, proof, derivation, or conceptual research. Use when Codex must enforce the mandatory DTE loop: structured DTERunSpec input, SearchNode generation, Judge/scoring, EvolutionController UCB allocation, executor adapter expansion, relation-oracle merge classification, and final DTE synthesis with assumptions, rejected alternatives, confidence levels, unresolved risks, and reproducibility metadata. Also use when developing or validating DTE executor adapters that must return structured SearchNode objects instead of direct final answers."
---

# DTE Extreme Research Skill

## Purpose

Use this skill to run a fixed Deep Think Evolving research protocol for high-depth mathematical, physical, academic, or conceptual research.

The skill exists to package DTE as a backend protocol for Codex/Kimi/OpenClaw-style agents. It is not a general brainstorming prompt.

## Inputs

The skill accepts a structured `DTERunSpec`:

```json
{
  "problem": "research problem",
  "goal": "desired output",
  "constraints": ["must preserve derivations", "mark uncertainty"],
  "embedding_provider": "gemini-embedding-2",
  "embedding_dimension": 3072,
  "budget": {
    "max_iterations": 2,
    "total_child_budget": 3,
    "max_research_iterations": 1
  },
  "mode": "mandatory_frontier",
  "allow_self_organized_executor": true,
  "require_final_synthesis": true
}
```

## Bundled backend

Use the bundled Python backend in this skill folder when a runnable local DTE loop is needed. From the skill/backend root:

```bash
python -m pip install -e .[dev]
python -m dte_backend validate examples/run_spec.json
python -m dte_backend run --spec examples/run_spec.json --out-dir artifacts/prototype --cache-path .dte_cache/cache.json
```

## Required flow

1. Generate or ingest initial SearchNodes.
2. Validate all SearchNodes against schema.
3. Score frontier nodes through a Judge oracle. This can be a subagent, but it must return observable scores/reasoning only.
4. Compute embedding/KDE density, entropy, uncertainty, UCB, and expansion budgets.
5. Expand selected frontier nodes.
6. Optionally run Codex/Kimi executor episodes through the executor adapter, but only inside the Executor role.
7. Validate returned node/evidence/counterexample objects.
8. Call relation-oracle subagents when frontier nodes are semantically close, branches conflict, complementary routes should be compressed, or entropy plateaus.
9. Repeat within budget.
10. Produce final synthesis through DTE synthesis.

## Relation oracle workflow

After expansion, send only the relevant frontier nodes to the relation oracle.
Validate its result before graph effects. A validated `equivalent`,
`complementary`, or `conflict` result may become a `MergeProposal` or a
discriminator task; raw subagent output must not mutate the graph directly. An
`independent` result keeps both branches in the frontier.

## Compile behavior

Compile is not a mandatory Distiller phase. Codex or subagents may compile their own local context when useful. Compilation should preserve assumptions, evidence, counterexamples, branch conflicts, and uncertainty, while dropping irrelevant logs and transcript bulk.

## Output contract

The final output must include:

- answer/report;
- selected search path;
- key rejected alternatives;
- assumptions;
- confidence levels;
- unresolved risks;
- reproducibility metadata: run spec, budget, model/backend choices.

## Prohibited behavior

- Do not return a final answer directly from an executor episode.
- Do not silently skip Judge or EvolutionController.
- Do not replace role isolation with a single all-in-one agent.
- Do not modify UCB to be cost-aware by default.
- Do not rely on free-form Markdown as machine truth.
- Do not let executor adapters return synthesis nodes or pre-filled Judge/Evolution metrics.
- Do not treat Judge as an embedding model; Judge is a closed oracle that returns observable judgments only.
- Do not let relation-oracle output directly rewrite graph state before validation.
