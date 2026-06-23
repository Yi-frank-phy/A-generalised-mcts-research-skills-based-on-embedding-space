---
name: dte-extreme-research
description: "Run the fixed Deep Think Evolving research protocol as a Codex skill/backend for high-depth mathematical, physical, academic, proof, derivation, or conceptual research. Use when Codex must enforce the mandatory DTE loop: structured DTERunSpec input, SearchNode generation, Judge/scoring, EvolutionController UCB allocation, executor adapter expansion, and final DTE synthesis with assumptions, rejected alternatives, confidence levels, unresolved risks, and reproducibility metadata. Also use when developing or validating DTE executor adapters that must return structured SearchNode objects instead of direct final answers."
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

Use the bundled Python backend in this skill folder when a runnable local DTE
loop is needed. From the skill/backend root:

```bash
python -m pip install -e .[dev]
python -m dte_backend validate examples/run_spec.json
python -m dte_backend run --spec examples/run_spec.json --nodes examples/frontier_nodes.json --out-dir artifacts/prototype
```

Use `validate-executor` before trusting a Codex/Kimi/OpenClaw executor wrapper:

```bash
python -m dte_backend validate-executor --request examples/expansion_request.json --executor-command "python examples/echo_executor_adapter.py"
```

## Required flow

1. Generate or ingest initial SearchNodes.
2. Validate all SearchNodes against schema.
3. Score frontier nodes through Judge.
4. Compute density/uncertainty/UCB and allocate expansion budgets.
5. Expand selected frontier nodes.
6. Optionally run Codex/Kimi executor episodes through the executor adapter, but only inside the Executor role.
7. Validate returned node/evidence/counterexample objects.
8. Repeat within budget.
9. Produce final synthesis through DTE synthesis.

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
