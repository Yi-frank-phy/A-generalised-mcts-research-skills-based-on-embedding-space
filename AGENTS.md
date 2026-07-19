# AGENTS.md — DTE Codex Skill Backend

> **Read first:** [`SPEC.md`](./SPEC.md) and [`ARCHITECTURE.md`](./ARCHITECTURE.md) define the current protocol and architecture. GitHub issue #2 tracks the active implementation plan.

This repository implements a fixed DTE research protocol. When acting as an agent in this repo, you must treat DTE as the controlling architecture, not as optional advice.

## Non-negotiable protocol invariants

1. **DTE is the only outer controller.** A model-facing root agent may start and supervise the backend and may issue privileged controller commands when `OperatorPolicy` authorizes them. It must not directly advance the state machine, allocate, mutate controller-owned state, claim convergence, or commit synthesis. A final user report must follow the backend-controlled node generation → Judge/scoring → allocation/expansion → Relation/readiness protocol and be grounded in the terminal provisional selection and handoff.
2. **Preserve role separation.** Strategy generation, judging, execution, relation classification, and synthesis are logically separate roles, even if implemented in fewer physical model calls or subagents.
3. **Executor is not the final authority.** Codex/Kimi/OpenClaw may perform local research episodes, write code, run tests, or draft candidate reasoning, but must return structured SearchNode objects.
4. **No direct final answer from subagents.** A self-organized executor episode may produce evidence, counterexamples, candidate nodes, Judge outputs, or merge relation outputs, but only the main agent may report after the backend terminal handoff. The current App slice does not manufacture a final Synthesis episode.
5. **UCB is not cost-aware by default.** Exploration is stabilized by UCB/uncertainty and hard budget caps. Do not silently change the objective to penalize cost unless the user explicitly chooses an experimental profile.
6. **Budget limits are hard.** Never increase `max_iterations`, `allocation_mass_per_iteration`, `max_children_per_iteration`, `max_relation_enrichment_pairs`, or backend model strength without explicit user instruction.
7. **Schema is source of truth.** Free-form Markdown or natural language cannot override the JSON/Pydantic run spec.
8. **Use max geometry by default.** For real embedding geometry, prefer `embedding_dimension=3072`; lower dimensions are debug/fallback profiles.
9. **Observation alone is not authority.** A model-facing agent may read and summarize checkpoints. Authority comes from user delegation plus validated `OperatorPolicy` and is exercised only through backend-validated controller commands, never direct graph/state mutation.
10. **Codex App drives native episodes in place.** In an App/Work task, use `create-run` / `next-episode` / native App work / `submit-episode-result`. Do not launch another Codex process to simulate the current App runtime. Opaque internal subagent topology is allowed and is not a backend correctness input.

## Codex App native driver loop

When the current Codex App main agent runs DTE research:

1. Start or resume the persistent backend run.
2. Call `next-episode`; do not manually select the global branch, Judge batch, or child grant. The backend consumes deterministic embedding/entropy/UCB/allocation transitions internally.
3. Read the complete versioned `EpisodeRequest` and its `attempt_id`.
4. Perform only that bounded episode with current App reasoning, tools, and optional native subagents.
5. Construct one complete strict `EpisodeResult`; progress chat, Markdown, files, and subagent summaries are not committed results.
6. Call `submit-episode-result` and inspect `CommitOutcome` plus the backend controller action.
7. Repeat only when the backend requests another episode; use explicit fail/cancel/retry transitions when needed.

For a sufficiently complex episode, use native subagents and tools when they
materially improve coverage. Decompose according to the problem rather than a
fixed worker count or topology, parallelize only independent work, avoid
duplicating the same route across internal branches, reconcile disagreements,
and submit one integrated result that satisfies the current role schema. The
internal decomposition creates no graph, score, allocation, stopping, merge, or
synthesis authority.

Keep the logical goals distinct even when the same App runtime performs them:

- Judge independently evaluates research potential, risks, and uncertainty; it
  does not expand a new research node.
- Executor develops only the backend-granted branch; it may explore independent
  approaches internally, but it does not score or allocate them.
- Relation compares only the granted pairs; it is not a verifier and does not
  choose a scientific winner.

For a Judge request, return only the granted nodes' observable scores, reasoning, risks, and optional uncertainty evidence. Never hand-fill score into graph state, embedding, density, entropy, uncertainty, UCB, allocation, graph/node revision, stopping, or synthesis fields. Never bypass submission validation or treat hidden agent count, names, routing, traces, tokens, or quota as required graph facts.

Executor and Judge outputs may include the optional bounded
`epistemic_contributions` object, but only when there is material information to
preserve. Do not manufacture generic assumptions to fill the schema. Prefer
critical dependencies, substantive support or challenge evidence, explicit
counterexamples, transferable failure modes, and unresolved questions. Use
machine identities for internal facts; give external evidence an `external:` or
safe `artifact:` basis reference. An Executor may target only its granted parent
and children created by the same accepted output; a Judge may target only its
granted nodes. Neither role may submit `human_confirmed` or `backend_derived`
facts, infer formal edges from free text, equate a low score or non-selection
with contradiction, or claim that the backend verified scientific truth.

For a Relation request, inspect only `relation_payload.candidate_pairs`. One Relation episode contains only node-disjoint candidate pairs: each node ID may appear in at most one granted pair. Classify every granted pair exactly once as `equivalent`, `complementary`, `conflict`, or `independent`; use only granted evidence references; construct one strict `RelationEpisodeResult`; submit it; then inspect `CommitOutcome`. Do not scan the graph for extra pairs, select a canonical node, merge or close nodes, edit the candidate/Relation ledger, set synthesis readiness, write `ready_for_synthesis`, or return correctness/pass/fail/reward state. Node-disjointness is a transactional merge-safety invariant, not a verification rule. Relation is not a second Judge, verifier, or final Synthesis agent. Discriminator proposals are persisted metadata and are not executed in the current workflow.

Relation does not receive a second epistemic output schema. Epistemic summaries
project equivalent/complementary/conflict/independent records, disclosure
obligations, and merge provenance from the authoritative Relation ledger.

## Terminal observability, epistemic handoff, and feedback

When `next-episode` returns `ready_for_synthesis` or `run_complete`, first run:

```bash
python -m dte_backend observability-summary --run-dir <run-dir> --format json
python -m dte_backend epistemic-summary --run-dir <run-dir> --format json
```

Read both formal summaries together with provisional selection and Relation
disclosures. The epistemic handoff describes provisional-selected node claims;
it does not claim that the backend audited the main agent's final prose, and the
current App slice does not require a final Synthesis episode. Report the result
with: the current most credible conclusion; its key assumptions; supporting,
challenging, and conditional evidence; the most dangerous unverified
assumptions and unresolved dependencies; whether an important abandoned route
was merely not searched/low priority or has an explicit counterexample; a short
correlated-error risk note; and possible transferable heuristics. Preserve every
source label, especially the distinction between `agent_reported`,
`external_artifact_backed`, and `human_confirmed`.

Also give a compact operational summary covering Judge/Executor/Relation counts,
initial → committed → selected nodes, major allocations and committed children,
merge/conflict outcomes, rejections/retries, terminal reason, and data-quality
limitations. Do not hand-calculate formal metrics from raw JSONL. Internal
allocation, survival, merge, conflict, latency, and retry measures are process
proxies; they do not prove scientific correctness or that DTE is effective.

When the user explicitly evaluates a run or a concrete decision, the main agent
may append source-labelled feedback with `record-feedback`. It must preserve the
user's source as `user`, label its own assessment as `main_agent`, and never infer
positive feedback from silence or mere acceptance. Feedback is not Judge input
and cannot modify graph or controller state.

Potential learning inferred by the main agent is not user learning. Record it,
if useful, as `source=main_agent` with `user_confirmed=false`. Only after the
user explicitly states that their view changed or confirms a reusable research
lesson may the main agent append a new confirmation using
`record-learning --source user`; never infer learning from silence, continued
conversation, or acceptance of an answer. Learning records are append-only and
do not modify epistemic edges, Relation, Judge, UCB, allocation, or stopping.

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
In the current App slice, this is the main agent's terminal reporting step over
backend-controlled provisional selection plus operational and epistemic
handoffs, not a native episode. It must preserve uncertainty and failure modes.

## When editing this repo

Before changing architecture-level files, update `SPEC.md` and tests. Do not introduce a new framework dependency unless it replaces a real repeated pain point.
