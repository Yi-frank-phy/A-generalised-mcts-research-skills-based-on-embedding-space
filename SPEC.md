# Technical Specification

## 1. Core abstraction

DTE is a frontier-based beam graph search system with evolutionary budget allocation.

Let the frontier at step `t` be:

```text
F_t = {v_i | v_i is a currently expandable leaf/search node}
```

Each node has claim/hypothesis, rationale, assumptions, evidence, risks, score, embedding representation, parent IDs, and status. The controller evaluates frontier nodes and allocates expansion budget.

## 2. UCB objective

Default UCB remains value/uncertainty driven:

```text
U_i = V_i + c * tau * uncertainty_i
```

where `V_i` is Judge value, `tau` is normalized temperature, and `uncertainty_i` comes from embedding-space density/KDE. Cost is not part of UCB by default. Cost is controlled by hard caps: `max_iterations`, `total_child_budget`, `max_research_iterations`, backend/model policy, and synthesis triggers.

## 3. Geometry and embedding dimension

DTE's entropy controller requires a continuous embedding geometry:

```text
x_i = E(v_i)
rho_i = KDE(x_i)
S_t = - mean(log rho_i)
```

For real runs, geometry should be max-quality by default. `embedding_dimension` defaults to `3072`; lower dimensions are debug/fallback profiles. Hash embeddings are only for offline tests and CI.

## 4. Boltzmann allocation

Given allocation values `A_i`, temperature `T`, and total expansion budget `C`:

```text
p_i = exp(A_i / T) / Σ_j exp(A_j / T)
k_i = round_or_ceil(C * p_i)
```

By default `A_i = U_i`, so entropy/uncertainty affects actual expansion rather than merely display ranking.

## 5. Status model

```text
frontier       currently expandable leaf node
closed         expanded/internal node
archived       preserved but not in active frontier
merged         absorbed into another node
synthesis      graph-compression/synthesis node
```

Legacy compatibility can map `active -> frontier`, `expanded -> closed`, and `child_quota -> expansion_budget`.

## 6. Mandatory phases

### Phase A: Seed
Generate or ingest initial SearchNodes. The seed pipeline keeps decomposition/research/strategy generation logically separate. The old mandatory Distiller role is removed; compile is an optional prompt-level operation available to each agent/subagent.

### Phase B: Judge Oracle
Score SearchNodes. The Judge may be implemented by a strong subagent. It returns observable scores, reasons, and risk notes. It does not expose hidden vectors, allocate budget, or synthesize the final answer.

### Phase C: EvolutionController
Compute embedding/KDE density, entropy, uncertainty, temperature, UCB, and expansion budget. This is the deterministic mathematical controller.

### Phase D: Executor
Run expansion or research episodes. External agents may be used here but must return structured SearchNode children.

### Phase E: Relation/Merge Oracle
Equivalent, complementary, and conflict merge judgments are callable oracle tasks. A model/subagent may classify relations or propose discriminator questions, but final graph mutation must pass backend validation.

### Phase F: Synthesis
Compress graph state into a report or synthesis node after DTE-controlled selection.

## 7. External agent boundary

Codex/Kimi/OpenClaw may run inside Executor. They cannot bypass DTE.

The backend exposes this boundary as an executor adapter invoked only after Judge and EvolutionController have assigned an expansion budget. The adapter receives one parent SearchNode, the allocated child count, the iteration number, and optionally the validated DTERunSpec. It must not allocate budget or produce final synthesis.

Required executor output:

```json
{
  "node_type": "candidate",
  "claim": "...",
  "rationale": "...",
  "assumptions": [],
  "evidence": [],
  "risks": [],
  "parent_ids": [],
  "confidence": 0.0
}
```

Adapter outputs are validated before being appended to the graph:

- returned children must include the expanded parent id;
- returned children must have `status = "frontier"`;
- returned children cannot be `synthesis` nodes;
- returned children cannot pre-fill `score`, `uncertainty`, `ucb_score`, or `expansion_budget`;
- returned child count cannot exceed the allocated budget.

## 8. Oracle task boundary

Judge and relation tasks are not hard-coded backend intelligence. They are observable oracle tasks:

```text
JudgeOracle: nodes -> scores/reasoning/risks
RelationOracle: nodes -> equivalent|complementary|conflict|independent + rationale
DiscriminatorOracle: conflicting nodes -> discriminator question
```

These tasks may be implemented by Codex subagents. They do not provide latent token vectors and do not replace embedding geometry.

## 9. Performance requirements

- Cache embeddings by stable node text hash.
- Cache Judge scores when node content has not changed.
- Use a persistent file cache when using Gemini Embedding 2 or other high-quality providers.
- Batch multiple node evaluations where feasible.
- Avoid injecting full graph context into every role.
- Prefer node summaries and deltas.

## 10. Merge skeleton

The deterministic backend implements conservative `equivalent_merge` for exact normalized-claim duplicates. Complementary/conflict merge is represented as a relation oracle task and can be delegated to a strong subagent. Merge may compress the graph, but final conclusions still require DTE synthesis.

Relation oracle workflow:

1. Call the relation oracle after expansion when new frontier nodes are close in embedding space, when branches appear to conflict, when complementary claims should be compressed, or when entropy plateaus without a clear synthesis path.
2. Pass only the relevant candidate nodes and the relation task contract. The oracle returns `equivalent`, `complementary`, `conflict`, or `independent`, plus rationale and an optional discriminator question.
3. Validate the returned object with `validate_relation_output()` or `python hooks/dte_guard.py relation ...` before any graph effect.
4. Convert a validated `equivalent`, `complementary`, or `conflict` result into a `MergeProposal` or a discriminator task. Do not let raw oracle output mutate graph state directly.
5. For `independent`, preserve both branches and continue normal Judge/EvolutionController allocation.
