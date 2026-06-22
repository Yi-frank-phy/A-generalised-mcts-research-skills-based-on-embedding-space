# Technical Specification

## 1. Core abstraction

DTE is a frontier-based beam graph search system with evolutionary budget allocation.

Let the frontier at step `t` be:

```text
F_t = {v_i | v_i is a currently expandable leaf/search node}
```

Each node has:

- claim / hypothesis;
- rationale;
- assumptions;
- evidence;
- risks;
- score;
- embedding or feature representation;
- parent IDs;
- status.

The controller evaluates frontier nodes and allocates expansion budget.

## 2. UCB objective

Default UCB remains value/uncertainty driven:

```text
U_i = V_i + c * tau * uncertainty_i
```

where:

- `V_i`: Judge value score;
- `c`: exploration coefficient;
- `tau`: normalized system temperature;
- `uncertainty_i`: density-derived or novelty-derived uncertainty proxy.

Cost is not part of UCB by default. Cost is controlled by hard caps:

- `max_iterations`;
- `total_child_budget`;
- `max_research_iterations`;
- backend/model policy;
- early synthesis.

An experimental cost-aware profile may be added later, but must not be default.

## 3. Boltzmann allocation

Given allocation values `A_i`, temperature `T`, and total expansion budget `C`:

By default the prototype uses `A_i = U_i`, so entropy/uncertainty affects actual expansion rather than merely display ranking. Use score-only allocation only as a compatibility profile.

```text
p_i = exp(A_i / T) / Σ_j exp(A_j / T)
k_i = round_or_ceil(C * p_i)
```

`k_i` is the expansion budget for node `i`.

## 4. Status model

Recommended internal statuses:

```text
frontier       currently expandable leaf node
closed         expanded/internal node
archived       preserved but not in active frontier
merged         absorbed into another node
synthesis      graph-compression/synthesis node
```

Legacy compatibility can map:

```text
active   -> frontier
expanded -> closed
child_quota -> expansion_budget
```

## 5. Mandatory phases

### Phase A: Seed
Generate or ingest initial SearchNodes.

### Phase B: Judge
Score SearchNodes. The Judge does not directly allocate budget.

### Phase C: EvolutionController
Compute density/uncertainty/UCB and expansion budget.

### Phase D: Executor
Run expansion or research episodes. External agents may be used here.

### Phase E: Merge/Synthesis
Compress graph state into a report or synthesis node.

## 6. External agent boundary

Codex/Kimi/OpenClaw may run inside Executor. They cannot bypass DTE.

The backend exposes this boundary as an executor adapter invoked only by the
Expansion phase after Judge and EvolutionController have assigned an expansion
budget. The adapter receives one parent SearchNode, the allocated child count,
the iteration number, and optionally the validated DTERunSpec. It must not run
Judge, allocate budget, or produce final synthesis.

This input is represented as a validated `ExpansionRequest` object:

```json
{
  "parent": {"node_id": "...", "node_type": "candidate", "claim": "..."},
  "count": 1,
  "iteration": 1,
  "spec": null
}
```

Like `SearchNode` and `DTERunSpec`, `ExpansionRequest` rejects extra fields.

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
- returned children cannot pre-fill `score`, `uncertainty`, `ucb_score`, or
  `expansion_budget`;
- returned child count cannot exceed the allocated budget.

## 7. Performance requirements

- Cache embeddings by stable node text hash.
- Cache Judge scores when node content and trajectory have not changed.
- Batch multiple node evaluations where feasible.
- Avoid injecting full graph context into every role.
- Prefer node summaries and deltas.
