# Technical Specification

## 1. Core abstraction

DTE is a frontier-based epistemic graph-search system for research problems whose true value is not directly observable through a stable, deterministic reward.

Let the frontier at step `t` be:

```text
F_t = {v_i | v_i is a currently expandable leaf/search node}
```

Each node records a claim or hypothesis, rationale, assumptions, evidence, risks, parent revisions, and controller-owned metrics. DTE searches over research strategies, explanations, conjectures, counterexamples, and formalizations. It is not primarily a program optimizer and must not be reduced to an AlphaEvolve-style scalar-reward loop.

The DTE backend is the only authority allowed to:

- own graph state and graph revision;
- compute embeddings, density, uncertainty, entropy, UCB, and allocation;
- accept or reject episode outputs;
- apply merges and relation results;
- decide whether another iteration or synthesis is permitted;
- select the graph checkpoint used for final synthesis.

External model runtimes may perform bounded research episodes, including native self-organized subagent work, but they cannot advance the DTE state machine directly.

## 2. UCB objective

Default UCB remains value/uncertainty driven:

```text
U_i = V_i + c * tau * uncertainty_i
```

where:

- `V_i` is an observable Judge estimate of research potential;
- `tau` is normalized temperature;
- `uncertainty_i` comes from embedding-space density/KDE or another explicitly configured novelty estimate.

Cost is not part of UCB by default. Compute and quota limits are enforced outside the objective through run policy, episode policy, and hard caps.

Judge value is not a correctness proof. It estimates whether a branch is coherent, informative, tractable, and worth further investigation under incomplete evidence.

## 3. Geometry and embedding dimension

DTE's entropy controller requires a continuous embedding geometry:

```text
x_i = E(v_i)
rho_i = KDE(x_i)
S_t = -mean(log rho_i)
```

For real runs, geometry should use the highest-quality configured embedding profile by default. `embedding_dimension` currently defaults to `3072`; lower dimensions are debug/fallback profiles. Hash embeddings are only for offline tests and CI.

Embedding cache identity must include at least:

```text
content_hash
embedding_provider
embedding_model_or_snapshot
embedding_dimension
embedding_contract_version
```

## 4. Boltzmann allocation and budget semantics

Given allocation values `A_i`, temperature `T`, and a per-iteration allocation mass `C`:

```text
p_i = exp(A_i / T) / sum_j exp(A_j / T)
q_i = C * p_i
```

`q_i` is a continuous expansion mass, not a conserved integer child count.

The intended prototype discretization is:

```text
if q_i < 1:
    tentative_i = round_half_up(q_i)
else:
    tentative_i = ceil(q_i)
```

Therefore the realized child count may exceed `C`. For example:

```text
q = [0.7, 0.6, 0.3, 1.4]
tentative = [1, 1, 0, 2]
```

This is valid even though the realized count is `4` while the soft allocation mass is `3`.

A separate hard per-iteration cap `H` limits actual graph expansion:

```text
sum_i children_i <= H
```

The target default semantics are:

```text
allocation_mass_per_iteration = 3
max_children_per_iteration = 5
```

For temporary input compatibility, the legacy field `total_child_budget` is accepted only as a deprecated alias for `allocation_mass_per_iteration`. Canonical serialization and schemas use the new fields.

If tentative allocation exceeds `H`, the controller must trim children by a deterministic marginal-priority rule derived from allocation mass and node priority. It must not trim by input order.

Python's built-in bankers rounding is not the normative rule. Half values below one use round-half-up semantics.

By default `A_i = U_i`, so entropy and uncertainty affect actual expansion rather than merely display ranking.

## 5. Status and ownership model

```text
frontier       currently expandable leaf node
closed         expanded/internal node
archived       preserved but not in active frontier
merged         absorbed into another node
synthesis      graph-compression/synthesis node
```

Legacy compatibility can map `active -> frontier`, `expanded -> closed`, and `child_quota -> expansion_budget`.

Controller-owned fields include:

```text
embedding
score
judge verdict references
uncertainty
ucb_score
expansion_budget
graph status
graph revision
```

Seed or Executor episodes must not pre-fill these fields.

## 6. Logical phases

Logical role separation does not imply one physical model call per role.

### Phase A: Seed

Generate or ingest initial SearchNodes. A real run may use one bounded native Seed Episode that internally self-organizes exploration and returns several materially distinct, unranked SearchNodes.

A mandatory physical Explorer agent is not required. The fixed direct/counterexample/formalism/merge templates remain valid as smoke-test or fallback seed generation.

Seed output must:

- contain materially distinct branches rather than paraphrases;
- preserve conflicts and uncertainty;
- include counterexample or boundary-case directions when meaningful;
- avoid ranking or selecting a winner;
- avoid producing final synthesis.

### Phase B: Judge Oracle

Score SearchNodes. The Judge may be implemented by a strong model episode. It returns observable scores, reasons, evidence gaps, and risk notes. It does not expose hidden vectors, allocate budget, create children, or synthesize the final answer.

Judge context must be logically isolated from the Executor episode being judged. Physical implementation may reuse the same base model, but it must use a separate role contract and bounded input context.

### Phase C: EvolutionController

Compute embedding/KDE density, entropy, uncertainty, temperature, UCB, and expansion allocation. This is deterministic or mostly deterministic backend code and is the unique allocation authority.

### Phase D: Executor Episode

Run a bounded research, coding, proof, or analysis episode for one assigned parent revision.

The model runtime may use native self-organized subagents internally. DTE does not prescribe the physical subagent topology. The episode must return only the child SearchNodes allowed by its request.

DTE controls recursion across graph iterations. The model runtime may control bounded horizontal parallelism inside one episode.

### Phase E: Relation/Merge Oracle

Relation is a callable semantic graph-maintenance oracle. It classifies selected node pairs or sets as:

```text
equivalent
complementary
conflict
independent
```

It may propose a discriminator question. Raw Relation output cannot mutate the graph.

Relation is not a mandatory blocking step for every candidate pair. Recommended policy:

- exact deterministic duplicates may be handled immediately;
- embedding-close or near-tied branches create optional/high-priority Relation tasks;
- entropy plateau increases Relation priority;
- unresolved material conflicts among branches selected for synthesis must either be resolved or explicitly disclosed;
- the mere existence of a Relation candidate must not automatically forbid synthesis.

### Phase F: Synthesis

Compress a DTE-selected graph checkpoint into a report or synthesis node. Synthesis reads validated graph state and recorded evidence. It must not continue open-ended research or silently fill unresolved verification gaps.

## 7. AgentEpisode boundary

Codex, Ultra, Kimi, OpenClaw, or another model runtime may operate inside a bounded `AgentEpisode`. They cannot be the DTE controller.

The normative boundary is transport-neutral:

```text
AgentEpisodeAdapter:
    EpisodeRequest -> EpisodeResult
```

CLI subprocesses, Codex SDK, Codex App Server, hosted runtimes, or future transports may implement the same interface.

An `EpisodeRequest` should contain at least:

```text
episode_id
role
input_graph_revision
parent_node_revision or selected node revisions
max_returned_children
objective
coverage_requirements
allowed_output_types
output_schema_version
runtime limits / deadline
optional tool policy
```

Internal delegation is declared as permitted, not prescribed:

```text
native_orchestration_allowed = true
parallelize_only_independent_work = true
return_summaries_not_raw_transcripts = true
```

DTE must not require a fixed `explorer + critic + verifier` topology. It specifies research obligations and output constraints; the native runtime chooses whether and how to delegate.

An `EpisodeResult` should contain at least:

```text
episode_id
input_graph_revision
status
structured role output
runtime reference / diagnostics
output hash
```

Runtime thread IDs, response IDs, compaction summaries, and descendant-agent traces are optional observability or recovery metadata. They are not graph facts.

## 8. Executor output contract

The backend invokes an Executor Episode only after Judge and EvolutionController have assigned expansion rights.

Required output shape:

```json
{
  "nodes": [
    {
      "node_id": "unique-id",
      "node_type": "candidate",
      "claim": "...",
      "rationale": "...",
      "assumptions": [],
      "evidence": [],
      "risks": [],
      "parent_ids": ["expanded-parent-id"],
      "confidence": 0.0,
      "status": "frontier"
    }
  ]
}
```

Outputs are validated before any graph mutation:

- returned children must include the expanded parent id;
- returned children must have `status = "frontier"`;
- returned children cannot be `synthesis` nodes;
- returned children cannot pre-fill controller-owned fields;
- returned child count cannot exceed the episode grant;
- node IDs must not collide with committed graph nodes;
- the input parent/graph revision must still be current;
- stale, malformed, timed-out, or rejected results leave graph state unchanged.

All accepted graph mutation must pass through one backend commit boundary. Natural-language chat output, Markdown reports, temporary files, or raw subagent summaries cannot mutate DTE state.

## 9. Oracle task boundary

Judge and Relation tasks are observable bounded functions:

```text
JudgeOracle: nodes -> scores/reasoning/risks/evidence_gaps
RelationOracle: nodes -> equivalent|complementary|conflict|independent + rationale
DiscriminatorOracle: conflicting nodes -> discriminator question
```

These tasks may be implemented by native model episodes and may internally use subagents. They do not provide latent token vectors and do not replace embedding geometry.

Judge, Relation, Executor, and Synthesis must remain logically isolated even when one native runtime executes all of them at different times.

## 10. Optional evaluator and evidence services

Executable, symbolic, numerical, bibliographic, or formal evaluators may be called when a local claim is verifiable.

They are optional evidence providers, not the global DTE objective.

DTE must distinguish at least:

```text
research potential
strength of available evidence
epistemic uncertainty
```

A branch with high potential and weak evidence may deserve exploration. A branch with strong local evidence may still have low research value. These quantities must not be collapsed into an AlphaEvolve-style scalar correctness reward by default.

Evaluator facts may support or refute node claims, but they do not decide global allocation or final synthesis on their own.

## 11. Persistence, revisions, and anti-bypass guarantees

The minimum reliable runtime should persist enough information to reject stale or duplicate work:

```text
run_created
episode_granted
episode_started
episode_completed
episode_failed
output_rejected
nodes_committed
judge_recorded
allocation_recorded
synthesis_completed
```

A lightweight SQLite or JSONL ledger is sufficient for the first implementation. A large workflow framework or cryptographic event-sourcing platform is not required.

Every model episode is treated as an untrusted producer. Ultra or another root model may self-organize inside the episode, but it must not receive write access to controller state, graph storage, Judge/embedding caches, allocation functions, merge application, stop conditions, or final report commitment.

The only successful exit from a model episode is a validated structured result.

## 12. Cache requirements

Embedding and Judge caches use separate namespaces.

Judge cache identity must include at least:

```text
content_hash
judge_model_or_snapshot
reasoning_profile
rubric_version
prompt_version
output_schema_version
```

Do not reuse cached Judge output across materially different model, rubric, prompt, or schema profiles.

## 13. Performance requirements

- Cache embeddings by canonical node-content identity and provider namespace.
- Cache Judge scores only when content and Judge contract identity are unchanged.
- Batch multiple node evaluations where feasible.
- Avoid injecting full graph context into every episode.
- Prefer node summaries, evidence references, and graph deltas.
- Keep stable prompt prefixes where the runtime benefits from prefix caching.
- Let native runtimes perform bounded internal parallelism instead of repeatedly launching unnecessary top-level CLI processes.
- Do not require or audit a minimum physical subagent count as a correctness condition.

## 14. Merge skeleton

The deterministic backend implements conservative `equivalent_merge` for exact normalized-claim duplicates. Complementary/conflict merge is represented as a Relation Oracle task and may be delegated to a strong model episode.

Relation workflow:

1. Select only relevant pairs or sets using exact duplication, embedding proximity, near-tied value, explicit conflict, or entropy plateau.
2. Pass only the relevant nodes and the Relation contract.
3. Validate the returned object before any graph effect.
4. Convert validated `equivalent`, `complementary`, or `conflict` output into a `MergeProposal` or discriminator task.
5. Let the backend, not the model, decide whether and when to apply the proposal.
6. For `independent`, preserve the branches and continue normal Judge/EvolutionController allocation.

## 15. Strict-run control and forced synthesis

`strict-run` may be supervised by the main agent/user through a narrow control file. This is not a new oracle and does not replace Judge, EvolutionController, allocation, relation, or synthesis.

Default CLI location:

```text
<out-dir>/strict_run_control.json
```

Supported control object:

```json
{
  "action": "force_synthesis_after_current_task",
  "requested_by": "main_agent",
  "reason": "checkpoint has enough coverage",
  "scope": "all"
}
```

For targeted synthesis:

```json
{
  "action": "force_synthesis_after_current_task",
  "requested_by": "user",
  "reason": "focus on the no-go branch",
  "scope": "node_ids",
  "node_ids": ["n1"]
}
```

The backend reads this file only at safe points: after a Judge/EvolutionController/allocation checkpoint and after each expanded node has finished. It must not interrupt a running oracle subprocess. Invalid control JSON fails the run instead of being ignored.

Forced synthesis must be recorded as one of:

```text
main_agent_requested_synthesis
user_interrupted_for_synthesis
```

It must not be recorded as `entropy_plateau`. Artifacts must include the control path, stop reason, selected scope, and frontier branches left unexplored.

## 16. Explicit non-goals

DTE must not:

- become a wrapper around AlphaEvolve or OpenEvolve;
- treat automatically verifiable reward as the primary signal for all research nodes;
- let Ultra or another native orchestrator decide global allocation, graph mutation, stopping, or final synthesis;
- prescribe every physical subagent call from backend code;
- require a mandatory physical Explorer before seed generation;
- expose DTE controller state as writable model context;
- accept direct final answers from Executor or Seed episodes;
- depend on one specific transport such as `codex exec`;
- introduce unbounded recursive agent fan-out.
