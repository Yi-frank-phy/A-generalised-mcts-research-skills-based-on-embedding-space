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

The canonical local upper-confidence primitive is:

```text
U_i = V_i + SD_i
```

where:

- `V_i` is an observable Judge estimate of research potential;
- `SD_i` is a standard-deviation/standard-error-like uncertainty estimate for branch `i`;
- the current `uncertainty_i` field is the provisional estimator supplied as `SD_i`.

Global temperature does not multiply the local uncertainty term. Legacy `tau` and `c_explore` inputs may remain temporarily parseable for compatibility, but they do not change canonical UCB.

Cost is not part of UCB by default. Compute and quota limits are enforced outside the objective through run policy, episode policy, and hard caps.

Judge value is not a correctness proof. It estimates whether a branch is coherent, informative, tractable, and worth further investigation under incomplete evidence.

## 3. Geometry and embedding dimension

The current production-compatible controller uses normalized embedding geometry as a kernel-smoothed continuation of the original finite-state entropy construction. For normalized embeddings `z_i`, let

```text
d_ij^2 = ||z_i - z_j||^2
h^2 = median_{i<j}(d_ij^2) / 2
K_ij = exp(-d_ij^2 / (2 h^2))
rho_i = (1/N) * sum_j K_ij
H_geom = -(1/N) * sum_i log(rho_i)
```

The adaptive scale makes a typical pair have kernel overlap `exp(-1)`. Because the self-kernel is one,

```text
1/N <= rho_i <= 1
0 <= H_geom <= log(N)
```

`H_geom` is a bounded soft-discrete entropy. It is not the coordinate-dependent differential entropy of a normalized `D`-dimensional Gaussian KDE, and no Gaussian density-normalization factor belongs in it.

Current-frontier absolute local evidence and uncertainty are

```text
n_eff,i = N * rho_i
SD_i = 1 / sqrt(n_eff,i)
```

so the geometry layer does not min-max normalize uncertainty within each batch. Per-batch min-max normalization would erase the absolute evidence scale by forcing the densest branch to uncertainty zero and the sparsest branch to one.

The public compatibility controller still uses the observable Judge score as provisional `V_i`, and canonical UCB remains

```text
U_i = V_i + SD_i
```

For the current UCB spectrum define

```text
p_i(T) = exp(U_i / T) / sum_j exp(U_j / T)
H_B(T) = -sum_i p_i(T) log p_i(T)
```

For non-degenerate UCB values, `H_B(T)` is monotone increasing in `T`. The current controller hypothesis therefore determines the effective allocation temperature by solving

```text
H_B(T_t) = H_geom(t)
```

for the current frontier. Temperature must therefore depend on both the geometry entropy target and the current UCB scale. The old formula

```text
T = T_max * H_geom / log(N)
```

is superseded and must not control allocation.

The persisted field `normalized_temperature` remains for compatibility and replay telemetry; its value is the normalized geometry-entropy coordinate `H_geom/log(N)` when `N>1`, not `T/T_max` and not the effective Boltzmann temperature.

Cross-iteration `delta H` is plateau/continuation telemetry only. It does not define `U_i` or the current effective temperature.

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

Given canonical UCB values `U_i`, global temperature `T`, and a per-iteration allocation mass `C`:

```text
p_i = exp(U_i / T) / sum_j exp(U_j / T)
q_i = C * p_i
```

At `T = 0`, this is interpreted by its zero-temperature limit: probability mass is supported only on the maximal-`U` branch or tied maximal branches. A tiny positive denominator may be used internally for numerical stability, but controller telemetry must not expose an artificial positive temperature floor.

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
max_committed_search_nodes = 20
max_iterations = 10
allocation_mass_per_iteration = 3
max_children_per_iteration = 5
entropy_plateau_confirmations = 2
max_relation_pairs_per_episode = 3
max_relation_enrichment_pairs = 3
```

`max_committed_search_nodes` is the primary, non-renewable search-cost
budget. It counts Seed nodes and every successfully committed Executor child,
including nodes later marked `closed`, `archived`, or `merged`, and excludes
Synthesis nodes. Retry, rejection, failure, cancellation, and empty results do
not consume a node slot. Merge reduces canonical graph complexity but never
refunds already-spent node budget. Initial nodes above the cap are invalid;
initial nodes equal to the cap are still judged before the terminal gate.

Before issuing Executor grants, the controller caps the sum of all granted
children by the remaining node slots. The Executor request and commit boundary
revalidate that grant, so already-authorized work can drain without exceeding
the cap. `max_iterations` is only an absolute pathological-loop guard.

For temporary input compatibility, the legacy field `total_child_budget` is accepted only as a deprecated alias for `allocation_mass_per_iteration`. Canonical serialization and schemas use the new fields.

If tentative allocation exceeds `H`, the controller must trim children by a deterministic marginal-priority rule derived from allocation mass and node priority. It must not trim by input order.

Python's built-in bankers rounding is not the normative rule. Half values below one use round-half-up semantics.

Boltzmann allocation uses `U_i`, so local uncertainty affects actual expansion rather than merely display ranking. Global temperature controls how concentrated that allocation is across the already-computed UCB values.

The current soft-discrete geometry entropy sets the target Boltzmann allocation entropy, while its cross-iteration delta emits a replayable plateau signal. It has no
direct Synthesis authority under `bounded_node_yield_v1`. A confirmed plateau
or a single canonical frontier triggers a continuation gate. Continuation is
granted only when committed facts show a narrow material-yield signal, a
specific frontier target has positive allocation, and a node slot remains.
Epistemic-ledger signals are one-use bounded heuristics, not scientific truth;
they never change UCB, model strength, or either hard budget. Persisted legacy
App runs retain `legacy_entropy_v1` behavior after hash-checked migration.

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

Logical role separation does not imply one physical model call per role, and a
role-specific payload does not by itself isolate execution context. A run must
record `strict_fresh_context`, `shared_context_single_agent`, or
`legacy_unverified` for every role episode. Role labels must never be described
as independent review unless the runtime supplies a fresh role-session
attestation that the backend accepts.

The request contract records `fresh_context_required=true` but never
pre-claims verification. Verification is a committed transport/runtime fact.
Model-authored results may report only `runtime_reported`; they cannot grant
themselves `backend_verified`, which is reserved for a trusted adapter path
outside structured model output. DTE validates only host-exposed attestations,
not hidden provider-internal context state.

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

Judge receives only its blinded grant, rubric, run problem/goal, constraints,
and allowed evidence. In `strict_fresh_context`, it must execute in a fresh
role session whose returned session ID has not appeared in the run and whose
manifest hash matches the exact serialized request. A shared main conversation
is `shared_context_single_agent`, not isolation; its results carry a
correlated-error-risk disclosure.

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
- unresolved material conflicts in the material review pool must either be resolved or explicitly disclosed;
- the mere existence of a Relation candidate must not automatically forbid synthesis.

The App-native implementation separates the material synthesis scope from the
presentation headline scope. Structured required coverage is satisfied first;
support/challenge dependency closure, material counterexamples, and disclosures
remain material even when the headline projection is capped at eight nodes.
Every eligible node outside material scope receives a versioned counterfactual
disposition. Its materiality audit compares exact structured marginal coverage,
assumptions, evidence, limitations, counterexamples, dependencies, conflicts,
and normalized claims; Judge score is not the primary proxy. Required coverage,
undisposed material nodes, incomplete material provenance, and unresolved
blocking Relation pairs fail natural readiness closed.

The material Relation review pool includes material scope, counterfactual-
material unselected disclosures, dependencies, and counterexamples. Shared
coverage alone is never a blocking conflict: coverage-overlapping alternatives
are bounded enrichment unless exact duplication, shared-evidence claim
divergence, or an explicit challenge/contradiction/counterexample supplies a
blocking reason. Deterministic node-disjoint batching uses polynomial weighted
greedy matching plus a bounded one-edge replacement pass. It preserves the
critical/material/priority/stable-ID preference but does not claim global
optimality. Enrichment
is ledger-aware and capped by the run-level
`max_relation_enrichment_pairs` budget (default `3`, `0` disables enrichment).
Only a successfully committed nonblocking observation consumes one pair;
retry, failure, cancellation, and expiry do not. Blocking work never consumes
this budget.

Oracle-visible Relation v2 input contains only blinded node material, direct
evidence references, minimal blinded ancestry, the relation labels, and the
output contract. Selection membership, materiality, priority, candidate reason,
Judge conclusions, and Judge provenance remain backend-only and are reattached
after classification.

Readiness is true only when the complete current blocking inventory is registered, its unresolved count is zero, confirmed equivalent merges have been applied, and every material conflict is resolved or represented by an explicit disclosure obligation. Readiness may be true while bounded enrichment remains pending; the sticky terminal action is written only after eligible enrichment is exhausted, absent, or disabled.

Relation observations are committed through `commit_episode_result(...)` into a versioned relation ledger. Non-merge observations increment graph revision once without revising source nodes. An equivalent observation is recorded first and then backend deterministic canonicalization applies an atomic merge transition, preserves all source nodes and provenance, revises only affected nodes, and excludes absorbed aliases from provisional Synthesis selection. A request/result with overlapping pairs, or a merge transition that would map one absorbed node to different canonical nodes, is rejected as a whole. This is a transactional merge-safety invariant, not a verification rule. Material conflicts must be resolved or carried forward as an explicit disclosure obligation. `DiscriminatorTaskProposal` remains persisted metadata only: this implementation does not schedule a discriminator, source checker, proof checker, verifier, correctness verdict, reward, or pass/fail gate. Relation classifies semantic relationships; backend validation checks only protocol and transaction legality.

### Phase F: Synthesis

Compress the serialized terminal handoff into a report or synthesis node. In
strict mode, natural-language Synthesis requires a fresh context whose complete
input is that handoff. If no isolated Synthesis runtime is available, the
backend emits only its deterministic report; main-conversation prose is
unisolated commentary and is not committed Synthesis. Hidden, undisclosed
unselected node content is absent from the handoff.

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

The implemented P1 App-native slice uses strict `EpisodeRequest` and `EpisodeResult` envelopes with `attempt_id`, persistent Judge/Executor/Relation lifecycle, graph and per-node revisions, and role-dispatched `commit_episode_result(...)` as the only mutation path for episode output. A valid Judge result commits only observable score/reasoning/risk observations; `next-episode` then runs the existing embedding/KDE, entropy, uncertainty, UCB, and allocation functions inside the backend before granting Executor work. When the controller intends to terminate, the backend selects provisional synthesis branches, completely inventories blocking Relation obligations, commits validated relation facts and permitted equivalent merges, evaluates readiness, optionally spends a bounded run-level semantic-enrichment budget, and only then writes a sticky terminal action. The current Codex App main agent performs only the bounded role episode and never interprets controller mathematics or launches a second Codex. Native Seed and final Synthesis remain deferred. See the normative `docs/specs/p1-native-ultra-agentepisode-codex-app-profile.md`.

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

Judge, Relation, Executor, and Synthesis remain distinct authority contracts.
Prompt-level role switching inside one conversation is not fresh-context
isolation. Strict mode fails before scientific commit when the manifest,
attestation, or fresh role-session identity is missing, mismatched, or reused.
Shared mode is an explicit unverified fallback and cannot claim independent
review.

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

## 15. Strict-run operator synthesis command

`strict-run` accepts a narrow synthesis request through a control file. This is a privileged controller command, not a new oracle, and it does not replace Judge, EvolutionController, allocation, relation, or synthesis. A model-facing main agent is a user-delegated operator proxy and may submit this command when the validated `DTERunSpec.operator_policy.main_agent_may_request_synthesis` is true. It may not directly mutate controller-owned state.

```text
observation != authority
delegation + policy + validated command = authority
```

Reading `checkpoint_summary.md`, `main_agent_status.md`, `frontier.md`, `entropy_trace.md`, or `strict_run_status.json` does not grant state-machine permission by itself.

The CLI polls this path by default:

```text
<out-dir>/strict_run_control.json
```

`--control-path <operator-controlled-path>` may select another location. `requested_by` identifies the actor for audit; `operator_policy` determines whether that actor is authorized. The JSON field is not cryptographic proof of identity and does not create authority by itself. This headless compatibility phase trusts the root/operator execution context invoking the backend. Codex App runs instead use the hook-enforced driver contract in Section 19.

Supported control object:

```json
{
  "action": "force_synthesis_after_current_task",
  "requested_by": "main_agent",
  "reason": "operator proxy found sufficient coverage for synthesis",
  "scope": "all"
}
```

For targeted synthesis:

```json
{
  "action": "force_synthesis_after_current_task",
  "requested_by": "main_agent",
  "reason": "focus on the no-go branch",
  "scope": "node_ids",
  "node_ids": ["n1"]
}
```

The backend reads this file only at safe points: after a complete Judge/EvolutionController/allocation checkpoint and after an already-started node expansion has returned complete, validated Executor output. It validates the schema and `OperatorPolicy` before applying the command. It must not interrupt a running oracle subprocess, consume partial output, skip validation, or commit a partial expansion. Invalid or unauthorized control JSON fails closed instead of being ignored or remapped.

Main-agent-requested synthesis must be recorded as:

```text
main_agent_requested_synthesis
```

Direct user requests record `user_interrupted_for_synthesis`. Main-agent requests record `main_agent_requested_synthesis`. Neither may be recorded as `entropy_plateau` or algorithmic convergence. Artifacts must include the control path, actor, audit reason, selected scope, and frontier branches left unexplored. Normal search ends only because the DTE controller reaches its stopping policy or because it accepts an authorized synthesis command at a safe boundary.

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

### 16.1 V1 architecture freeze and evidence-gated evolution

The current protocol is feature-complete for the v1 research workflow.
Repository development now moves from speculative architecture expansion to
real-run use, evaluation, maintenance, and fixes.
Do not pre-emptively add:

- a native final Synthesis episode: terminal synthesis remains the main agent's
  reporting step over backend-controlled provisional selection and handoffs;
- a verifier, human-approval gate, or artifact-backed correctness authority;
- a dormant-node state: zero-allocation candidates remain ordinary `frontier`
  nodes and continue to participate in the existing allocation policy;
- additional reward, convergence, learning, reliability, or control metrics
  beyond the current read-only observability contract.

These boundaries are not claims that no future change is possible. Reconsider
one only when real runs provide a reproducible failure case, comparative outcome
evidence, or a concrete protocol requirement that the current v1 mechanisms
cannot satisfy. A new metric must first be justified as decision-relevant
observability and must not become controller input by implication. Architecture
may not expand merely because a plausible extension can be imagined.

## 17. Deterministic observability and feedback boundary

Observability is a versioned, read-only projection over the persistent App run
state, committed episode results, controller transition records, Relation and
readiness ledgers, append-only telemetry, and an independent feedback ledger.
It is not a second graph, an event-sourced controller, or a source of mutation
authority.

The stable first-version projection must expose:

```text
run identity and immutable configuration
Judge / Executor / Relation episode and attempt funnels
node creation, Judge, allocation, expansion, selection, Relation, and merge lineage
allocation outcomes and explicitly named internal proxy yields
Judge score versus later observable state, labelled as non-causal posterior proxies
Relation yields by scheduling class and candidate reason
controller iteration trajectory
committed/remaining search-node budget and canonical/live/merged counts
bounded continuation-gate trajectory and its material-yield record identities
read-only frontier eligibility and zero-allocation streak diagnostics
deterministic rejection categories
self-reported data-quality limitations
```

The projection must be deterministically rebuildable from committed artifacts.
It must not call a repair-on-read path, write graph or artifact mirrors, revise a
node, recompute a Judge score, or change controller decisions. Missing legacy
fields remain `null` or are reported as missing; they are never silently treated
as zero. Non-deterministic generation timestamps are not part of the core run
summary.

Frontier wait diagnostics are reconstructed only from controller iteration
records. They are not stored on `SearchNode`, do not add an age bonus, and must
not affect UCB or allocation. Continuation material-yield fields explain a
bounded process heuristic; they are not scientific truth, convergence evidence,
or permission to increase a node, iteration, child, or model-compute budget.

Runtime aggregate diagnostics may include provider- or main-agent-reported
counts for internal subagents, parallelism, tool calls, rounds, failures, and
tokens. Every such field is optional, nullable, source-labelled, and ignored by
all commit and controller decisions. Hidden reasoning, full prompts, internal
transcripts, and a complete subagent topology are outside the contract.

Explicit evaluation is written only to a separate append-only feedback ledger.
Feedback may target a run, episode, attempt, node, Relation record, merge
application, or allocation decision. It must validate that the target exists,
preserve the declared source (`user`, `main_agent`, or `external_evaluator`), and
contain at least one substantive score, label, comment, or metadata field. It
must never rewrite Judge output, graph state, telemetry history, allocation, or
stopping state.

Internal process proxies such as allocation yield, selected-descendant yield,
merge rate, conflict discovery rate, retry/rejection rate, readiness cost, and
latency describe only the recorded DTE process. Claims about scientific utility,
novel route discovery, avoided false progress, time saved, or advantage over a
non-DTE baseline require user feedback, a benchmark, or later external outcomes.
The observability interface must not present internal correlation as calibration,
causation, or proof that the architecture is effective.

## 18. Epistemic provenance and researcher handoff

The App-native run state owns one additional versioned committed-fact
collection:

```text
AppRunState.epistemic_ledger
```

It contains only structured statements, directed epistemic edges, and explicit
path epistemic dispositions accepted with a completed Judge or Executor result.
The ledger is committed in the same copy-validate-replace transaction as that
episode result. A JSON artifact under `epistemic/ledger.json` is a derived
mirror, never a second fact source. Legacy App states migrate to an empty ledger.

Every current epistemic record carries one source label:

```text
agent_reported
external_artifact_backed
backend_derived
```

Episode producers may submit only `agent_reported` or genuinely referenced
`external_artifact_backed` contributions. They may not manufacture
`backend_derived` facts. Backend validation establishes
identity, authorization, lifecycle, reference existence, safe artifact paths,
and provenance; it does not establish scientific truth. An external artifact
reference means only that the record points to that artifact. The backend does
not verify the artifact, its assumptions, its applicability, or the scientific
claim. Human-readable output labels this provenance as `artifact_referenced`;
the persisted `external_artifact_backed` token remains for output-hash and state
compatibility.

Executor and Judge output may contain one bounded `epistemic_contributions`
object. Nonmaterial nodes are not forced to emit filler. A material node with
no qualifying contribution may still commit useful node content.
`material_provenance_policy=terminal_disclosure` (the App-native default)
records missing provenance in an explicit degraded terminal result.
`strict_repair` grants at most one backend-controlled provenance-only Judge
repair per missing material node; it cannot rescore nodes or verify truth. If
repair remains incomplete, the run records exhaustion and terminates degraded
rather than looping in `await_operator_decision`. It
supports:

Natural synthesis remains blocked by unresolved required coverage. An
authorized forced or scoped synthesis request is honored at a synthesis-safe
checkpoint; omitted coverage is then preserved in the degraded terminal record
instead of negating the stop command.

```text
statement types: claim | assumption | evidence | open_question | failure_mode | heuristic
edge types: supports | challenges | requires | qualifies | contradicts | derived_from
path dispositions: blocked_by_assumption | counterexample_found | challenged |
                   contradicted | inconclusive | insufficient_support
```

Directed-edge convention is:

```text
evidence --supports--> claim
claim --requires--> assumption
counterexample --challenges--> claim
claim B --qualifies--> claim A
claim --derived_from--> source
```

Node claims use `node-claim:<node_id>`. Current-output statements use
`local-statement:<local_id>` and are resolved transactionally to committed
stable IDs. Other machine references may identify an existing committed
epistemic record, Relation record, merge application, episode result, safe run
artifact, or explicit external reference. `learning:` references are not part
of the DTE contract. Free text is never mined to infer edges.

Stable record IDs bind:

```text
run_id
episode_id
attempt_id
output_hash
record local_id
record type
```

They do not semantically merge equal natural-language text across contexts.
Unknown identities, unauthorized target nodes, unsafe or missing artifact
references, duplicate local IDs, forged source types, invalid dispositions, or
duplicate stable IDs reject the whole episode commit. Failed, stale, late,
cancelled, expired, superseded, or rejected attempts contribute no epistemic
facts. `counterexample_found` and `contradicted` require non-empty basis
references.

Search disposition and epistemic disposition are independent projections.
Backend-derived search facts include `selected`, `not_selected`, `merged`,
`closed`, `out_of_budget`, and `not_explored`. They never imply a scientific
status. In particular:

```text
not_selected != contradicted
low Judge score != false
out_of_budget != unpromising
merged != universally scientifically redundant
```

The deterministic read-only epistemic model projects committed run state,
episode results, the epistemic ledger, the existing Relation ledger, merge
applications, provisional selection, and observability into a terminal
handoff. It never repairs or rewrites the run. The formal JSON
handoff traces each material-scope node claim through required
assumptions, supporting and challenging records, producer episode/attempt,
artifacts, Relation conflicts, and merge provenance. Relation classifications
are referenced from the existing Relation ledger; no second Relation truth is
created. The presentation headline scope is a compact projection only.
Undisclosed nonmaterial node content is excluded. The handoff describes
committed material claims and explicit disclosures, not an audited final
natural-language Synthesis answer.

The handoff also reports correlated-error risk indicators. Model/runtime
metadata is used only when explicitly persisted; missing metadata remains
unavailable rather than guessed. Same-model cross-role correlation,
agent-only support, absent structured support, unresolved assumptions, and
self-referential support are risk indicators, not correctness rates,
independent-validation rates, or scientific reliability scores.

The former file:

```text
epistemic/researcher_learning.jsonl
```

is a deprecated external artifact ignored by current DTE. Current code does not
read, interpret, export, migrate, repair, or modify it. Human learning, external
tool validation, literature checking, independent proof, and final researcher
judgment remain outside DTE authority. Explicit evaluations of a run or decision
continue to use the independent `record-feedback` ledger, which is never an
epistemic verifier or controller input.

Historical ledgers containing the retired `human_confirmed` source are not
upgraded or rewritten. The read-only handoff may isolate those legacy records
and report a partial-data limitation; it never converts them to
`agent_reported` or treats them as verification. Controller resume remains
fail-closed for such an invalid legacy authority source rather than rewriting
the run or recomputing a historical EpisodeResult hash. Normal PR #19/#20 runs,
whose commit boundary already rejected that source, resume unchanged.

## 19. Hook-enforced Codex App execution contract

The production Codex App entrypoint is the deterministic `hook-driver` protocol:

```text
activate -> init -> (step -> bounded App episode -> submit)* -> handoff
```

The hook dispatcher owns activation, lifecycle interception, resume context, and
early-stop prevention. It never computes Judge output, embeddings, KDE density,
entropy, uncertainty, UCB, allocation, Relation classifications, readiness, or a
terminal decision. Those remain backend-owned transitions. The main agent may
only execute the currently granted `EpisodeRequest` and submit one complete
structured `EpisodeResult`.

Every new production App run persists an `execution_contract` with mode
`hook_enforced_v1`, the enforcing Codex session identity, activation source,
manifest identity hash, driver protocol version, and a hash of the current
single-use capability. A mutating App API validates that contract before it
loads or changes protected run state. A direct call which omits the capability,
uses another session, presents an old capability, or disagrees with the manifest
identity fails closed. After every accepted state transition the driver rotates
the capability. Hook filtering is therefore a lifecycle guardrail, while the
backend mutation boundary is the anti-bypass authority.

Existing persisted states and explicit development, fixture, smoke, and
headless-legacy runs use `direct_legacy`. Loading an old state supplies that
compatibility contract in memory without rewriting historical result payloads,
hashes, revisions, or telemetry. `strict-run --mode real` remains an explicitly
labelled headless legacy path and is not an App-native production entrypoint.

One Codex `session_id` has at most one nonterminal DTE hook session. Its
`dte-hook-session.v1` manifest is stored outside the run and points to a run
under `<activation-cwd>/.dte/runs/<run_id>`. It records the root turn, phase,
current episode/attempt/revision identity, previous receipt hash, the unique next
action, and protected paths. Protected paths include the complete supplying
`src/dte_backend` package, not only the driver modules. Hook path comparisons
resolve relative targets against the tool's declared workdir or event cwd and
handle Windows cross-drive paths without treating an unformable relative path
as authority. Direct production
mutators and headless-real entrypoints are recognized through both
`python -m dte_backend` and the published `dte-backend` console command.
Driver receipts use `dte-hook-receipt.v1`, include
before/after state hashes and the preceding receipt hash, and are atomically
written as an append-only reconstructable chain. Timestamps may be recorded for
operations but never participate in receipt or manifest state identity.

An `episode_required` transition never embeds the complete `EpisodeRequest` in
its receipt. The receipt carries one compact `dte-request-ref.v1` identity with
the request hash, canonical UTF-8 size, and bounded chunk count. The owning root
turn reads the immutable current request through `hook-driver request
--chunk-index N`; each `dte-request-chunk.v1` is capability-bound, hash-checked
against the durable attempt, and limited to 8192 content bytes. Request reads are
read-only projections: they append no receipt, rotate no capability, and advance
no controller or attempt state. Calling `step` while a request is already active
is the same idempotent projection and cannot create another transition.

A repeated Stop caused by an unfinished DTE turn is an operational pause, not an
episode failure. The hook blocks the first early Stop, then records a `pause-turn`
receipt while preserving the backend phase and active attempt. A later root turn
records `resume-turn` and continues that identity. When the same invocation is
resumed from another Codex session, the driver may atomically transfer the
execution contract and capability only after matching the invocation, RunSpec,
initial nodes, worktree identity, and durable run. The old capability becomes
invalid immediately; historical receipts are never rewritten.

Hook initialization also uses an atomic create-if-absent invocation registry
keyed by repository and worktree identity, hook type, RunSpec hash,
initial-node hash, explicit nonce, and replay source where applicable.
Git identity includes HEAD, the complete index, binary staged/unstaged state,
untracked nonignored content, and linked-worktree/common-directory identity;
ignored DTE outputs are excluded. Non-Git activation uses an explicit
filesystem snapshot fallback. Registry v2 records generation, owner PID/token,
timestamps, and recovery lineage. A live initializer is never overwritten;
failed or ownerless generations may be retried deterministically.
Repeating the same invocation returns the existing run instead of creating a
nominally independent result. An explicit replay creates a distinct key and
persists `replay_of_run_id`, source committed-result hashes, trigger source,
and whether model work was reused, rerun, or unavailable. Replay lineage is
audit metadata, not evidence of independence.

Retry is valid only after a durable rejected, failed, cancelled, or expired
attempt. An active attempt must first pass through the explicit `fail-attempt`
or `cancel-attempt` control transition with a reason; transport rereads and turn
resumption do not consume retry. Commit preparation may return at most two
`repair_required` outcomes on the same active attempt for correctable JSON,
schema, output-hash, or epistemic-reference errors. Repair never changes the
graph, ledger, controller, or attempt identity. Authority, capability, stale or
late identity, ungranted-node, and forged-controller violations remain hard
failures. Bare `repository:` references are not ledger identities and receive a
repair diagnostic directing the producer to `artifact:` or `external:`.

`cancel-attempt` preserves a resumable run. `cancel-run` (and the legacy
App-facing `cancel` alias) writes a backend-owned cancellation record, cancels
any active attempt, clears active identity, emits `run_cancelled`, and terminates
the invocation. A cancelled duplicate invocation is observable but never
silently restarted; a new run requires an explicit replay or nonce.

Before any App call which may persist state, the driver writes a run-external
internal operation intent. Recovery distinguishes an unchanged authoritative
App state, which is safe to retry, from a fully advanced App state, which is
reconciled into an explicit `recovery:<operation>` receipt without inventing the
lost original return payload. Capability rotation records current and pending
values before changing the App execution contract. A prepared receipt is
persisted before the manifest advances its sequence/head; restart idempotently
completes receipt, manifest, capability promotion, and terminal artifact
materialization. The external session, receipt, and capability schemas remain
stable; the internal journal is not model-facing authority.

Static installation verification proves only the expected handler definitions,
matcher, dispatcher content hash, normalized pinned Skill root, and local
self-test. The trusted command passes both bindings before the dispatcher
imports the backend, so an unrelated editable package cannot become the
enforcement implementation. Production use also requires a trusted definition,
full App restart, and explicit no-state
`UserPromptSubmit` plus denied `PreToolUse` delivery probes. Missing probe
acknowledgement is an App integration failure and never authorizes a direct
mutator fallback.

The enforced terminal boundary ends only after both deterministic terminal read
models and `terminal-handoff.json` have been generated for a backend terminal
state. The main agent then writes the final natural-language report. The hook
does not grade that prose and the handoff does not assert scientific correctness.
This contract protects protocol safety and workflow completeness against normal
Codex agents and ordinary user permissions; it is not a defence against an
administrator changing managed files, disabling the platform, or replacing the
installed backend.
