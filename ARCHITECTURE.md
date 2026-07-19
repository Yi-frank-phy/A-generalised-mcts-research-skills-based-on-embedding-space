# Architecture Decision: DTE as the Outer Epistemic Controller

## Decision

DTE remains the mandatory outer controller for frontier search, graph state, uncertainty-aware allocation, Relation handling, stopping, and final synthesis.

Native model runtimes such as Codex/Ultra may self-organize subagents inside one bounded role episode, but they are not the root controller of the DTE run.

The architecture is therefore:

```text
DTE controls cross-episode recursion.
Native model runtimes control bounded within-episode orchestration.
Guard scripts and a single commit boundary separate the two.
```

This decision replaces two weaker alternatives:

1. DTE explicitly micromanages every physical subagent call.
2. Ultra is the root agent and merely receives a prompt asking it to follow DTE.

The first duplicates native orchestration and creates unnecessary I/O. The second allows the root model to bypass Judge, allocation, Relation, or synthesis. Neither is acceptable as the target architecture.

## Why DTE remains necessary

DTE is designed for research settings without a stable, automatically verifiable reward. Its advantage is the information-theoretic and evolutionary selection layer:

```text
Judge potential
  -> embedding geometry / density / novelty
  -> uncertainty-aware UCB
  -> Boltzmann expansion
  -> Relation / discriminator handling
  -> DTE-selected synthesis
```

AlphaEvolve-style evaluators may provide local evidence for verifiable subclaims, but they do not replace this outer search process. DTE searches over strategies, explanations, conjectures, representations, and research directions whose value cannot usually be reduced to a deterministic scalar score.

## Key distinctions

### Logical roles are not physical model calls

```text
logical role separation != one model call per role
```

Judge, Executor, Relation, Seed, and Synthesis remain separate contracts to reduce self-justification and preserve graph semantics. One native runtime may execute different role episodes at different times, and one Executor episode may internally use multiple subagents.

### DTE children are not subagent threads

A DTE child is a committed graph node. A Codex subagent is an implementation detail inside an episode.

For example, DTE may grant:

```text
max_returned_children = 2
```

while the native runtime internally uses five workers for literature search, counterexample generation, coding, critique, and aggregation. The episode may still commit at most two validated SearchNodes.

### DTE owns vertical search; native orchestration owns horizontal work

```text
DTE graph depth      = cross-iteration epistemic recursion
native subagent work = bounded parallelism inside one episode
```

The implemented Judge/Executor/Relation slice represents structural graph state with a monotonically increasing graph revision and a revision for every committed node. A grant snapshots both. `commit_episode_result(...)` dispatches by the committed request role, rechecks the full envelope and selected revisions, validates on a copy, then replaces graph state once. Judge commits revise only granted nodes with validated observable judgments; controller progression separately revises frontier geometry/allocation fields; Executor commits close one granted parent and add bounded children; Relation commits add validated semantic edges and only backend-applied equivalent merges revise affected nodes. Rejections change neither graph nor node revisions. This is intentionally not event sourcing or distributed version control.

The production Codex App path is a persistent driver protocol: the backend grants one request, the current App main agent performs it with native opaque orchestration, and the backend accepts one complete result. The backend does not launch another Codex process. Command/subprocess and deterministic adapters remain legacy/headless and test implementations of the transport-neutral boundary. Internal agent count, names, routing, traces, token usage, and quota remain unavailable telemetry rather than correctness conditions.

Each logical episode has explicit attempts. Only the active `in_progress` attempt may submit; retry supersedes the previous attempt and creates a new `attempt_id`. Cancelled, expired, failed, superseded, rejected, or committed attempts are terminal for acceptance. The App main agent supervises its invisible internal work, while the backend enforces deadlines and lifecycle at submission.

The default native delegation depth should remain shallow. Recursive fan-out inside an episode duplicates the DTE search tree and makes budget semantics difficult to interpret.

## Control ownership

| Component | Owns | Must not own |
|---|---|---|
| DTE backend | graph revision, frontier, embedding, entropy, UCB, allocation, merge application, stop conditions, synthesis checkpoint | open-ended model research |
| Seed episode | generation of materially distinct initial candidate nodes | ranking, allocation, winner selection, final answer |
| Executor episode | bounded research/coding/proof work on one assigned parent | graph mutation, Judge metrics, global budget, synthesis |
| Judge episode | observable potential score, reasoning, risks, evidence gaps | child generation, allocation, embedding, graph mutation |
| Relation episode | equivalent/complementary/conflict/independent classification and discriminator proposal | direct merge or deletion |
| Synthesis episode | report generation from a fixed selected checkpoint | continued exploration or hidden graph mutation |
| Optional evaluator | reproducible evidence about a locally verifiable claim | global branch value or DTE stopping |

## Runtime architecture

```text
User / RunSpec
      |
      v
DTE Driver / Controller
      |
      +-- SeedEpisodeRequest --------------------------+
      |                                                |
      +-- JudgeEpisodeRequest -------------------------|-- Native model runtime
      |                                                |     may self-organize
      +-- ExecutorEpisodeRequest ----------------------|     internal subagents
      |                                                |
      +-- RelationEpisodeRequest ----------------------|
      |                                                |
      +-- SynthesisEpisodeRequest ---------------------+
      |
      v
Role-specific guard and schema validation
      |
      v
Single DTE commit boundary
      |
      v
Graph revision / next controller step
```

The native runtime may choose whether to delegate internally. DTE specifies obligations, permissions, output limits, and accepted schemas; it does not prescribe `explorer + critic + verifier` as a mandatory topology.

## AgentEpisode contract

The stable integration boundary is transport-neutral:

```text
AgentEpisodeAdapter:
    EpisodeRequest -> EpisodeResult
```

Possible transports include:

```text
subprocess / codex exec
Codex SDK
Codex App Server
hosted native runtime
future provider-specific adapters
```

Transport choice must not change DTE graph semantics.

A request includes:

```text
episode_id
role
input_graph_revision
selected node revisions
max_returned_children where applicable
objective
coverage requirements
allowed output types
schema version
runtime limits
optional tool and write-root policy
```

A result includes:

```text
episode_id
input_graph_revision
status
structured role output
runtime reference / diagnostics
output hash
```

Thread IDs, response IDs, compaction summaries, and descendant-agent traces are recovery or observability metadata. They are not accepted as graph facts.

## Anti-bypass boundary

A native model episode is treated as an untrusted producer, regardless of model capability.

It must not receive write access to:

```text
DTE graph storage
controller state
embedding or Judge caches
allocation functions
merge application
stop conditions
final report commitment
```

The only successful output path is a validated structured result.

```text
natural-language answer       -> rejected as graph input
Markdown report from Executor -> rejected as graph input
stale parent revision         -> rejected
controller fields pre-filled  -> rejected
too many returned children    -> rejected
timeout or failed episode     -> graph unchanged
valid structured output       -> eligible for backend commit
```

The current adapter validation is the basis of this firewall and should be extended with graph-revision checks, ID collision checks, and one atomic commit function.

A prompt or Skill instruction alone is not a hard boundary. In the final architecture, DTE Driver calls the native runtime; the native runtime does not decide whether to call DTE.

### Observability is not authority

Checkpoint and status artifacts are read-only observations. Observation alone does not grant permission to advance the state machine, allocate work, stop the search, or commit synthesis. A model-facing root agent is an operator proxy: when the validated `DTERunSpec.operator_policy` permits it, the agent may submit a controller command requesting synthesis. The Python backend validates the command, waits for a safe point, applies the state transition, and records the actor and reason. The main agent never directly mutates controller-owned state or represents its request as algorithmic convergence.

```text
observation != authority
delegation + policy + validated command = authority
```

### Observability is a derived read model

The stable observability interface projects the authoritative App run state,
committed episode results, controller iteration records, Relation/readiness
ledgers, and append-only telemetry into versioned run, episode, node-lineage,
allocation, Judge-posterior, Relation-yield, trajectory, rejection, and
data-quality records.

```text
authoritative persistent facts
        -> deterministic read-only projection
        -> JSON summary / text view / cross-run JSONL
```

The projection does not repair on read, mutate graph or revision state, rescore
nodes, or feed proxy statistics back into allocation. Derived mirrors and
telemetry are cross-checked but do not become a second fact store. Legacy or
missing data is represented as `null` plus explicit data-quality limitations,
not as zero or a fabricated confidence score.

User, main-agent, and external-evaluator judgments are appended to an independent
feedback ledger and remain bound to an existing run, episode, attempt, node,
Relation record, merge application, or allocation decision. Feedback never
rewrites a Judge score or controller decision. Internal process proxies support
diagnosis and comparison; external research effectiveness still requires human,
benchmark, or later-outcome evidence.

### Epistemic provenance is committed fact, not verification

Executor and Judge may add optional, bounded structured contributions to their
role-valid output. The backend validates role authority, node and fact identity,
safe artifact paths, lifecycle, provenance source, and references. It does not
validate scientific truth. The records are installed in the same commit
transaction as the episode result:

```text
accepted Judge / Executor EpisodeResult
        -> stable statement, edge, and path-disposition IDs
        -> AppRunState.epistemic_ledger (authoritative)
        -> epistemic/ledger.json (derived mirror)
```

Stable identities bind `run_id`, `episode_id`, `attempt_id`, `output_hash`, the
output-local ID, and record type. Local statement references resolve inside the
transaction. Unknown node, committed episode/attempt, epistemic record, Relation
record, merge application, or run artifact references reject the entire result.
Stale, failed, cancelled, expired, late, or superseded attempts never contribute
records, and retry makes only the committed attempt visible.

The deterministic epistemic read model combines that ledger with canonical node
claims, committed attempts, Relation records, merge applications, provisional
selection, operational observability, and the independent researcher-learning
ledger. It never extracts edges from legacy free text and never repairs on read.
Relation projection prevents a competing Relation truth. Search lifecycle
dispositions and epistemic dispositions are distinct, so non-selection, low
Judge score, merge, and budget exhaustion cannot silently become contradiction.

The terminal handoff describes provisional-selected node claims because the
current App-native slice has no final Synthesis episode. Its model-profile and
support-source comparisons are correlated-error risk indicators, not correctness
or scientific reliability scores. Researcher learning is append-only feedback:
main-agent inferences remain unconfirmed, explicit user confirmation creates a
new record, and no learning append feeds graph, Judge, Relation, allocation, or
stopping state.

## Seed architecture and the Explorer role

A mandatory physical Explorer is removed from the target real-run architecture.

Exploration remains necessary, but it becomes the responsibility of a bounded Seed Episode:

```text
problem + constraints
      |
      v
Seed Episode
  - may self-organize internal exploration
  - returns 3-5 materially distinct, unranked SearchNodes
      |
      v
DTE validation / duplicate checks
      |
      v
Judge + geometry + allocation
```

The fixed direct/counterexample/formalism/merge seed templates remain useful for smoke tests, deterministic fallback, and regression testing.

Removing the physical Explorer does not permit the native runtime to collapse alternatives or return one preferred answer. Seed obligations include diversity, boundary cases, uncertainty preservation, and no self-ranking.

## Relation architecture

Relation is semantic graph maintenance, not an evaluator or verifier. It never returns correctness, pass/fail, reward, or certified-node state; backend validation checks schema, identity, revision, lifecycle, authority, and atomicity rather than scientific truth.

Candidate selection may use:

- exact normalized duplicates;
- embedding proximity;
- near-tied Judge/UCB values;
- explicit contradictory claims;
- entropy plateau.

Relation output is one of:

```text
equivalent
complementary
conflict
independent
```

The backend converts validated output into a merge proposal or persisted discriminator-task proposal. The model never applies the merge directly. Discriminator proposals are not executed in this slice and have no authority to close, reward, reject, certify, or select nodes.

Relation is not a universal synchronous barrier. Exact duplicates may be handled immediately; ordinary proximity creates optional or high-priority tasks. Only unresolved material conflicts among branches selected for synthesis must be resolved or explicitly disclosed.

In the App-native path, Relation is scheduled only after backend provisional Synthesis selection and before a new terminal action is committed. Blocking inventory generation completely enumerates selected-selected exact duplicates and shared-evidence divergent claims over the at-most-eight-node provisional set, so it has a hard upper bound of 28 pairs. These blockers are refreshed into the persistent candidate ledger before readiness and are never truncated by the separate enrichment window. Existing persisted terminal runs remain sticky and are reported as legacy-unchecked rather than reopened.

After the complete blocking inventory resolves, high-priority selected or directly selected-related semantic pairs may be scheduled as nonblocking enrichment. Current candidate/record identities are removed before the enrichment window is truncated. Enrichment can therefore progress past previously seen pairs without becoming a whole-graph all-pairs pass.

One App-native Relation episode contains only node-disjoint candidate pairs, for both blocking and enrichment grants. The request builder and commit boundary reject overlap, and merge provenance permits only one canonical target for each absorbed node. This is a transactional merge-safety invariant, not a verification rule.

Equivalent classification does not give the model merge authority. The backend selects a canonical node from committed status, information/evidence completeness, Judge value, provenance stability, and a node-ID tie-break; absorbed nodes remain auditable aliases and cannot receive future Executor allocation or be double-counted by Synthesis selection.

## Budget architecture

DTE separates two budget layers.

### Epistemic budget

Owned by DTE:

```text
iterations
soft allocation mass per iteration
hard committed-child cap per iteration
frontier selection
DTE graph expansion
```

The default intended semantics are:

```text
allocation_mass_per_iteration = 3
max_children_per_iteration = 5
max_relation_pairs_per_episode = 3
max_relation_enrichment_pairs = 3
```

The continuous Boltzmann mass is discretized into children and may realize more than three children, but never more than the hard cap. `max_relation_enrichment_pairs` is a run-level successful-pair budget reconstructed from the persistent Relation ledger; blocking work and failed, cancelled, expired, or retried-uncommitted attempts do not consume it.

### Compute budget

Bounded by run policy and used by the native runtime:

```text
internal subagent count
parallel threads
model and reasoning profile
retries
tool calls
wall-clock or token envelope
```

Compute budget cannot create additional DTE graph children beyond the episode grant.

## Reliability infrastructure

Reliability work is justified to support long-running, resumable, self-organized episodes, not to turn DTE into an evaluator-first optimizer.

Minimum target infrastructure:

- graph revision and parent revision;
- stale-result rejection;
- role-specific schemas and guards;
- one atomic commit boundary;
- minimal JSONL or SQLite event log;
- transport-neutral episode adapter;
- cache namespaces that include provider/model/rubric/prompt/schema identity;
- command-adapter fallback during SDK/App Server rollout.

Initial event types may include:

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

Do not begin with a large workflow framework, cryptographic event sourcing, or a distributed scheduler.

## Optional evaluators

Symbolic, numerical, executable, bibliographic, or formal evaluators are useful when a local subclaim is verifiable.

They provide evidence, not the global objective:

```text
DTE research potential != evaluator correctness metric
```

DTE should preserve separate concepts for research potential, evidence strength, and epistemic uncertainty. Easy-to-measure branches must not automatically dominate high-potential conceptual branches.

## Migration plan

### Phase 0: specification correction

- clarify soft allocation mass versus hard child cap;
- use round-half-up semantics where intended;
- restore Relation as a conditional semantic oracle;
- state explicitly that evaluator evidence is optional;
- state that DTE, not Ultra, is the root controller.

### Phase 1: hard episode boundary

- add graph/parent revisions to requests and results;
- reject stale results and ID collisions;
- centralize graph mutation in one commit function;
- add minimal episode event logging;
- preserve the existing command adapter as fallback.

### Phase 2: native runtime integration

- add SDK/App Server or hosted episode adapters;
- allow native self-organized subagents inside Seed and Executor episodes;
- keep Judge, Relation, and Synthesis logically isolated;
- treat descendant-thread inspection as optional observability, not correctness.

### Phase 3: controlled comparison

Compare:

```text
legacy explicit role calls
native guided episode orchestration
more autonomous native episode orchestration
```

Measure latency, quota, branch diversity, duplicate rate, counterexample coverage, schema violations, attempted bypasses, Judge survival, and human-rated research value.

## Explicit non-goals

Do not:

- make Ultra the root controller and rely on prompt compliance;
- make DTE explicitly schedule every physical subagent;
- require a fixed physical Explorer;
- merge Seed, Executor, Judge, Relation, and Synthesis into one implicit context;
- turn optional evaluators into the global reward;
- expose writable controller state to model episodes;
- accept direct final answers from Seed or Executor;
- depend permanently on `codex exec`;
- enable unbounded recursive subagent fan-out.

## Final architecture

```text
DTE Driver
  -> owns graph, epistemic budget, geometry, Relation, stopping, synthesis
  -> issues bounded role episode contracts

Native Codex/Ultra runtime
  -> owns internal within-episode orchestration
  -> may use self-organized subagents
  -> returns only bounded structured outputs

Guards + commit boundary
  -> validate the protocol boundary
  -> prevent model episodes from bypassing DTE
```
