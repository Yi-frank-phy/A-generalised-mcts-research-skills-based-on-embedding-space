# Normative Implementation Profile: Codex App Native AgentEpisode

Status: implementation-ready normative profile  
Parent specification: `docs/specs/p1-native-ultra-agentepisode.md`  
Related issue: #2

This profile resolves the first implementation surface for the native Ultra AgentEpisode vertical slice.

Where this file is more specific than the parent specification, this file governs the first Codex implementation.

## 1. First production surface

The first production surface is the Codex App / Work session itself.

The implementation must not treat the repository as responsible for spawning a second Codex process through `codex exec` in order to simulate native orchestration.

The physical execution model is:

```text
User
  -> delegates operation to the Codex App main agent
Codex App main agent
  -> acts as the user-authorized operator proxy and runtime driver
  -> asks the DTE backend for the next bounded episode contract
DTE backend
  -> selects role, graph revision, parent revision, obligations, and output grant
EpisodeRequest
  -> returned to the Codex App main agent
Codex App main agent
  -> uses the App's native reasoning, tools, and internal subagents
  -> internal topology remains opaque to repository code
EpisodeResult
  -> submitted back through a validated backend command
DTE backend
  -> validates, rejects or commits, and selects the next controller step
```

The DTE backend is the logical outer epistemic controller.

The Codex App main agent is the physical runtime driver and the user's authorized operator proxy.

These are not contradictory roles:

```text
DTE decides what bounded epistemic work is granted and what may enter the graph.
The App main agent decides how to perform the granted work inside the native runtime.
```

## 2. No repository-spawned Codex orchestration

The first implementation must not require:

```text
DTE backend -> codex exec -> external child Codex process
```

as the normal App integration path.

A subprocess adapter may remain for regression tests, headless environments, or future automation, but it is not the target architecture for seamless Codex App use.

The repository must not claim that external CLI subprocesses expose or reproduce the App's internal Ultra orchestration.

Do not add a fake `--ultra` flag, hard-coded model topology, or repeated fixed-role subprocess calls.

## 3. App-native driver protocol

The first implementation must expose a resumable backend protocol that the App main agent can drive explicitly.

A suitable command surface is conceptually equivalent to:

```text
create-run
next-episode
submit-episode-result
fail-episode
cancel-episode
retry-episode
request-synthesis
run-status
```

Exact CLI names may follow the repository's existing command style, but the semantics are normative.

### 3.1 `next-episode`

The backend:

- reads the committed DTE state;
- performs any deterministic controller work that must precede a model episode;
- selects the next logical role;
- selects the required graph and node revisions;
- writes or returns exactly one versioned `EpisodeRequest`;
- records `episode_granted`;
- does not start a Codex subprocess;
- does not prescribe the internal subagent topology.

If no model episode is currently required, it returns an explicit controller action such as:

```text
continue_controller
await_operator_decision
ready_for_synthesis
run_complete
```

### 3.2 native App execution

The Codex App main agent receives the complete `EpisodeRequest` and performs the episode using the native App runtime.

The main agent may:

- reason directly;
- use the App's tools;
- delegate independent work to native subagents;
- choose internal roles and parallelism;
- inspect the permitted repository or external sources;
- aggregate the work into one bounded structured result.

Repository code is not expected to observe:

- internal subagent count;
- internal agent names;
- routing decisions;
- descendant transcripts;
- per-subagent latency;
- per-subagent token or quota use;
- hidden reasoning.

The absence of this information is normal and must not be treated as an integration failure.

### 3.3 `submit-episode-result`

The App main agent submits one complete `EpisodeResult` through a backend command or equivalent validated API.

The backend:

- parses the strict schema;
- verifies run, episode, role, graph revision, and node revisions;
- validates output count, node ancestry, IDs, status, and controller-owned fields;
- verifies the canonical output hash when used;
- rejects the whole result on any violation;
- commits only through `commit_episode_result(...)`;
- records the result and commit outcome;
- advances the controller state only after successful validation.

Natural-language chat output is not graph input.

A Markdown summary shown to the user is not graph input.

Only the submitted structured result may be considered for commit.

## 4. Main-agent authority and limits

The main agent is the user's authorized operator proxy.

Subject to validated RunSpec and `OperatorPolicy`, it may:

- start and supervise runs;
- request the next episode;
- perform the granted episode;
- use native subagents;
- submit a complete result;
- retry failed work;
- cancel an episode;
- change permitted runtime limits;
- request synthesis;
- explain progress and ask the user for decisions.

It may not directly:

- mutate graph storage;
- write Judge scores;
- write embeddings, density, entropy, uncertainty, UCB, or allocation;
- change committed node or graph revisions;
- apply Relation or merge results;
- mark an episode committed without backend validation;
- convert a cancelled, expired, malformed, or failed episode into success;
- claim algorithmic convergence;
- commit final synthesis.

The correct rule is:

```text
main agent may issue privileged controller commands
main agent may not directly mutate controller-owned state
```

## 5. Episode lifecycle and revisions

The backend must persist a minimal episode record containing at least:

```text
episode_id
run_id
role
status
input_graph_revision
selected_node_revisions
parent_node_revision where applicable
grant timestamp
expiry or deadline when configured
attempt count
max returned children
result hash when submitted
commit outcome
```

Recommended statuses:

```text
granted
in_progress
completed_uncommitted
committed
rejected
failed
cancelled
expired
superseded
```

Only one active attempt may be accepted for an episode unless the backend has explicitly issued a retry attempt.

A late result from a cancelled, expired, superseded, or already committed attempt must be rejected even when its JSON is otherwise valid.

Stale graph or parent revisions reject the complete result and leave graph state unchanged.

## 6. Timeout, cancellation, and retry

The Codex App main agent commonly selects or adjusts timeout, cancellation, and retry policy on behalf of the user.

The backend must record the policy source:

```text
selected_by: user | main_agent | run_default
```

Because the App's internal subagent lifecycle may be opaque to repository code, enforcement is split:

```text
App main agent
  -> supervises and cancels native App work when possible
DTE backend
  -> records deadline/cancellation state
  -> refuses late or cancelled results
  -> issues a new attempt only through an explicit retry transition
```

The backend does not need to kill invisible App subagents directly.

It must still enforce acceptance semantics:

- an expired attempt cannot commit;
- a cancelled attempt cannot commit;
- a superseded attempt cannot commit;
- a failed attempt cannot commit;
- a retry receives a new attempt identifier;
- at most one validated attempt is committed.

Partial chat output, partial files, or unfinished subagent summaries must not be submitted as a successful result.

## 7. Low-I/O execution

The default App integration is:

```text
one next-episode request
one bounded native App episode
zero or a few user-facing progress updates
one submit-episode-result call
```

The backend must not:

- poll one file per internal subagent;
- require every internal tool or agent event;
- serialize hidden reasoning;
- reconstruct the native agent tree;
- require synchronized descendant transcripts.

The App main agent may provide ordinary conversational progress to the user. Those updates are not DTE graph facts.

Backend telemetry should remain coarse-grained.

## 8. Usage and cost visibility

Assume the App may not expose reliable episode-level values for:

```text
input tokens
output tokens
cached tokens
provider cost
quota delta
internal subagent count
per-subagent latency
per-subagent usage
```

When unavailable, record explicit nulls and:

```text
usage_source = unavailable
```

Do not estimate tokens from character count or infer precise episode usage from account-level quota movement.

The minimum directly observable telemetry is:

```text
grant timestamp
submission timestamp
wall_clock_ms between grant and submission
attempt count
role
profile name when explicitly known
returned node count
accepted node count
schema validity
rejection reason
commit outcome
```

Optional user- or platform-observed quota information may be attached as:

```text
external_usage_observation
```

It must be labelled as external, approximate, and not episode-ground-truth.

Quality telemetry remains required even when usage telemetry is unavailable.

## 9. Structured request/result artifacts

The backend should write stable machine-readable artifacts such as:

```text
<run-dir>/episodes/<episode-id>/request.json
<run-dir>/episodes/<episode-id>/result.json
<run-dir>/episodes/<episode-id>/status.json
<run-dir>/episode_events.jsonl
```

Equivalent stdout/stdin command payloads are allowed, but files are useful for App continuity, review, recovery, and compaction.

The App main agent may read `request.json` and write a candidate `result.json`, but graph mutation occurs only when the backend validates the result through the submit command.

The result artifact must not itself be treated as committed state.

App-native Relation adds stable versioned epistemic artifacts:

```text
<run-dir>/relations/candidates.json
<run-dir>/relations/relation_ledger.json
<run-dir>/relations/synthesis_readiness.json
```

They mirror controller-owned persistent state and survive restart. Writing an episode `result.json` alone cannot modify them.

## 10. Skill and prompt integration

`SKILL.md`, `AGENTS.md`, and relevant runtime guidance must teach the Codex App main agent to follow this loop:

```text
1. start or resume DTE run
2. call next-episode
3. read the complete EpisodeRequest
4. perform the episode using native App orchestration
5. construct only the required structured EpisodeResult
6. call submit-episode-result
7. inspect the backend outcome
8. repeat until backend requests synthesis, operator decision, or completion
```

The instructions must explicitly state:

- do not manually choose the global next branch;
- do not invent allocation;
- do not write controller-owned fields;
- do not bypass submission validation;
- do not replace DTE synthesis with a direct chat answer;
- native internal delegation is allowed and need not be exposed;
- keep App-side progress concise and do not serialize internal traces.

## 11. First vertical-slice scope

The first production vertical slice must support one Executor path:

```text
committed parent
  -> backend next-episode
  -> Executor EpisodeRequest
  -> Codex App native episode
  -> Executor EpisodeResult
  -> submit-episode-result
  -> complete validation
  -> commit_episode_result(...)
  -> child nodes committed or whole result rejected
```

The envelope must remain role-extensible for Seed, Judge, Relation, and Synthesis.

The first PR does not need to implement all native role loops at once.

## 12. Required implementation components

At minimum implement:

1. strict `EpisodeRequest` and `EpisodeResult` models;
2. graph and node revision fields;
3. persistent episode lifecycle records;
4. a `next-episode` backend operation;
5. a `submit-episode-result` backend operation;
6. `fail`, `cancel`, and `retry` transitions or equivalent commands;
7. a single `commit_episode_result(...)` graph mutation boundary;
8. stale, cancelled, expired, superseded, collision, over-grant, ancestry, status, and controller-field rejection;
9. coarse append-only telemetry;
10. Codex App skill instructions for the native driver loop;
11. deterministic tests that do not require live App internals.

## 13. Required tests

Add tests for at least:

- strict request/result schemas;
- next-episode creates one bounded grant;
- valid Executor result commit;
- valid zero-child result;
- stale graph revision;
- stale parent revision;
- returned children over grant;
- committed node ID collision;
- duplicate IDs inside one result;
- missing assigned parent;
- forbidden synthesis node;
- every controller-owned field, including `local_embedding`;
- failed result rejection;
- cancelled result rejection;
- expired result rejection;
- superseded attempt rejection;
- retry creates a distinct attempt;
- only one attempt can commit;
- graph unchanged on every rejection path;
- coarse telemetry emission;
- request/result artifacts are not themselves graph state;
- no requirement for subagent count, names, or traces;
- skill instructions preserve DTE controller ownership;
- existing P0 allocation, cache, and OperatorPolicy tests remain green.

Ordinary CI must not require access to hidden Codex App subagent state.

## 14. Acceptance criteria

The first App-native vertical slice is complete when:

1. A Codex App main agent can request the next DTE episode without spawning another Codex process.
2. The backend returns a strict bounded Executor request.
3. The App main agent can perform the episode with opaque native subagents.
4. The main agent can submit one strict result through the backend.
5. Valid output commits only through `commit_episode_result(...)`.
6. Invalid, stale, cancelled, expired, superseded, or over-grant output leaves graph state unchanged.
7. The backend, not the main agent, selects the next DTE controller step.
8. The App main agent retains authorized operator commands without direct state mutation.
9. No internal subagent trace or usage disclosure is required.
10. The normal path does not invoke `codex exec` to simulate App orchestration.
11. Coarse telemetry and episode artifacts are produced.
12. The existing strict-run and regression tests remain functional or are migrated with an explicit compatibility path.

## 15. Deferred work

This phase deliberately defers:

- programmatic access to hidden App subagent traces;
- precise per-episode token, cost, or quota accounting when not exposed;
- SDK/App Server transports;
- headless CLI orchestration as the primary path;
- complete native loops for every logical role;
- distributed schedulers and locks;
- cryptographic actor identity;
- large dashboards;
- automatic telemetry-driven UCB changes;
- unbounded recursive native-agent fan-out.

## 16. Correct interpretation

The target is not:

```text
Python DTE launches Codex CLI workers and reconstructs Ultra externally.
```

The target is:

```text
Codex App main agent runs the native opaque orchestration.
DTE backend supplies and validates the bounded epistemic protocol.
```

This is the required meaning of seamless Ultra-mode adaptation for the first implementation.

## 17. Implemented vertical-slice status

The App-native Judge → controller → Executor → Relation/readiness slice now implements:

- persistent `create-run`, `next-episode`, `submit-episode-result`, `fail-episode`, `cancel-episode`, `retry-episode`, `request-synthesis`, and `run-status` operations;
- strict request/result envelopes with distinct `episode_id` and `attempt_id`;
- persistent request, result, status, graph, node-revision, deadline, retry, and commit-outcome records;
- App-main-agent grants that never launch a Codex subprocess;
- rejection of stale, failed, cancelled, expired, superseded, rejected, or already committed attempts;
- a single `commit_episode_result(...)` graph mutation boundary;
- strict versioned Judge payload/observation schemas and exact-grant atomic Judge commits;
- backend-only deterministic embedding/KDE, entropy, uncertainty, UCB, and allocation progression after Judge commit;
- seamless progression from an ordinary unscored frontier to a bounded Executor grant without a main-agent `continue_controller` decision;
- sticky terminal controller actions and iteration-cap enforcement before new Judge grants, while preserving already-committed positive Executor allocations;
- run-scoped App embedding persistence at `<run-dir>/dte_cache.json` using the existing provider/model/dimension/contract-version namespace;
- coarse append-only telemetry with App usage marked `unavailable`;
- Skill and `AGENTS.md` instructions for the current-App loop.
- deterministic provisional synthesis branch selection from committed non-merged nodes;
- complete selected-set blocking Relation inventory (at most 28 pairs), never truncated by enrichment windows;
- bounded `role=relation` requests with canonical candidate pairs and independent `max_relation_pairs_per_episode`;
- ledger-aware high-priority Relation enrichment with run-level `max_relation_enrichment_pairs=3` successful-pair budget;
- node-disjoint blocking and enrichment grants within every Relation episode, defended again at request and commit boundaries;
- strict equivalent/complementary/conflict/independent observations committed through `commit_episode_result(...)`;
- persistent candidate, Relation, merge-application, and Synthesis-readiness records;
- backend-only canonical equivalent merge with source-node provenance preservation;
- readiness invariants requiring complete inventory and zero unresolved blocking pairs, while enrichment remains logically nonblocking;
- explicit material-conflict disclosure obligations because full discriminator Executor scheduling remains deferred;
- Relation gating before a new sticky terminal action, without reopening legacy persisted terminal runs or incrementing controller search iteration;
- semantic-only Relation outputs; discriminator proposals remain persisted and unexecuted, with no verifier/correctness/pass-fail loop.

Node-disjoint Relation batching and single-canonical absorbed-node provenance are transactional merge-safety invariants, not verification rules.

The command/subprocess adapter remains only a legacy/headless fallback and regression baseline. SDK/App Server transport, hidden App-subagent inspection, native Seed and final Synthesis episodes, any future discriminator research-task scheduling, full production role closure, and precise App token/quota telemetry remain deferred. A future discriminator would remain evidence-producing research work, not a correctness-certifying verifier.
