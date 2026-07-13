# P1/P2 Vertical Slice Specification: Native Ultra AgentEpisode

Status: implementation-ready specification  
Target branch: `main`  
Related issue: #2  
Depends on: P0/P0.5 controller and operator boundaries merged through PR #3

## 1. Purpose

The next implementation phase must make DTE usable with Codex Ultra's native self-organization without either of the two failure modes below:

1. DTE micromanages every physical subagent call and duplicates the native orchestrator.
2. Ultra becomes the outer controller and can bypass DTE allocation, graph mutation, stopping, or synthesis.

The target boundary is:

```text
User
  -> delegates operation to the main agent
Main agent / operator proxy
  -> starts and supervises DTE through validated controller commands
DTE controller
  -> grants one bounded role episode
EpisodeRequest
  -> native Ultra runtime
Ultra runtime
  -> chooses its own internal reasoning, tools, parallelism, and subagents
EpisodeResult
  -> schema validation + revision checks + single commit boundary
DTE graph
```

DTE owns cross-episode epistemic recursion. Ultra owns bounded within-episode orchestration.

This specification defines the smallest useful vertical slice that proves that boundary and records enough telemetry to compare it against the legacy explicit-call workflow.

## 2. Priority order

For this personal research project, implementation priorities are:

```text
native Ultra adaptation
> measurable performance / cost / quality
> minimum correctness boundary
> production reliability infrastructure
```

The implementation must not expand into enterprise authorization, distributed scheduling, high availability, or cryptographic identity work.

Minimum correctness is still required where it affects the validity of the experiment:

- structured request and result contracts;
- no writable controller-owned fields;
- stale-result rejection;
- one graph commit function;
- bounded outputs;
- role separation;
- minimal run and episode telemetry.

## 3. Normative ownership

### 3.1 Main agent

The main agent is the user's authorized operator proxy. Subject to `OperatorPolicy`, it may:

- create or select a RunSpec;
- start and supervise DTE;
- inspect checkpoints and telemetry;
- retry a failed runtime call;
- issue supported controller commands;
- request synthesis when authorized;
- report progress and ask the user for decisions when needed.

It may not directly:

- mutate the graph;
- write Judge scores;
- write embeddings, density, entropy, uncertainty, or UCB;
- assign expansion budgets;
- apply Relation or merge results;
- claim algorithmic convergence;
- commit final synthesis.

### 3.2 DTE controller

Only the DTE backend may:

- own graph and node revisions;
- select the role and input nodes for an episode;
- set `max_returned_children`;
- validate episode output;
- reject stale or malformed output;
- compute Judge/embedding/controller fields;
- commit accepted graph changes;
- decide the next controller step;
- select the synthesis checkpoint.

### 3.3 Native Ultra runtime

Within one granted episode, Ultra may:

- reason autonomously;
- decide whether subagents are useful;
- choose internal roles and work decomposition;
- run bounded parallel work;
- use allowed tools;
- research, calculate, code, test, critique, and aggregate;
- return several candidate SearchNodes when granted.

Ultra must not:

- act as the DTE outer controller;
- create additional DTE iterations;
- exceed the granted output count;
- mutate graph storage or controller state;
- fill controller-owned fields;
- decide global allocation or stopping;
- return an Executor or Seed answer as final synthesis.

DTE must not require a fixed physical topology such as:

```text
explorer + critic + verifier
one High root + two Medium workers
minimum N subagents
```

The internal topology is an Ultra runtime implementation detail.

## 4. Scope of the first vertical slice

The first implementation must support one complete Executor path:

```text
committed parent node
  -> controller grants Executor EpisodeRequest
  -> native-capable AgentEpisodeAdapter executes it
  -> EpisodeResult returns zero or more child candidates
  -> backend validates the complete result
  -> commit_episode_result(...) atomically adds accepted children
  -> telemetry records the episode and commit outcome
```

The contracts must be role-extensible so Seed, Judge, Relation, and Synthesis can reuse the same envelope later, but the first production path does not need to implement all roles simultaneously.

The existing command adapter remains a fallback and regression baseline.

## 5. Transport-neutral adapter

Define a stable adapter interface equivalent to:

```python
class AgentEpisodeAdapter(Protocol):
    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        ...
```

An asynchronous equivalent is allowed if it does not change protocol semantics.

Possible transports include:

- native Codex/Ultra hosted runtime;
- Codex SDK;
- Codex App Server;
- a local command/subprocess adapter;
- future provider-specific runtimes.

The protocol must not depend on one transport's thread IDs, response IDs, tool event format, or subagent trace representation.

Transport metadata may be stored for diagnostics, but it is not accepted as graph evidence.

## 6. EpisodeRequest schema

Implement a strict versioned model. Extra fields must be rejected unless explicitly namespaced as transport metadata.

Minimum envelope:

```text
episode_id: unique string
run_id: unique string
role: executor | seed | judge | relation | synthesis
input_graph_revision: integer
selected_node_revisions: map[node_id, revision]
objective: string
coverage_requirements: list[string]
allowed_output_types: list[string]
output_schema_version: string
native_orchestration_allowed: bool
runtime_limits: RuntimeLimits
tool_policy: optional ToolPolicy
transport_hints: optional map[string, JSON value]
```

Executor-specific fields:

```text
parent_node_id: string
parent_node_revision: integer
max_returned_children: integer >= 0
required_parent_id_on_children: true
```

`runtime_limits` should support only limits that can be measured or enforced by the available runtime:

```text
wall_clock_seconds: optional integer
max_retries: integer
max_parallelism_hint: optional integer
max_tool_calls_hint: optional integer
```

These are compute limits or hints. They do not alter DTE's epistemic child grant.

`tool_policy` may declare:

```text
network_allowed
shell_allowed
write_allowed
allowed_write_roots
```

The first implementation may support only a subset. Unsupported requested permissions must fail explicitly rather than silently broadening access.

## 7. EpisodeResult schema

Minimum envelope:

```text
episode_id: string
run_id: string
role: same as request
input_graph_revision: integer
selected_node_revisions: same revisions observed by the episode
status: completed | failed | timed_out | cancelled
structured_output: role-specific object or null
runtime_diagnostics: RuntimeDiagnostics
output_hash: string
schema_version: string
```

Executor structured output:

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
      "parent_ids": ["assigned-parent-id"],
      "confidence": 0.0,
      "status": "frontier"
    }
  ],
  "episode_summary": "short summary of work performed",
  "unresolved_questions": []
}
```

`episode_summary`, diagnostics, and internal-subagent summaries are observational metadata. They cannot mutate graph state.

The result must never include controller-owned values such as:

```text
embedding
local_embedding
score
judge verdict assignment
uncertainty
ucb_score
expansion_budget
allocation
committed graph revision
graph status other than the allowed candidate status
stop reason
synthesis checkpoint
```

## 8. Validation and commit boundary

All successful graph mutation must pass through one backend function conceptually equivalent to:

```python
commit_episode_result(
    graph,
    request,
    result,
) -> CommitOutcome
```

The function must validate the entire result before mutating graph state.

For an Executor result, reject the whole result when any of the following is true:

- request/result episode IDs differ;
- request/result roles differ;
- graph revision is stale;
- assigned parent revision is stale;
- a returned node omits the assigned parent;
- a returned node uses a forbidden status or node type;
- a returned node pre-fills a controller-owned field;
- the returned child count exceeds the grant;
- a node ID collides with an already committed node;
- duplicate node IDs exist inside the result;
- schema or output hash validation fails;
- result status is not `completed`;
- output is partial, malformed, timed out, or cancelled.

Rejection semantics:

```text
graph unchanged
no partial child commit
explicit rejection reason
telemetry event written
retry policy decided outside the commit function
```

The first implementation may use copy-validate-replace or an equivalent simple atomic pattern. It does not require a database transaction framework.

## 9. Native Ultra integration semantics

The first native adapter must preserve the following semantics regardless of the concrete runtime API:

1. DTE sends one bounded request rather than manually replaying a subagent tree.
2. Ultra receives the objective, selected node context, coverage obligations, output limit, and role schema.
3. Ultra may internally self-organize.
4. DTE does not inspect or enforce the number or names of internal agents.
5. The adapter returns one complete `EpisodeResult`.
6. Only the backend validator and commit boundary can affect graph state.

The adapter should prefer low-I/O execution:

```text
one request envelope
zero or a small number of progress events
one complete result envelope
```

Do not require the DTE controller to poll or serialize every internal subagent message.

If the current Codex environment does not expose a stable native Ultra transport, implement:

- the transport-neutral interface and schemas;
- a deterministic fake/native-stub adapter for tests;
- the existing command adapter behind the same interface;
- a clearly isolated placeholder for the first true native adapter.

Do not invent undocumented SDK methods or claim native Ultra support when only a subprocess fallback exists.

## 10. Role isolation

Logical roles remain separate contracts even when executed by the same Ultra runtime at different times.

The first Executor episode must not receive:

- its Judge result as an instruction to justify that score;
- writable controller state;
- hidden allocation internals not needed for its task;
- a synthesis instruction.

Future Judge, Relation, and Synthesis requests must use distinct role prompts and output schemas.

Internal role diversity inside one Executor episode is allowed. It does not replace DTE-level logical role isolation.

## 11. Minimal telemetry

Every run and episode must emit append-only JSONL or equivalent simple structured records.

The minimum event types for this vertical slice are:

```text
run_created
episode_granted
episode_started
episode_completed
episode_failed
output_rejected
nodes_committed
run_completed
```

Each event should include when available:

```text
timestamp
run_id
episode_id
role
adapter_name
transport_name
model/runtime profile
wall_clock_ms
queue_or_io_ms
retry_count
status
input_graph_revision
returned_node_count
accepted_node_count
rejection_reason
```

Usage fields must distinguish source and uncertainty:

```text
input_tokens
output_tokens
cached_tokens
provider_reported_cost
estimated_cost
quota_delta
usage_source: provider_reported | estimated | unavailable
```

Do not fabricate precise token, cost, or quota figures when the runtime does not expose them.

Quality-oriented fields for Executor episodes should include:

```text
schema_valid
controller_field_violation_count
duplicate_within_result_count
accepted_node_count
later_judge_survival_count (may be filled later)
later_relation_outcome (may be filled later)
```

Telemetry is observational. It does not change UCB or allocation in this phase.

## 12. Comparison profiles

The implementation must preserve the ability to compare:

```text
A. legacy explicit command/role calls
B. native guided AgentEpisode
C. more autonomous native AgentEpisode
```

Profiles B and C differ in within-episode freedom, not in DTE ownership.

Example distinction:

- guided: stronger coverage requirements and narrower tool/runtime hints;
- autonomous: broader decomposition freedom inside the same output and budget boundary.

All profiles must use the same DTE child grants, graph semantics, validation, and final synthesis rules.

Minimum comparison metrics:

- wall-clock and I/O latency;
- provider-reported or estimated usage;
- accepted SearchNodes per episode;
- duplicate rate;
- counterexample/boundary-case coverage;
- schema and controller-field violations;
- Judge survival of committed nodes;
- human-rated research value;
- cost or quota per later-surviving node.

Repeated runs are required before drawing conclusions about architecture quality.

## 13. Implementation increments

### Increment 1: contract and commit boundary

- add `EpisodeRequest`, `EpisodeResult`, `RuntimeLimits`, and diagnostics models;
- add role-specific Executor payload models;
- add graph and parent revisions to the complete path;
- implement `commit_episode_result(...)`;
- add stale, collision, count, parent, status, and controller-field checks;
- add deterministic unit tests.

### Increment 2: adapter unification

- define `AgentEpisodeAdapter`;
- wrap the existing command adapter behind it;
- add a deterministic fake/native-stub adapter;
- keep existing production behavior working.

### Increment 3: first native Ultra adapter

- inspect the actually available Codex runtime interface;
- implement the native adapter only against a real supported interface;
- allow internal native self-organization;
- return one complete result envelope;
- retain command fallback.

### Increment 4: telemetry and comparison harness

- write minimal append-only events;
- capture runtime usage when available;
- add profile metadata;
- provide a small repeatable comparison runner or documented procedure.

The implementation may combine increments in one PR only if review remains tractable. The preferred order is contract/commit first, then runtime transport.

## 14. Acceptance criteria

The vertical slice is complete when all of the following hold:

1. A committed parent can be expanded through `AgentEpisodeAdapter`.
2. The adapter may represent a native self-organized runtime without exposing internal topology to DTE.
3. A valid completed result commits no more than the granted number of children.
4. A stale, malformed, failed, timed-out, colliding, over-budget, or controller-field-polluted result leaves graph state unchanged.
5. All accepted graph mutation passes through one commit function.
6. The existing command path remains usable as fallback.
7. Tests do not require a production Ultra credential or live Gemini key.
8. The code does not claim native Ultra support unless a real supported transport is exercised.
9. Minimal episode telemetry is produced.
10. The implementation can identify which comparison profile produced a run.
11. DTE still owns allocation, stopping, Relation application, and final synthesis.
12. No fixed High/Medium configuration or fixed physical subagent topology is encoded in the protocol.

## 15. Required tests

At minimum, add tests for:

- request/result schema strictness;
- valid Executor result commit;
- zero-child valid result;
- returned child count over grant;
- stale graph revision;
- stale parent revision;
- existing node ID collision;
- duplicate IDs inside one result;
- missing assigned parent;
- forbidden synthesis node;
- forbidden controller-owned fields, including `local_embedding`;
- failed/timed-out/cancelled result;
- graph unchanged for every rejection path;
- event emission for grant, completion, rejection, and commit;
- command adapter and fake/native-stub conformance to the same protocol;
- internal subagent metadata ignored as graph facts;
- no requirement for a minimum subagent count.

A live native-runtime smoke test may be optional and locally gated. It must not be required in ordinary CI when credentials or the native runtime are unavailable.

## 16. Deferred work

This phase deliberately defers:

- complete RBAC or cryptographic actor identity;
- OAuth/capability token systems;
- distributed locks or schedulers;
- high availability;
- full event sourcing;
- a large observability dashboard;
- automatic telemetry-driven modification of UCB;
- unbounded recursive native-agent fan-out;
- a permanent dependency on one Codex transport;
- mandatory live Gemini or Ultra integration in CI.

## 17. Open implementation questions

These questions should be resolved by repository and runtime inspection rather than assumption:

1. Which stable native Codex/Ultra interface is actually available in the target environment: hosted Work runtime, SDK, App Server, or another bridge?
2. Which usage fields are exposed reliably by that interface?
3. Can cancellation and wall-clock limits be enforced, or only recorded?
4. Can native runtime progress be observed without serializing every internal event?

None of these questions changes the protocol. When unavailable, the implementation must preserve the transport-neutral boundary and use the command fallback honestly.

## 18. Implementation status (Executor vertical slice)

Status as of this implementation:

| Increment | Status | Implemented fact / remaining boundary |
| --- | --- | --- |
| Increment 1: contract and commit | **implemented** | Strict versioned request/result, Executor payload/output, runtime/tool/diagnostic models, graph and parent revisions, canonical output hash, atomic `commit_episode_result(...)`, and rejection telemetry. |
| Increment 2: adapter unification | **implemented** | Existing subprocess/command Executor is bridged behind `AgentEpisodeAdapter`; deterministic native-shaped stub uses the same interface; existing `strict-run` behavior is preserved. |
| Increment 3: App-native driver protocol | **implemented** | Persistent `create-run`, `next-episode`, `submit-episode-result`, `fail-episode`, `cancel-episode`, `retry-episode`, `request-synthesis`, and `run-status` let the current Codex App main agent perform opaque native work without repository-spawned Codex. SDK/App Server transport is deferred. |
| Increment 4: telemetry | **implemented** | Append-only `<run-dir>/episode_events.jsonl` records run/attempt lifecycle, submission, commit/rejection, quality counters, profile, and `usage_source=unavailable` without fabricated App usage or hidden topology. Later Judge survival and Relation outcomes remain deferred fields. |

The subsequent App-native progression slice implements strict Judge episodes, atomic observable-judgment commit, and backend-only geometry/entropy/UCB/allocation progression into the existing Executor grant. Deferred beyond these slices: native Seed, Relation, and final Synthesis episodes, full production role closure, SDK/App Server transports, visibility into hidden App subagents, telemetry-driven controller tuning, and distributed revisions. The comparison profiles alter only within-episode guidance; they do not change graph semantics or require a minimum subagent count. The normative first-production interpretation is `p1-native-ultra-agentepisode-codex-app-profile.md`.
