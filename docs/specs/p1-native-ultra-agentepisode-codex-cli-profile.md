# Normative Implementation Profile: Codex CLI First

Status: implementation-ready normative profile  
Parent specification: `docs/specs/p1-native-ultra-agentepisode.md`  
Related issue: #2

This profile resolves the parent specification's open transport questions for the first implementation. Where this file is more specific than the parent specification, this file governs the first Codex implementation.

## 1. First production transport

The first real `AgentEpisodeAdapter` must target the Codex CLI.

```text
CodexCliEpisodeAdapter
    EpisodeRequest
        -> codex exec
        -> schema-constrained final output
    EpisodeResult
```

The implementation must not wait for a Codex SDK, App Server, hosted bridge, or another provider transport before completing the vertical slice.

The generic `AgentEpisodeAdapter` boundary remains transport-neutral so additional transports can be added later, but the implementation order is:

```text
1. Codex CLI
2. other transports only when they provide a concrete advantage
```

For this phase, the Codex CLI adapter is the primary native-capable adapter, not merely a fallback. The existing narrower command adapters may remain as regression paths or be wrapped by the same common subprocess machinery.

## 2. Required CLI execution shape

Use the actual installed Codex CLI and inspect `codex exec --help` before fixing flags. The intended non-interactive shape is:

```text
EpisodeRequest
    -> canonical prompt on stdin
    -> codex exec ... --output-schema <role-schema.json> -
    -> optional JSONL lifecycle stream
    -> one schema-valid final role output
    -> EpisodeResult
```

The adapter should use supported CLI capabilities where available:

- prompt input from stdin;
- `--output-schema` for the role-specific final response;
- `--json` for machine-readable lifecycle events when useful;
- `--output-last-message` or an equivalent supported final-output path when it simplifies reliable capture;
- `--profile`, `--model`, or supported config overrides only when selected by RunSpec/operator configuration;
- an explicit sandbox and writable-root policy.

Do not invent an undocumented `--ultra` flag or provider-specific method. Ultra/model/reasoning selection belongs to the user's or main agent's Codex configuration and the validated run/runtime profile. The DTE protocol must not hard-code a High/Medium topology or a particular internal subagent layout.

## 3. Native orchestration is intentionally opaque

Codex may self-organize subagents internally. The CLI adapter is not required to expose:

- internal subagent count;
- internal agent names or roles;
- descendant transcripts;
- per-subagent token use;
- per-subagent latency;
- internal routing decisions.

DTE judges success only from the bounded request, final structured result, exit status, revision checks, and commit outcome.

The adapter must not reconstruct or micromanage Codex's internal orchestration by repeatedly launching fixed Explorer/Critic/Verifier processes.

## 4. Usage and cost telemetry

Assume that reliable token, cache, cost, quota, and internal-subagent usage may not be exposed by the CLI subprocess interface.

Always record directly observable fields:

```text
wall_clock_ms
process_start_time
process_end_time
exit_code
termination_reason
retry_count
stdout_bytes
stderr_bytes
jsonl_event_count
returned_node_count
accepted_node_count
adapter_name = codex-cli
transport_name = codex-exec
runtime/model profile when explicitly known
```

For token/cost/quota fields:

```text
usage_source = provider_reported
```

only when the CLI returns documented, machine-readable values for that run.

Otherwise record:

```text
input_tokens = null
output_tokens = null
cached_tokens = null
provider_reported_cost = null
estimated_cost = null
quota_delta = null
usage_source = unavailable
```

Do not scrape interactive TUI display text, infer tokens from character counts, or manufacture quota precision. A later user-supplied account-level observation may be attached as external telemetry, but it is not episode-ground-truth.

Quality telemetry remains required even when cost telemetry is unavailable.

## 5. Timeout, cancellation, and retry ownership

The main agent is the user's authorized operator proxy. It commonly selects episode timeout, cancellation, and retry policy through a validated RunSpec or supported controller command.

The ownership split is:

```text
main agent / OperatorPolicy
    -> selects or changes allowed timeout, cancellation, and retry policy
DTE controller
    -> validates the policy and grants the bounded EpisodeRequest
Codex CLI adapter
    -> enforces the subprocess deadline/cancellation and reports the actual outcome
commit_episode_result(...)
    -> never commits failed, timed-out, or cancelled output
```

`RuntimeLimits` should therefore distinguish:

```text
wall_clock_seconds
max_retries
selected_by: user | main_agent | run_default
```

A main-agent change must go through the backend command/policy boundary. The main agent must not edit graph state or convert a killed process into a successful result.

When the deadline is reached, the adapter must terminate the CLI subprocess using the platform's normal process-control mechanism, wait for cleanup within a bounded grace period, and return:

```text
status = timed_out
```

When an authorized cancellation is received, return:

```text
status = cancelled
```

Partial stdout, partial JSONL, temporary files, or a last incomplete answer must not be accepted as an `EpisodeResult`.

Retries occur outside `commit_episode_result(...)`. Every attempt receives an attempt identifier and separate telemetry record. Only one completed, validated attempt may be committed.

## 6. Low-I/O progress policy

The default execution path is:

```text
one EpisodeRequest
zero or a small number of coarse progress updates
one complete EpisodeResult
```

When `codex exec --json` is used, the adapter may consume all JSONL events internally for process management, but it should persist or expose only coarse-grained events such as:

```text
episode_started
meaningful_progress (optional and rate-limited)
episode_completed
episode_failed
episode_timed_out
episode_cancelled
```

Do not write every internal reasoning, tool, or subagent event into the DTE event ledger. Raw CLI event streams may be retained only as optional debug artifacts, disabled by default and excluded from graph facts.

The DTE controller must never poll one file per internal subagent or require synchronized serialization of the native agent tree.

## 7. Structured output and prompt construction

The Codex CLI adapter must generate a role-specific JSON Schema from the repository's strict Pydantic model or use a checked-in equivalent schema.

The prompt sent to Codex must contain:

- episode role;
- objective;
- selected parent/node context;
- coverage requirements;
- allowed tools and write roots;
- maximum returned child count;
- explicit prohibition on controller-owned fields;
- instruction to return only the schema-constrained result.

The prompt must not prescribe a fixed physical subagent topology. It may state that native orchestration is allowed and that independent work may be parallelized.

The final output must be parsed into the role-specific structured output, wrapped in `EpisodeResult`, hashed canonically, and passed to the backend validator. CLI exit success alone is not sufficient.

## 8. Authentication and preflight

The adapter may perform a cheap preflight such as checking that the Codex executable exists and that CLI authentication is available.

Authentication failure, missing executable, unsupported CLI flags, incompatible version, and invalid configured model/profile must fail explicitly before graph mutation.

Ordinary CI must not require a live authenticated Codex account. Use a fake executable or deterministic subprocess fixture to verify command construction, timeout handling, JSONL parsing, schema-output capture, and failure mapping.

A real Codex CLI smoke test is local/optional and must be clearly labelled as such.

## 9. First vertical-slice acceptance criteria

In addition to the parent specification, the first Codex implementation is accepted only when:

1. `CodexCliEpisodeAdapter` invokes the real supported `codex exec` command shape rather than an invented SDK.
2. A role-specific `--output-schema` is used or the installed CLI demonstrably lacks it and an equally strict documented alternative is implemented.
3. Internal Codex subagent topology is neither required nor reconstructed.
4. Valid structured Executor output passes through `commit_episode_result(...)`.
5. Malformed, stale, failed, timed-out, cancelled, partial, or over-grant output leaves graph state unchanged.
6. Timeout, cancellation, and retry settings identify whether they came from the user, main agent, or run default.
7. Usage fields are `unavailable` rather than estimated when the CLI does not expose reliable machine-readable data.
8. Lifecycle and quality telemetry are emitted without persisting the full internal event stream by default.
9. The existing strict-run/P0/P0.5 behavior remains green.
10. The PR does not claim that the CLI exposes Ultra's internal subagent traces or exact quota consumption.

## 10. Superseded open questions

For the first implementation, the parent specification's open questions are resolved as follows:

```text
transport:
    Codex CLI first
usage:
    assume unavailable unless explicitly exposed in documented machine-readable output
cancellation/timeouts:
    selected by user/main-agent/run policy; enforced and reported by the adapter
progress:
    coarse, rate-limited lifecycle observation; no full internal-event synchronization
```

These decisions do not change DTE ownership or the transport-neutral long-term interface.