# Implementation Tasks

- [x] 1. Remove current human-confirmation and researcher-learning contracts
  - [x] 1.1 Add failing model/schema tests for the absence of current
    `human_confirmed` inputs, human-count fields, learning models, and handoff
    learning output. Requirements: 1.1-1.4, 2.1-2.3.
  - [x] 1.2 Update epistemic models and generated schemas while retaining a
    read-only deprecated source token only where old ledger parsing requires
    it. Requirements: 2.1-2.3, 4.2.
  - [x] 1.3 Remove learning-ledger implementation and `learning:` reference
    authority; preserve existing files byte-for-byte. Requirements: 1.1-1.4.

- [x] 2. Narrow read-model and renderer semantics
  - [x] 2.1 Add failing handoff tests for ignored legacy learning files,
    legacy human-source limitations, and provenance-only artifact rendering.
    Requirements: 1.1-1.2, 2.3, 3.1-3.3.
  - [x] 2.2 Remove learning joins and human statistics from the terminal
    handoff and independence/source summaries. Requirements: 1.1, 2.1, 3.3.
  - [x] 2.3 Render `external_artifact_backed` as `artifact_referenced` with an
    explicit non-verification disclaimer. Requirements: 3.1-3.3.

- [x] 3. Correct terminal path disposition
  - [x] 3.1 Add regressions for `max_iterations`, `max_search_nodes`,
    `continuation_gate`, `controller_stop`, and `authorized_synthesis` terminal
    sources, including separation from epistemic contradiction. Requirements:
    5.1-5.3.
  - [x] 3.2 Update `_search_dispositions()` so both hard budget sources emit
    `out_of_budget` and other stop sources retain existing semantics.
    Requirements: 5.1-5.3.

- [x] 4. Remove CLI and documentation surface
  - [x] 4.1 Add/update CLI tests, remove `record-learning`, and prove
    `record-feedback` remains operational and isolated. Requirements: 1.3,
    4.3.
  - [x] 4.2 Update SPEC, ARCHITECTURE, AGENTS, SKILL, README, and App workflow
    to describe provenance-only authority and deprecated ignored learning
    artifacts. Requirements: 1.2, 3.2-3.3, 4.1-4.3.

- [x] 5. Integrate compatibility and regression coverage
  - [x] 5.1 Add/update epistemic smoke coverage for an artifact reference and
    `max_search_nodes` handoff, proving summaries do not mutate state.
    Requirements: 3.1-3.3, 4.2, 5.1-5.3, 6.1.
  - [x] 5.2 Run and repair the full test, schema, prompt, App/Judge/Executor/
    Relation/SearchNode/observability/epistemic/continuation matrices and all
    DTE guards without changing controller semantics. Requirements: 4.1-4.3,
    6.1.
