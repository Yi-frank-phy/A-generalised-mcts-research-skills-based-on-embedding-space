# Remove Human/Verifier Semantics from Epistemic Observability

## Introduction

This change narrows DTE epistemic observability to claims, assumptions,
evidence, challenges, dependencies, failure modes, lifecycle dispositions, and
provenance. Scientific verification, user confirmation, external-tool
validation, and researcher learning remain outside DTE authority. Existing run
files are read-only inputs and must not be rewritten or deleted.

## Requirements

### 1. Remove researcher-learning functionality

**User story:** As a researcher, I want DTE to avoid tracking my learning, so
that the backend records research provenance without claiming changes in human
capability or judgment.

1. WHEN current code builds an epistemic summary THEN it SHALL NOT read,
   interpret, export, or modify `epistemic/researcher_learning.jsonl`.
2. WHEN an old run directory contains that file THEN the current DTE SHALL
   ignore it and SHALL preserve its bytes unchanged.
3. WHEN the CLI is inspected THEN `record-learning` SHALL NOT exist, while
   `record-feedback` SHALL remain unchanged and controller-independent.
4. WHEN episode references are validated THEN new `learning:` references SHALL
   be rejected.

### 2. Remove human-confirmation semantics

**User story:** As a researcher, I want source provenance without human
certification levels, so that DTE cannot present a user as a verifier.

1. WHEN current episode or handoff schemas are generated THEN they SHALL NOT
   expose `human_confirmed` as a valid current source or count.
2. WHEN an Executor or Judge submits `human_confirmed` THEN schema or commit
   validation SHALL reject the result atomically.
3. WHEN a legacy ledger contains `human_confirmed` THEN read-only compatibility
   SHALL preserve the historical payload, report a legacy limitation, and
   SHALL NOT reinterpret it as `agent_reported` or scientific confirmation.

### 3. Preserve artifact provenance without verification authority

**User story:** As a researcher, I want references to Mathematica, literature,
and other artifacts preserved, so that I can inspect provenance without DTE
claiming those artifacts prove a scientific statement.

1. WHEN the persisted enum `external_artifact_backed` is encountered THEN the
   backend SHALL preserve it for hash and state compatibility.
2. WHEN human-readable output renders that source THEN it SHALL use the label
   `artifact_referenced` and state that artifact validity, assumptions,
   applicability, and claim truth are not checked by DTE.
3. WHEN independence indicators count artifact-bearing support THEN they SHALL
   describe reference coverage only, never verification, correctness,
   reliability, or independent validation.

### 4. Preserve epistemic provenance and controller boundaries

**User story:** As a DTE operator, I want this cleanup isolated from search and
controller semantics, so that existing DTE decisions remain reproducible.

1. WHEN this change is applied THEN UCB, entropy mathematics, allocation,
   committed-node budget, continuation material-yield rules, Judge scoring,
   Executor authority, Relation/merge, readiness, and sticky terminal behavior
   SHALL remain unchanged.
2. WHEN summaries are generated THEN they SHALL remain read-only and SHALL NOT
   mutate graph/controller state or existing run artifacts.
3. WHEN feedback is recorded THEN it SHALL remain an independent evaluation
   ledger and SHALL NOT become an epistemic verifier or controller input.

### 5. Correct budget-exhaustion path disposition

**User story:** As a researcher reading a terminal handoff, I want unexpanded
frontier paths stopped by the node cap classified accurately, so that budget
exhaustion is not confused with lack of exploration value.

1. WHEN terminal source is `max_iterations` or `max_search_nodes` THEN an
   unexpanded, non-selected frontier path SHALL include `out_of_budget`.
2. WHEN terminal source is `continuation_gate`, `controller_stop`, or
   `authorized_synthesis` THEN the path SHALL NOT become `out_of_budget` solely
   because the run terminated.
3. WHEN `out_of_budget` is emitted THEN it SHALL NOT create a contradicted or
   otherwise negative epistemic disposition.

### 6. Validation and delivery

**User story:** As a maintainer, I want compatibility and regression coverage,
so that the cleanup can be reviewed safely before merging.

1. WHEN implementation is complete THEN all existing tests, schema tests,
   App/Judge/Executor/Relation/SearchNode/observability/epistemic/continuation/
   prompt matrices, smoke workflows, and guards SHALL pass.
2. WHEN delivery is complete THEN changes SHALL be committed and pushed on
   `fix/remove-human-verifier-semantics` and a draft PR SHALL be opened without
   merging it to `main`.
