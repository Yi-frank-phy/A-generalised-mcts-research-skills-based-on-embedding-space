# P0 Budget and Executor Boundary Tasks

- [x] 1. Add failing allocation and budget-contract tests.
  - Cover half-up rounding, the normative quota vector, hard-cap trimming,
    permutation invariance, legacy mapping, conflict rejection, and canonical
    serialization/schema output.
  - Requirements: 1.1-1.4, 2.1-2.5.

- [x] 2. Implement canonical budget normalization and deterministic allocation.
  - Update the Pydantic budget model, math engine, runner integration, and trace
    fields without changing UCB.
  - Requirements: 1.1-1.4, 2.1-2.5.

- [x] 3. Add failing Executor authority-boundary tests and implement enforcement.
  - Reject every explicitly supplied controller field in raw and parsed outputs
    while preserving legal adapters and existing structural checks.
  - Requirements: 3.1-3.3.

- [x] 4. Add cache namespace tests and implement structured cache identities.
  - Namespace in-memory and persistent embedding/Judge keys; retain legacy cache
    data without migration or deletion.
  - Requirements: 4.1-4.3.

- [x] 5. Update machine-facing schemas, examples, and protocol documentation.
  - Replace formal legacy budget usage with canonical fields and repair the
    repository read-first guidance without adding P1/P2 architecture.
  - Requirements: 1.4, 5.2, 5.4.

- [x] 6. Run regression and protocol verification.
  - Execute targeted tests, all four guard paths, the complete pytest suite,
    diff validation, and repository legacy-name scanning.
  - Requirements: 5.1-5.4.
