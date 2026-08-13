# DTE New — Release Design

This file is the normative architecture authority for the `new` release line. Ordinary refactors and feature work do not alter its controller boundaries. Theory-affecting changes must update `docs/PHYSICS.md`, this design, tests, and the theory lock together.

## Product-line rule

The repository has two parallel release lines:

- `old`: the direct-node-embedding / RBF-KDE controller retained for comparison and reproducibility.
- `new`: the completed-transition / proper-volume metric-measure controller defined in `docs/PHYSICS.md`.

Neither line is conceptually subordinate to `main`. A GitHub default branch may exist for hosting mechanics only; release identity is the branch/tag line.

## New runtime data contract

Every actively searchable node must carry a completed research transition:

- `retrospective_method` — non-empty method/intervention/representation change;
- `epistemic_change_kind` — `new_understanding`, `sharper_unknown`, or `no_material_change`;
- `epistemic_change` — non-empty result description.

The canonical controller embedding contains only those fields. Claim, rationale, question/context, Judge score, parent IDs, UCB, allocation, and runtime metadata are excluded from geometry.

Seed producers and Executor outputs must provide the transition fields. The new controller fails closed if an active node lacks them.

## Runtime controller

The production `dte_backend` path on `new` owns the proper-volume implementation. `dte_nextgen` is not a second runtime and is removed after migration.

At run initialization, the controller freezes a reference atlas from the run's initial completed transitions (or an explicitly supplied compatible atlas when that interface is available). That atlas/gauge remains fixed while run-local returns are accumulated.

At each controller step:

1. embed canonical completed transitions;
2. anchor live transitions to the frozen reference atlas;
3. compute sparse angular graph geodesics and cumulative proper-volume displacement;
4. reconstruct historical parent→child realized returns on that same atlas and locally regress `V`;
5. compute live occupancy, `S=-log rho`, entropy-matched radial Boltzmann mass, and proper-volume reward SD;
6. compute `U=V+SD`;
7. solve one-action Boltzmann temperature from mean live occupancy entropy and allocate under hard budgets;
8. grant the bounded Executor episode selected by the controller;
9. commit completed children with their transition fields, retire/close the used parent in the active frontier, retain provenance/history, and repeat.

Judge remains an observable research-assessment role for provenance/risk/synthesis support. Its score is not controller value on `new`.

## Persistence

Persistent App runs record the frozen atlas identity and enough transition evidence to reconstruct proper-volume returns. Numeric return evidence is valid only under its atlas identity. Existing process telemetry may retain compatibility field names only when their new semantics are documented; it must not label old KDE density as the new physics.

## Packaging and release

`new` is independently buildable and releasable as the repository Skill bundle plus backend wheel. CI must run the complete test matrix and package verification on the `new` branch. New-line tags use a distinct prefix (for example `new-v0.3.0-alpha.1`) so a release is unambiguously tied to this physics line.

`old` remains independently buildable/releasable from its own branch/tag prefix.

## Repository hygiene

The release branch keeps only current operational documentation plus the formal `PHYSICS.md`, `DESIGN.md`, `SPEC.md`, `ARCHITECTURE.md`, and user-facing workflow documentation. Temporary handoffs, implementation plans, intermediate theory audits, superseded nextgen specs, and experiment notes are not release artifacts and are deleted from `new` after migration.

A CI theory-lock test pins the SHA-256 digests of `docs/PHYSICS.md` and `docs/DESIGN.md`. Changing either file requires an explicit update to the lock in the same intentional theory-change patch.