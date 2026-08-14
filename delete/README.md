# Delete review area

This directory contains material that is **not authoritative for the `new` release line** and is kept only so the repository owner can review manual deletion safely.

Nothing under `delete/` is imported by `dte_backend`, included in the production Skill bundle manifest, or allowed to define controller physics.

## Categories

- `blocked-sidecar/`: obsolete `dte_nextgen` source files whose direct deletion was intermittently blocked by the tool safety layer. The copies here are for audit only; the original paths are listed below and should be deleted manually after review if they still exist.
- `replaced-authority/`: superseded authority documents. `SPEC.old.md` is the pre-proper-volume specification and must not be treated as current.
- `release-metadata/`: superseded generated release metadata retained only for comparison. The live manifest is generated from the current production tree by CI before verification and packaging.

## Original paths pending manual deletion if still present

- `src/dte_nextgen/thought_space/allocation.py`
- `src/dte_nextgen/thought_space/entropy.py`
- `src/dte_nextgen/thought_space/history.py`
- `src/dte_nextgen/thought_space/metric_measure_controller.py`
- `src/dte_nextgen/thought_space/occupancy.py`

After reviewing the archived copies, it is safe to delete those original sidecar paths and later delete this entire `delete/` directory. Do not delete `docs/PHYSICS.md`, `docs/DESIGN.md`, or the new root `SPEC.md` as part of this cleanup.
