# Evolving Frontier Research Skill

**Evolving Frontier Research Skill (DTE)** is a local research-agent backend for structured frontier search over difficult mathematical, physical, academic, and conceptual problems.

## Branches

This repository has one primary development/release line and one preserved legacy line:

| Branch | Role | Controller |
|---|---|---|
| [`new`](https://github.com/Yi-frank-phy/A-generalised-mcts-research-skills-based-on-embedding-space/tree/new) | **Primary mainline / default / recommended** | completed-transition embedding + frozen-atlas proper-volume metric-measure controller |
| [`old`](https://github.com/Yi-frank-phy/A-generalised-mcts-research-skills-based-on-embedding-space/tree/old) | Legacy compatibility / reproducibility | direct node/semantic embedding + RBF-KDE compatibility controller |

**`new` is the active development, maintenance, and release branch.** Future fixes, documentation work, controller changes, and normal releases land on `new`. `old` is retained so previous runs and v1 controller behavior remain reproducible; it is not a peer development line.

## What `new` changes

The current `new` controller treats a completed research transition as the searchable state:

```text
(retrospective method, epistemic change)
```

The question/context, Judge score, provenance metadata, and runtime telemetry are not part of the canonical controller embedding.

The controller uses a frozen reference atlas and a metric-measure geometry. Research displacement is measured as crossed proper volume; local value `V` is estimated from realized transition returns; uncertainty `SD` is derived from live occupancy through entropy matching; allocation uses

```text
U = V + SD
```

Judge scores remain research observations and are not controller value. The authoritative mathematical definition is in [`docs/PHYSICS.md`](docs/PHYSICS.md); executable lifecycle and persistence requirements are in [`SPEC.md`](SPEC.md); release architecture is in [`docs/DESIGN.md`](docs/DESIGN.md).

## Status

This project is **public alpha**. The backend, persistence contracts, packaging, CI matrix, smoke workflow, and App/headless protocol are intended to be runnable and reproducible. Passing tests establishes implementation behavior; it does **not** establish scientific correctness or prove that DTE improves research outcomes.

Current package version on the primary mainline: `2.0.0`.

## Install

For current development and evaluation, use `new`:

```bash
git clone --branch new --single-branch https://github.com/Yi-frank-phy/A-generalised-mcts-research-skills-based-on-embedding-space.git
cd A-generalised-mcts-research-skills-based-on-embedding-space
python -m pip install -e .[dev]
```

For legacy v1 compatibility, replace `new` with `old`.

## Verify the checkout

```bash
python scripts/generate_bundle_manifest.py --verify
python -m pytest -q
python scripts/smoke_workflow.py
```

The smoke workflow may use mock adapters and is only a machinery check.

## Use as a Skill

`SKILL.md` is the primary runtime contract. The repository archive is the complete Skill distribution; the Python wheel contains the backend package.

To verify/install a repository Skill bundle:

```bash
python scripts/install_skill_bundle.py --source . --target <target-directory>
```

For Codex App / Work, inspect the native driver interface with:

```bash
python -m dte_backend hook-driver --help
```

The App-native path uses the backend-controlled `hook-driver` lifecycle. The compatible headless path remains available through `strict-run`.

## Repository layout

```text
SKILL.md                 runtime Skill contract
SPEC.md                  executable release contract
docs/PHYSICS.md          authoritative controller mathematics on new
docs/DESIGN.md           release architecture
src/dte_backend/         production backend
schemas/                 machine-readable episode/state schemas
hooks/                   App enforcement hooks
scripts/                 install, manifest, smoke, and adapter utilities
examples/                example run specifications and smoke adapters
tests/                   regression and release-contract tests
```

## Release policy

- **All future normal releases are cut from `new`.**
- `new` is the default and primary long-term development, maintenance, and release branch.
- `old` is retained only for v1 compatibility, reproducibility, and narrowly scoped maintenance.
- Do not silently move old controller mathematics into `new` or vice versa.
- A theory-affecting change on `new` must update the mathematical authority and its lock/tests explicitly.
- Tags/releases should point to the exact `new` commit being released.

The current mainline release is **v2.0.0**.

## License

Apache-2.0. See [`LICENSE`](LICENSE).