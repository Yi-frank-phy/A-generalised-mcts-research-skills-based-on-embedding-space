# Evolving Frontier Research Skill — `old` compatibility line

This branch preserves the **old DTE controller** for compatibility, reproducibility, and comparison with the current release line.

For new work, use the [`new`](https://github.com/Yi-frank-phy/A-generalised-mcts-research-skills-based-on-embedding-space/tree/new) branch.

## Branches

| Branch | Status | Controller |
|---|---|---|
| [`new`](https://github.com/Yi-frank-phy/A-generalised-mcts-research-skills-based-on-embedding-space/tree/new) | **Recommended / current** | completed-transition embedding + frozen-atlas proper-volume metric-measure controller |
| `old` | Compatibility / reproducibility | direct node/semantic embedding + RBF-KDE compatibility controller |

`old` is intentionally retained as a separate release line. It should not silently absorb the `new` controller mathematics, and `new` should not silently restore old controller semantics.

## Status

This branch preserves the **feature-complete v1** protocol as public-alpha compatibility software. It exists to reproduce the previous controller behavior and support older runs, comparisons, and maintenance fixes. Passing tests establishes implementation behavior; it does not establish scientific correctness or research effectiveness.

Current package version: `0.2.0`.

## Install

```bash
git clone --branch old --single-branch https://github.com/Yi-frank-phy/A-generalised-mcts-research-skills-based-on-embedding-space.git
cd A-generalised-mcts-research-skills-based-on-embedding-space
python -m pip install -e .[dev]
```

## Verify the checkout

```bash
python scripts/generate_bundle_manifest.py --verify
python -m pytest -q
python scripts/smoke_workflow.py
```

The smoke workflow may use mock adapters and is only a machinery check.

## Use as a Skill

`SKILL.md` is the primary runtime contract. The repository archive is the complete Skill distribution; the Python wheel contains the backend package.

```bash
python scripts/install_skill_bundle.py --source . --target <target-directory>
python -m dte_backend hook-driver --help
```

## Compatibility controller

The `old` line retains the earlier node/semantic-embedding controller and its RBF-KDE-style geometry/uncertainty behavior. It is kept so historical behavior remains inspectable and reproducible.

If you want the current transition-based proper-volume controller, switch to `new` rather than modifying `old` in place.

## Repository layout

```text
SKILL.md                 runtime Skill contract
SPEC.md                  executable protocol contract
src/dte_backend/         production compatibility backend
schemas/                 machine-readable episode/state schemas
hooks/                   App enforcement hooks
scripts/                 install, manifest, smoke, and adapter utilities
examples/                example run specifications and smoke adapters
tests/                   regression tests
```

## Release policy

- Current DTE releases should normally be cut from `new`.
- Keep `old` available for compatibility and reproducibility.
- Tags/releases must point to the exact branch commit being released.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
