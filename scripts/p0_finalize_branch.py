from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


spec = ROOT / "SPEC.md"
text = spec.read_text(encoding="utf-8")
replacement = r'''## 3. Geometry and embedding dimension

The current production-compatible controller uses normalized embedding geometry as a kernel-smoothed continuation of the original finite-state entropy construction. For normalized embeddings `z_i`, let

```text
d_ij^2 = ||z_i - z_j||^2
h^2 = median_{i<j}(d_ij^2) / 2
K_ij = exp(-d_ij^2 / (2 h^2))
rho_i = (1/N) * sum_j K_ij
H_geom = -(1/N) * sum_i log(rho_i)
```

The adaptive scale makes a typical pair have kernel overlap `exp(-1)`. Because the self-kernel is one,

```text
1/N <= rho_i <= 1
0 <= H_geom <= log(N)
```

`H_geom` is a bounded soft-discrete entropy. It is not the coordinate-dependent differential entropy of a normalized `D`-dimensional Gaussian KDE, and no Gaussian density-normalization factor belongs in it.

Current-frontier absolute local evidence and uncertainty are

```text
n_eff,i = N * rho_i
SD_i = 1 / sqrt(n_eff,i)
```

so the geometry layer does not min-max normalize uncertainty within each batch. Per-batch min-max normalization would erase the absolute evidence scale by forcing the densest branch to uncertainty zero and the sparsest branch to one.

The public compatibility controller still uses the observable Judge score as provisional `V_i`, and canonical UCB remains

```text
U_i = V_i + SD_i
```

For the current UCB spectrum define

```text
p_i(T) = exp(U_i / T) / sum_j exp(U_j / T)
H_B(T) = -sum_i p_i(T) log p_i(T)
```

For non-degenerate UCB values, `H_B(T)` is monotone increasing in `T`. The current controller hypothesis therefore determines the effective allocation temperature by solving

```text
H_B(T_t) = H_geom(t)
```

for the current frontier. Temperature must therefore depend on both the geometry entropy target and the current UCB scale. The old formula

```text
T = T_max * H_geom / log(N)
```

is superseded and must not control allocation.

The persisted field `normalized_temperature` remains for compatibility and replay telemetry; its value is the normalized geometry-entropy coordinate `H_geom/log(N)` when `N>1`, not `T/T_max` and not the effective Boltzmann temperature.

Cross-iteration `delta H` is plateau/continuation telemetry only. It does not define `U_i` or the current effective temperature.

For real runs, geometry should use the highest-quality configured embedding profile by default. `embedding_dimension` currently defaults to `3072`; lower dimensions are debug/fallback profiles. Hash embeddings are only for offline tests and CI.

Embedding cache identity must include at least:

```text
content_hash
embedding_provider
embedding_model_or_snapshot
embedding_dimension
embedding_contract_version
```

'''
pattern = re.compile(
    r"## 3\. Geometry and embedding dimension\n.*?(?=## 4\. Boltzmann allocation and budget semantics)",
    re.S,
)
text, count = pattern.subn(replacement, text, count=1)
assert count == 1, f"SPEC section replacement count={count}"
old = "The current diversity proxy controls temperature and its cross-iteration delta emits a replayable plateau signal."
new = "The current soft-discrete geometry entropy sets the target Boltzmann allocation entropy, while its cross-iteration delta emits a replayable plateau signal."
assert text.count(old) == 1, f"SPEC controller sentence count={text.count(old)}"
text = text.replace(old, new, 1)
spec.write_text(text, encoding="utf-8")

superseded = (
    "> **SUPERSEDED FOR GEOMETRY/CONTROLLER MATH (2026-08-11):** "
    "Use `docs/superpowers/specs/2026-08-11-p0-geometry-controller-recovery.md`. "
    "This file is retained only as historical implementation archaeology; its `H/log(N) -> T` law is not authoritative.\n\n"
)
for relative in [
    "docs/superpowers/plans/2026-08-10-canonical-dte-math-restoration.md",
    "docs/superpowers/specs/2026-08-10-canonical-dte-math-restoration-design.md",
]:
    path = ROOT / relative
    body = path.read_text(encoding="utf-8")
    if superseded not in body:
        first_newline = body.index("\n") + 1
        body = body[:first_newline] + "\n" + superseded + body[first_newline:]
        path.write_text(body, encoding="utf-8")

for relative in [
    ".github/workflows/p0-apply-wiring.yml",
    ".github/workflows/p0-math-tdd.yml",
    ".github/workflows/p0-finalize.yml",
    "scripts/p0_finalize_branch.py",
]:
    path = ROOT / relative
    if path.exists():
        path.unlink()
