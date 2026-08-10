# Strict Boltzmann entropy–temperature controller design — superseded

**Status: superseded. Do not implement this design.**

This document proposed recovering temperature by inverting a Boltzmann/Shannon entropy relation. Subsequent repository archaeology showed that this would be a new controller design rather than a faithful restoration of the earlier DTE mathematical structure.

The authoritative design for the current restoration work is:

`docs/superpowers/specs/2026-08-10-canonical-dte-math-restoration-design.md`

The canonical restoration separates:

\[
U_i=V_i+\mathrm{SD}_i
\]

from the population-level temperature

\[
\tau=H/\log N,\qquad T=T_{\max}\tau,
\]

and then uses

\[
p_i\propto \exp(U_i/T)
\]

for resource allocation.

The original version of this superseded proposal remains available in git history at commit `e53505279a6501f719e8f8560f5c8f262e2c2d99`.