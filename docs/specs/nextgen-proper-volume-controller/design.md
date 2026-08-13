# Design

## Module boundary

The proper-volume implementation stays inside `dte_nextgen.thought_space` and does not modify production `dte_backend` controller semantics.

Planned modules:

- `geometry.py`: angular kNN graph, shortest-path geodesic, nearest-reference anchoring;
- `proper_volume.py`: atlas cell volumes, radius->cumulative proper volume, occupancy entropy, radial Boltzmann inversion, reward statistics;
- `occupancy.py`: live-live source-centred proper-volume displacement and soft occupancy;
- `controller.py`: retain legacy RBF functions and add proper-volume scoring from radii/embeddings;
- `return_metric.py`: retain MMD baseline and add proper-volume parent->child return;
- `history.py`: run-local realized-return evidence store;
- `session.py`: frozen-atlas stateful score/select/complete loop.

Existing `transition.py` remains the canonical `(method, epistemic change)` serializer and existing `allocation.py` remains the one-action Boltzmann scheduler.

## Data flow

One session freezes the reference atlas and computes its sparse geodesic matrix once.

For each scoring pass:

1. embed current completed-transition frontier;
2. map live transitions onto the frozen atlas;
3. compute source-centred cumulative proper-volume separations `D_ij`;
4. estimate `rho_i` from current live mass only;
5. set `S_i=-log(rho_i)`;
6. invert each node's radial Boltzmann entropy;
7. push Boltzmann cell mass through `A_ia=D_i(r_ia)`;
8. use `SD[A_i]` as controller uncertainty;
9. regress historical realized proper-volume returns locally to obtain `V_i`;
10. score `U_i=V_i+SD[A_i]`;
11. use mean node entropy as the target entropy for one-action Boltzmann allocation.

After external execution returns one completed child, measure parent->child proper-volume return on the same frozen atlas, record it against the retired parent, replace the parent slot with the child, and recompute on the next pass.

## Failure boundaries

Reject zero-norm/nonfinite embeddings, invalid graph `k`, disconnected reference graphs, nonpositive quadrature weights, invalid occupancy, reference atlases smaller than the live frontier, and nonempty pre-numeric history that cannot prove atlas identity.

Default runtime never auto-fits a sampling-density correction. Experimental correction requires explicit positive `reference_density` input.

## Compatibility

Legacy RBF geometry and MMD helpers remain importable for preregistered falsification and backwards compatibility. They are not used by `ProperVolumeTransitionSession`.
