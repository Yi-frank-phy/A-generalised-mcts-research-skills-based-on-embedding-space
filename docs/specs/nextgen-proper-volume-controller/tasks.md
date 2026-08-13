# Tasks

1. Add RED tests for absolute/non-node-normalized cumulative proper volume and reward-SD on the same variable.
2. Add RED tests for angular sparse-graph geodesic and nearest-reference anchoring.
3. Add RED tests that live duplicates have higher occupancy/lower reward-SD than an isolated direction.
4. Add RED tests for end-to-end `U=V+SD` with default equal-weight frozen reference measure.
5. Add RED tests for a stateful session: frozen atlas, zero value before history, parent->child return writeback, slot replacement, history-local value regression, and rejection of unscoped pre-numeric history.
6. Implement the minimum modules needed to make those tests green while preserving legacy RBF/MMD baselines.
7. Update next-generation status documentation to supersede the old RBF/MMD authority only inside `dte_nextgen`.
8. Open a public Draft PR and require the repository's full CI matrix plus package job to pass.
9. Merge to public `main` only after every required GitHub Actions job is successful.
10. Keep the private experiment PR unmerged; it remains provenance/research history rather than the release target.
