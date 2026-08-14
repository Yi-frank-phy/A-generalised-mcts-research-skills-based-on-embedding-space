# Obsolete nextgen tests

These tests exercised the former experimental `dte_nextgen.thought_space` implementation. The `new` release has promoted its controller into production `src/dte_backend/**` and now uses `tests/test_new_release_controller.py`, `tests/test_new_release_physics.py`, the release-authority lock, and the full production test suite.

The files here are retained only for manual audit before deletion. They must not be collected by pytest and must not be used to preserve obsolete RBF/MMD/sidecar semantics on branch `new`.
