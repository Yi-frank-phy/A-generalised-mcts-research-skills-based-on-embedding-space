"""New-line cold-start entrypoint.

The old role-pipeline implementation lives on the parallel old release line.
The new line seeds completed method→epistemic transitions directly.
"""

from .seed_transitions import seed_frontier as seed_frontier_from_roles

__all__ = ["seed_frontier_from_roles"]
