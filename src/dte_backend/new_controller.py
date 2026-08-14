"""Public compatibility import for the `new` proper-volume controller."""

from .controller_atlas import FrozenReferenceAtlas, freeze_reference_atlas
from .controller_state import FrontierControllerState, score_frontier

__all__ = [
    "FrozenReferenceAtlas",
    "FrontierControllerState",
    "freeze_reference_atlas",
    "score_frontier",
]
