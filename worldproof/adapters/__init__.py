"""Model adapters and rollout sources.

v0.1 pure-Python surface: the :class:`WorldModelAdapter` protocol and the
folder-of-rollouts loader. External adapters (stable-worldmodel, LeRobot) land
in later work.
"""

from worldproof.adapters.base import WorldModelAdapter
from worldproof.adapters.folder import (
    FOLDER_FORMAT_VERSION,
    iter_rollouts,
    load_rollout,
    save_rollout,
)
from worldproof.adapters.swm import SWMAdapter

__all__ = [
    "WorldModelAdapter",
    "SWMAdapter",
    "FOLDER_FORMAT_VERSION",
    "save_rollout",
    "load_rollout",
    "iter_rollouts",
]
