"""Sim oracles — ground-truth providers that generate true futures.

A :class:`SimOracle` steps a simulator to produce the real future for arbitrary
action sequences (SPEC §5), unblocking the signature metrics. :class:`ToySimOracle`
is a pure-numpy reference that runs with the core install; :class:`GymSimOracle`
wraps a pixel-rendering gymnasium environment (``swm/*`` behind the ``swm`` extra,
``ALE/*`` Atari games via :class:`AtariSimOracle` behind the ``atari`` extra). The
builders assemble the frozen pair contracts the signature metrics consume.
"""

from worldproof.sim.base import OracleRollout, SimOracle
from worldproof.sim.build import (
    generate_rollouts,
    make_counterfactual_pair,
    make_rollout,
    make_voe_pair,
)
from worldproof.sim.dataset import DatasetSource, rollouts_from_dataset
from worldproof.sim.gym import AtariSimOracle, GymSimOracle
from worldproof.sim.lerobot import LeRobotDatasetSource
from worldproof.sim.toy import ToySimOracle

__all__ = [
    "SimOracle",
    "OracleRollout",
    "ToySimOracle",
    "GymSimOracle",
    "AtariSimOracle",
    "make_rollout",
    "make_counterfactual_pair",
    "make_voe_pair",
    "generate_rollouts",
    "DatasetSource",
    "rollouts_from_dataset",
    "LeRobotDatasetSource",
]
