"""Core import-surface invariant.

worldproof must import and expose its public API with core dependencies only
(numpy + torch + pillow). Importing it must NOT pull the optional adapter
stacks (stable-worldmodel, lerobot) — those are lazily imported only when an
adapter is constructed. This test encodes that invariant from the code side;
the CI ``core-only`` job proves the same mechanically in a fresh install.
"""

from __future__ import annotations

import subprocess
import sys


def test_public_api_exports():
    import worldproof

    for name in (
        "Rollout",
        "RolloutMetadata",
        "MetricResult",
        "WorldModelAdapter",
        "SWMAdapter",
        "CopyLastFrameBaseline",
        "ActionBlindBaseline",
        "save_rollout",
        "load_rollout",
        "iter_rollouts",
        "get_device",
    ):
        assert hasattr(worldproof, name), f"missing public export: {name}"


def test_importing_worldproof_does_not_pull_optional_adapter_stacks():
    # Checked in a fresh interpreter: other tests in this process may legitimately
    # import the swm stack, so an in-process sys.modules check would be meaningless.
    code = (
        "import sys, worldproof, worldproof.adapters.swm\n"
        "assert 'stable_worldmodel' not in sys.modules, 'pulled stable_worldmodel'\n"
        "assert 'lerobot' not in sys.modules, 'pulled lerobot'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_swm_adapter_errors_helpfully_without_extra():
    """If the swm stack is absent, constructing SWMAdapter must say how to fix it."""
    import importlib.util

    if importlib.util.find_spec("stable_worldmodel") is not None:
        import pytest

        pytest.skip("stable-worldmodel is installed; absence path not exercised")

    import pytest

    from worldproof.adapters.swm import SWMAdapter

    with pytest.raises(ImportError, match=r"pip install worldproof\[swm\]"):
        SWMAdapter("quentinll/lewm-pusht")
