"""Runs the bring-your-own example end-to-end (a 'tested claim', CLAUDE.md).

The README's "Your own model, your own data" section is abridged from
``examples/bring_your_own.py``; running it here means CI verifies all three
journeys (own adapter, own DatasetSource, precomputed rollout folder) on the
core install across all three operating systems.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example():
    path = Path(__file__).resolve().parent.parent / "examples" / "bring_your_own.py"
    spec = importlib.util.spec_from_file_location("wp_bring_your_own", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bring_your_own_journeys(tmp_path, capsys):
    example = _load_example()
    out = example.main(tmp_path)

    printed = capsys.readouterr().out
    # journey 1 + 2: the custom adapter scored against the custom dataset
    assert "your model vs your recordings" in printed
    assert "Evaluated 4 rollout(s)" in printed
    # journey 3: the precomputed folder round-trips and evaluates
    assert "precomputed rollout folder" in printed
    assert (out / "rollouts" / "ep_00" / "manifest.json").exists()

    # the folder is also scoreable through the CLI evaluate verb
    from worldproof.cli import main as cli_main

    assert cli_main(["evaluate", str(out / "rollouts")]) == 0
