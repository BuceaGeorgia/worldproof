"""Runs the README quickstart end-to-end (a 'tested claim', CLAUDE.md).

Exercising ``examples/quickstart.py`` in the suite means CI verifies the
quickstart on ubuntu / macos / windows with the core install.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_quickstart():
    path = Path(__file__).resolve().parent.parent / "examples" / "quickstart.py"
    spec = importlib.util.spec_from_file_location("wp_quickstart", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quickstart_generates_and_reports(tmp_path, capsys):
    quickstart = _load_quickstart()
    out = quickstart.main(tmp_path)

    assert (out / "report.json").exists()
    assert (out / "report.html").exists()

    report = json.loads((out / "report.json").read_text())
    assert report["n_rollouts"] == 6
    names = {m["name"] for m in report["metrics"]}
    # core (no-extra) metrics must have run
    assert {"psnr", "ssim", "calibration_ece"} <= names

    html = (out / "report.html").read_text()
    assert html.startswith("<!doctype html>")
    assert "http://" not in html and "https://" not in html  # self-contained
