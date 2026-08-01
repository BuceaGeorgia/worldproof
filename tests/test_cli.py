"""Tests for the ``worldproof`` CLI (evaluate / generate)."""

from __future__ import annotations

import json

import numpy as np

from worldproof import save_rollout
from worldproof.cli import main
from worldproof.core import Rollout, RolloutMetadata

H = W = 24
T = 4


def _save_rollout_folder(folder, n=3):
    for i in range(n):
        rng = np.random.default_rng(i)
        gt = rng.integers(0, 256, (T, H, W, 3), np.uint8)
        ctx = rng.integers(0, 256, (3, H, W, 3), np.uint8)
        preds = tuple(
            np.clip(gt.astype(int) + rng.integers(-15, 16, gt.shape), 0, 255).astype(
                np.uint8
            )
            for _ in range(2)
        )
        metadata = RolloutMetadata(
            fps=10.0, model_id="m", resolution=(H, W), inference_latency_s=0.1
        )
        save_rollout(
            Rollout("pixels", ctx, np.zeros((T, 2), np.float32), preds, metadata, gt),
            folder / f"ep_{i}",
        )
    return folder


def test_evaluate_writes_json_and_html(tmp_path, capsys):
    folder = _save_rollout_folder(tmp_path / "rollouts")
    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    code = main(
        ["evaluate", str(folder), "--json", str(json_path), "--html", str(html_path)]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Evaluated 3 rollout(s)" in out

    assert json_path.exists() and html_path.exists()
    blob = json.loads(json_path.read_text())
    assert blob["n_rollouts"] == 3
    assert any(m["name"] == "psnr" for m in blob["metrics"])
    assert html_path.read_text().startswith("<!doctype html>")


def test_evaluate_missing_folder_errors(tmp_path):
    assert main(["evaluate", str(tmp_path / "does_not_exist")]) == 1


def test_evaluate_empty_folder_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    assert main(["evaluate", str(tmp_path / "empty")]) == 1
