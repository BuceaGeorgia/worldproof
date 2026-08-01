"""Tests for reading LeRobotDataset v3.0 directly (parquet + mp4, no lerobot pkg).

Hermetic: each test writes a tiny dataset in the **real v3.0 on-disk layout**
(meta/info.json + data parquet + a real mp4 + meta/episodes parquet) to a temp
dir and reads it back — no network, no ``lerobot`` package. Gated on the
``lerobot-data`` extra (pyarrow + imageio-ffmpeg); skips cleanly on core CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("imageio_ffmpeg")

import imageio  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from worldproof.baselines import CopyLastFrameBaseline  # noqa: E402
from worldproof.metrics import PSNR  # noqa: E402
from worldproof.sim import LeRobotDatasetSource, rollouts_from_dataset  # noqa: E402

VK = "observation.image"


def _write_v3_dataset(
    root: Path, *, n_ep=3, ep_len=8, h=32, w=32, fps=10, version="v3.0"
):
    (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (root / f"videos/{VK}/chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": version,
        "fps": fps,
        "total_episodes": n_ep,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "features": {
            VK: {"dtype": "video", "shape": [h, w, 3]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")

    rng = np.random.default_rng(0)
    all_frames, actions, ep_index, success, ep_rows = [], [], [], [], []
    for e in range(n_ep):
        # distinct per-(episode,frame) content so decoded frames vary
        base = rng.integers(0, 200, (h, w, 3), np.uint8)
        for t in range(ep_len):
            all_frames.append(np.clip(base + t * 5 + e * 20, 0, 255).astype(np.uint8))
            actions.append([float(e), float(t)])
            ep_index.append(e)
            success.append(False)
        ep_rows.append(
            {
                "episode_index": e,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{VK}/chunk_index": 0,
                f"videos/{VK}/file_index": 0,
                f"videos/{VK}/from_timestamp": (e * ep_len) / fps,
                "length": ep_len,
            }
        )

    writer = imageio.get_writer(
        str(root / f"videos/{VK}/chunk-000/file-000.mp4"),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )
    for f in all_frames:
        writer.append_data(f)
    writer.close()

    pq.write_table(
        pa.table(
            {
                "action": actions,
                "episode_index": ep_index,
                "next.success": success,
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(ep_rows),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    return root


def test_reads_v3_windows(tmp_path):
    _write_v3_dataset(tmp_path, n_ep=3, ep_len=8)
    source = LeRobotDatasetSource(str(tmp_path))
    truths = list(source.truths(n=2, context_steps=2, horizon=3))
    assert len(truths) == 2
    t = truths[0]
    assert t.context.shape == (2, 32, 32, 3) and t.context.dtype == np.uint8
    assert t.future.shape == (3, 32, 32, 3)
    assert t.actions.shape == (3, 2)  # one action per predicted step
    assert t.context_id.endswith(":ep=0")
    assert t.is_failure is True  # never succeeded


def test_v3_rollouts_are_evaluable(tmp_path):
    _write_v3_dataset(tmp_path, n_ep=3, ep_len=10)
    source = LeRobotDatasetSource(str(tmp_path))
    rollouts = rollouts_from_dataset(
        source, CopyLastFrameBaseline(), n=3, context_steps=3, horizon=4
    )
    assert len(rollouts) == 3
    r = rollouts[0]
    assert r.modality == "pixels" and r.has_ground_truth
    import torch

    result = PSNR().compute(r, device=torch.device("cpu"))
    assert result.horizon == 4 and np.isfinite(result.summary)


def test_rejects_non_v3(tmp_path):
    _write_v3_dataset(tmp_path, version="v2.1")
    with pytest.raises(ValueError, match="v3"):
        LeRobotDatasetSource(str(tmp_path))


@pytest.mark.slow
def test_reads_real_lerobot_pusht():
    """End-to-end on a real HF LeRobotDataset (small: ~7 MB). Skips offline."""
    try:
        source = LeRobotDatasetSource("lerobot/pusht")
        rollouts = rollouts_from_dataset(
            source, CopyLastFrameBaseline(), n=3, context_steps=3, horizon=5
        )
    except Exception as exc:  # network / hub unavailable
        pytest.skip(f"lerobot/pusht unavailable: {exc}")
    assert len(rollouts) == 3
    r = rollouts[0]
    assert r.modality == "pixels" and r.has_ground_truth
    assert r.context.shape == (3, 96, 96, 3) and r.actions.shape == (5, 2)
