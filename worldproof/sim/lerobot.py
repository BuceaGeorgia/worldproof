"""Read a LeRobotDataset (v3.0) directly — parquet + mp4, no ``lerobot`` package.

LeRobotDataset is the de-facto open standard for action-conditioned robot data
(thousands of datasets on the HF Hub, including Open-X mirrors). Its v3.0 layout
is open — low-dim columns in per-chunk **parquet**, camera frames in per-chunk
**mp4** keyed by timestamp — so reading it directly (``pyarrow`` +
``imageio-ffmpeg``) sidesteps the ``lerobot`` package, which requires Python
3.12+. worldproof therefore ingests these datasets on its supported 3.10 floor.

``LeRobotDatasetSource`` is a :class:`~worldproof.sim.dataset.DatasetSource`: it
windows recorded episodes into :class:`~worldproof.sim.base.OracleRollout`s
(context frames + the window's actions + the true future frames), which
:func:`~worldproof.sim.dataset.rollouts_from_dataset` turns into evaluable
rollouts against any model. Frames come from mp4, so they are lossy — a fidelity
caveat inherent to video-backed datasets.

Behind the ``lerobot-data`` extra; ``pyarrow`` / ``imageio`` / ``huggingface_hub``
are imported lazily, so the core install is unaffected.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from worldproof.sim.base import OracleRollout
from worldproof.sim.dataset import DatasetSource

__all__ = ["LeRobotDatasetSource"]

_EXTRA_HINT = "install it with `pip install worldproof[lerobot-data]`"


class LeRobotDatasetSource(DatasetSource):
    """A dataset source over a LeRobotDataset v3.0 (local path or HF repo id).

    Args:
        repo_id: A local dataset directory, or a HuggingFace dataset repo id
            (its files are downloaded on demand and cached).
        video_key: Which camera feature to read (e.g. ``"observation.image"``);
            defaults to the first ``video`` feature declared in ``meta/info.json``.
        action_key: The action column (default ``"action"``).
        cache_dir: HF download cache root (passed to ``hf_hub_download``).
    """

    def __init__(
        self,
        repo_id: str,
        *,
        video_key: str | None = None,
        action_key: str = "action",
        cache_dir: str | None = None,
    ) -> None:
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"LeRobotDatasetSource needs pyarrow + imageio-ffmpeg; {_EXTRA_HINT}"
            ) from exc

        self._repo_id = repo_id
        self._action_key = action_key
        self._cache_dir = cache_dir
        self._local_root = Path(repo_id) if Path(repo_id).is_dir() else None

        info = json.loads(self._path("meta/info.json").read_text(encoding="utf-8"))
        version = str(info.get("codebase_version", ""))
        if not version.startswith("v3"):
            raise ValueError(
                f"LeRobotDatasetSource supports LeRobotDataset v3.x; {repo_id!r} is "
                f"{version!r}. Convert it to v3, or read v2.x with the lerobot tools."
            )
        self._info = info
        self._fps = float(info["fps"])
        self._video_key = video_key or self._first_video_key(info)
        self._episodes = self._load_episode_index()

    # -- path resolution (local dir or HF repo) ----------------------------- #

    def _path(self, relpath: str) -> Path:
        if self._local_root is not None:
            return self._local_root / relpath
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self._repo_id,
                relpath,
                repo_type="dataset",
                cache_dir=self._cache_dir,
            )
        )

    def _list(self, prefix: str) -> list[str]:
        if self._local_root is not None:
            base = self._local_root / prefix
            return sorted(
                str(p.relative_to(self._local_root)) for p in base.rglob("*.parquet")
            )
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(self._repo_id, repo_type="dataset")
        return sorted(
            f for f in files if f.startswith(prefix) and f.endswith(".parquet")
        )

    @staticmethod
    def _first_video_key(info: dict) -> str:
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") == "video":
                return key
        raise ValueError("no video feature found in meta/info.json")

    def _load_episode_index(self) -> list[dict]:
        import pyarrow.parquet as pq

        rows: list[dict] = []
        vk = self._video_key
        for relpath in self._list("meta/episodes/"):
            table = pq.read_table(self._path(relpath))
            wanted = [
                "episode_index",
                "data/chunk_index",
                "data/file_index",
                f"videos/{vk}/chunk_index",
                f"videos/{vk}/file_index",
                f"videos/{vk}/from_timestamp",
                "length",
            ]
            cols = {c: table.column(c).to_pylist() for c in wanted}
            for i in range(table.num_rows):
                rows.append({c: cols[c][i] for c in wanted})
        rows.sort(key=lambda r: r["episode_index"])
        return rows

    # -- data / video readers ----------------------------------------------- #

    def _data_rows(self, ep: dict) -> dict[str, np.ndarray]:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        relpath = self._info["data_path"].format(
            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
        )
        table = pq.read_table(self._path(relpath))
        table = table.filter(pc.equal(table["episode_index"], ep["episode_index"]))
        out = {self._action_key: np.asarray(table[self._action_key].to_pylist())}
        if "next.success" in table.column_names:
            out["next.success"] = np.asarray(table["next.success"].to_pylist())
        return out

    def _video_frames(self, ep: dict, count: int) -> np.ndarray:
        import imageio

        vk = self._video_key
        relpath = self._info["video_path"].format(
            video_key=vk,
            chunk_index=ep[f"videos/{vk}/chunk_index"],
            file_index=ep[f"videos/{vk}/file_index"],
        )
        start = int(round(float(ep[f"videos/{vk}/from_timestamp"]) * self._fps))
        reader = imageio.get_reader(str(self._path(relpath)))
        frames: list[np.ndarray] = []
        try:
            for i, frame in enumerate(reader):
                if i < start:
                    continue
                frames.append(np.asarray(frame, dtype=np.uint8)[..., :3])
                if len(frames) >= count:
                    break
        finally:
            reader.close()
        return np.stack(frames, axis=0)

    # -- DatasetSource ------------------------------------------------------- #

    def truths(
        self, *, n: int, context_steps: int, horizon: int, seed: int = 0
    ) -> Iterator[OracleRollout]:
        span = context_steps + horizon
        yielded = 0
        for ep in self._episodes:
            if yielded >= n:
                return
            if int(ep["length"]) < span:
                continue  # episode too short for one window
            frames = self._video_frames(ep, span)
            if frames.shape[0] < span:
                continue  # decode fell short
            data = self._data_rows(ep)
            actions = data[self._action_key].astype(np.float32)
            # actions[i] produces future[i]: future frame (context_steps+i) is the
            # result of the action applied at frame (context_steps+i-1).
            action_window = actions[context_steps - 1 : context_steps - 1 + horizon]
            if action_window.shape[0] < horizon:
                continue
            success = data.get("next.success")
            is_failure = (
                bool(not np.asarray(success).any()) if success is not None else False
            )
            yield OracleRollout(
                context=frames[:context_steps],
                actions=action_window,
                future=frames[context_steps:span],
                context_id=f"{self._repo_id}:ep={ep['episode_index']}",
                is_failure=is_failure,
                info={"source": "lerobot", "video_key": self._video_key},
                fps=self._fps,
            )
            yielded += 1
