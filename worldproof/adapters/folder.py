"""Folder-of-rollouts source: load pre-generated rollouts from disk.

This is the "no model required" source from SPEC §3/§5. Unlike a
:class:`~worldproof.adapters.base.WorldModelAdapter` it does not *predict*; it
reconstructs complete :class:`~worldproof.core.Rollout` objects that were
generated earlier (by ``worldproof generate``, a sim oracle, or by hand).

Provisional on-disk format — v0.1, FLAGGED FOR REVIEW
-----------------------------------------------------
This layout will become a public contract, so it is intentionally simple and
lossless. One rollout lives in one directory::

    <rollout_dir>/
        manifest.json         # all scalar metadata + storage mode (required)
        actions.npy           # (horizon, *action_dims)
        # pixels  (manifest.storage == "png_sequence"):
        context/  frame_00000.png frame_00001.png ...
        predictions/
            sample_00/ frame_00000.png ...
            sample_01/ frame_00000.png ...
        ground_truth/ frame_00000.png ...     # optional
        # latents (manifest.storage == "npy"):
        context.npy
        predictions/ sample_00.npy sample_01.npy ...
        ground_truth.npy                       # optional

``manifest.json`` fields::

    {
        "format_version": "0.1-provisional",
        "modality": "pixels" | "latents",
        "storage": "png_sequence" | "npy" | "video",
        "fps": 10.0,
        "resolution": [H, W] | null,
        "channels": 3 | null,        # pixels only
        "model_id": "...",
        "seed": 0 | null,
        "camera_id": "..." | null,
        "inference_latency_s": 0.0 | null,
        "context_id": "..." | null,
        "is_failure": true | false | null,
        "horizon": 8,
        "n_samples": 2,
        "has_ground_truth": true | false
    }

Design choices:

- **Lossless by default — this is a metric-correctness decision, not just
  convenience.** Pixels are written as PNG sequences (via pillow, a core
  dependency); latents as ``.npy``. worldproof exists to compute fidelity
  metrics (PSNR/SSIM/LPIPS); evaluating those on lossily re-encoded frames
  would pollute the very scores we report with codec artefacts. The canonical
  format is therefore lossless, guaranteeing an exact round-trip: the numbers a
  metric reads are the numbers that were generated.
- **Video is read-only and lossy.** ``storage == "video"`` lets a folder point
  at ``context.mp4`` / ``predictions/sample_00.mp4`` etc. for user-supplied
  clips. Decoding is lazily imported behind the ``io`` extra
  (``pip install worldproof[io]``); it is never used when *writing*. Rollouts
  loaded this way are flagged ``metadata.lossy_source = True`` so downstream
  reports can annotate their scores; exact equality is not guaranteed.
- **One directory = one rollout = one camera.** Many rollouts per episode live
  as sibling directories; :func:`iter_rollouts` walks them.
- **Versioned.** Every manifest records ``format_version``; :func:`load_rollout`
  rejects a mismatch rather than silently misreading a future layout.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from worldproof.core import Modality, Rollout, RolloutMetadata

__all__ = [
    "FOLDER_FORMAT_VERSION",
    "save_rollout",
    "load_rollout",
    "iter_rollouts",
]

FOLDER_FORMAT_VERSION = "0.1-provisional"
_MANIFEST_NAME = "manifest.json"
_FRAME_TEMPLATE = "frame_{:05d}.png"


def _index_of(path: Path) -> int:
    """Numeric suffix of a ``name_N`` stem — sort by this, not lexicographically,
    so ``sample_100`` follows ``sample_99`` regardless of zero-pad width."""
    return int(path.stem.rsplit("_", 1)[-1])


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def save_rollout(rollout: Rollout, path: str | Path) -> Path:
    """Write ``rollout`` to ``path`` in the lossless provisional format.

    Pixels are stored as PNG sequences, latents as ``.npy``. Returns the
    directory written to.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / "actions.npy", rollout.actions)

    if rollout.modality == "pixels":
        storage = "png_sequence"
        channels = int(rollout.predictions[0].shape[-1])
        _write_frames(path / "context", rollout.context)
        predictions_dir = path / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        for i, prediction in enumerate(rollout.predictions):
            _write_frames(predictions_dir / f"sample_{i:02d}", prediction)
        if rollout.ground_truth is not None:
            _write_frames(path / "ground_truth", rollout.ground_truth)
    else:
        storage = "npy"
        channels = None
        np.save(path / "context.npy", rollout.context)
        predictions_dir = path / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        for i, prediction in enumerate(rollout.predictions):
            np.save(predictions_dir / f"sample_{i:02d}.npy", prediction)
        if rollout.ground_truth is not None:
            np.save(path / "ground_truth.npy", rollout.ground_truth)

    manifest = {
        "format_version": FOLDER_FORMAT_VERSION,
        "modality": rollout.modality,
        "storage": storage,
        "fps": rollout.metadata.fps,
        "resolution": (
            list(rollout.metadata.resolution)
            if rollout.metadata.resolution is not None
            else None
        ),
        "channels": channels,
        "model_id": rollout.metadata.model_id,
        "seed": rollout.metadata.seed,
        "camera_id": rollout.metadata.camera_id,
        "lossy_source": rollout.metadata.lossy_source,
        "inference_latency_s": rollout.metadata.inference_latency_s,
        "context_id": rollout.metadata.context_id,
        "is_failure": rollout.metadata.is_failure,
        "horizon": rollout.horizon,
        "n_samples": rollout.n_samples,
        "has_ground_truth": rollout.has_ground_truth,
    }
    (path / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _write_frames(directory: Path, frames: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for t, frame in enumerate(frames):
        if frame.shape[-1] == 1:
            image = Image.fromarray(frame[..., 0], mode="L")
        else:
            image = Image.fromarray(frame, mode="RGB")
        image.save(directory / _FRAME_TEMPLATE.format(t))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def load_rollout(path: str | Path) -> Rollout:
    """Load a single rollout directory back into a :class:`Rollout`."""
    path = Path(path)
    manifest_path = path / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no {_MANIFEST_NAME} in {path}; not a worldproof rollout directory"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    version = manifest.get("format_version")
    if version != FOLDER_FORMAT_VERSION:
        raise ValueError(
            f"rollout at {path} has format_version {version!r}, but this "
            f"worldproof reads {FOLDER_FORMAT_VERSION!r}. Re-export the rollout "
            "with a matching worldproof version."
        )

    modality: Modality = manifest["modality"]
    storage = manifest["storage"]
    channels = manifest.get("channels")

    actions = np.load(path / "actions.npy")

    if modality == "pixels":
        context = _read_pixel_stream(path / "context", storage, channels)
        predictions = tuple(
            _read_pixel_stream(sample_dir, storage, channels)
            for sample_dir in _sample_dirs(path / "predictions", storage)
        )
        ground_truth = (
            _read_pixel_stream(path / "ground_truth", storage, channels)
            if manifest["has_ground_truth"]
            else None
        )
    else:
        context = np.load(path / "context.npy")
        predictions = tuple(
            np.load(sample_file)
            for sample_file in sorted(
                (path / "predictions").glob("sample_*.npy"), key=_index_of
            )
        )
        ground_truth = (
            np.load(path / "ground_truth.npy") if manifest["has_ground_truth"] else None
        )

    resolution = manifest.get("resolution")
    lossy_source = bool(manifest.get("lossy_source", False)) or storage == "video"
    metadata = RolloutMetadata(
        fps=manifest["fps"],
        model_id=manifest["model_id"],
        seed=manifest.get("seed"),
        resolution=tuple(resolution) if resolution is not None else None,
        camera_id=manifest.get("camera_id"),
        lossy_source=lossy_source,
        inference_latency_s=manifest.get("inference_latency_s"),
        context_id=manifest.get("context_id"),
        is_failure=manifest.get("is_failure"),
    )
    return Rollout(
        modality=modality,
        context=context,
        actions=actions,
        predictions=predictions,
        metadata=metadata,
        ground_truth=ground_truth,
    )


def iter_rollouts(root: str | Path) -> Iterator[Rollout]:
    """Yield every rollout under ``root`` (each subdirectory holding a manifest).

    Directories are visited in sorted order. A subdirectory without a manifest
    is skipped, so a mixed tree does not crash the walk.
    """
    root = Path(root)
    if (root / _MANIFEST_NAME).is_file():
        yield load_rollout(root)
        return
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / _MANIFEST_NAME).is_file():
            yield load_rollout(child)


def _sample_dirs(predictions_dir: Path, storage: str) -> list[Path]:
    if storage == "video":
        return sorted(predictions_dir.glob("sample_*.mp4"), key=_index_of)
    return sorted(
        (p for p in predictions_dir.glob("sample_*") if p.is_dir()), key=_index_of
    )


def _read_pixel_stream(
    location: Path, storage: str, channels: int | None
) -> np.ndarray:
    if storage == "png_sequence":
        return _read_frames(location, channels)
    if storage == "video":
        video_path = location if location.suffix else location.with_suffix(".mp4")
        return _read_video(video_path)
    raise ValueError(
        f"unknown pixel storage mode {storage!r}; expected 'png_sequence' or 'video'"
    )


def _read_frames(directory: Path, channels: int | None) -> np.ndarray:
    frame_paths = sorted(directory.glob("frame_*.png"), key=_index_of)
    if not frame_paths:
        raise FileNotFoundError(f"no frames found in {directory}")
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            array = np.asarray(image)
        if array.ndim == 2:
            array = array[..., None]
        frames.append(array)
    stacked = np.stack(frames, axis=0).astype(np.uint8)
    if channels is not None and stacked.shape[-1] != channels:
        raise ValueError(
            f"manifest declares {channels} channels but decoded "
            f"{stacked.shape[-1]} from {directory}"
        )
    return stacked


def _read_video(path: Path) -> np.ndarray:
    """Decode a video file to ``(T, H, W, 3)`` uint8 (lossy; lazily imported)."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "reading video-backed rollouts requires the optional 'io' extra; "
            "install it with `pip install worldproof[io]`"
        ) from exc
    frames = [np.asarray(frame) for frame in iio.imiter(path)]
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0).astype(np.uint8)
