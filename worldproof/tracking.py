"""Object detection/tracking for the invariant metrics.

The invariant metrics (object-count conservation, object permanence) need to
find objects in each frame. This module defines a lightweight :class:`Tracker`
protocol and a pure-numpy reference implementation, :class:`BlobTracker`, that
segments foreground blobs from the background and returns per-frame detections.

``BlobTracker`` is deterministic and needs only the core install, so the
invariant metrics are testable everywhere. It is intended for *clean* scenes
(distinct objects on a simple background); messy real video wants a pinned
lightweight segmenter (SAM variant) + point tracker behind the ``trackers``
extra, which slots in behind this same interface (CLAUDE.md pins are still TBD).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["Detection", "Tracker", "BlobTracker"]


@dataclass(frozen=True)
class Detection:
    """One detected object in a frame."""

    centroid: tuple[float, float]  # (row, col)
    area: int  # pixel count


class Tracker(ABC):
    """Per-frame object detector for pixel frames."""

    @abstractmethod
    def detect(self, frames: np.ndarray) -> tuple[tuple[Detection, ...], ...]:
        """Detect objects in each frame of ``(T, H, W, C)`` uint8 ``frames``.

        Returns a tuple with one entry per frame, each a tuple of
        :class:`Detection`.
        """


def _connected_components(mask: np.ndarray) -> np.ndarray:
    """Label 4-connected components of a boolean mask (pure numpy).

    Returns an int array where each foreground pixel holds its component's id
    (the min raster index in that component) and background pixels hold 0.
    """
    height, width = mask.shape
    background = height * width + 1
    ids = np.where(
        mask, np.arange(1, height * width + 1).reshape(height, width), background
    )
    while True:
        previous = ids
        up = np.full_like(ids, background)
        up[1:, :] = ids[:-1, :]
        down = np.full_like(ids, background)
        down[:-1, :] = ids[1:, :]
        left = np.full_like(ids, background)
        left[:, 1:] = ids[:, :-1]
        right = np.full_like(ids, background)
        right[:, :-1] = ids[:, 1:]
        merged = np.minimum.reduce([ids, up, down, left, right])
        ids = np.where(mask, merged, background)
        if np.array_equal(ids, previous):
            break
    return np.where(mask, ids, 0)


class BlobTracker(Tracker):
    """Foreground-blob detector: background from the border, blobs by connectivity.

    Args:
        threshold: Sum-over-channels colour distance from the background above
            which a pixel is foreground.
        min_area: Blobs smaller than this many pixels are discarded (noise).
    """

    def __init__(self, *, threshold: int = 40, min_area: int = 4) -> None:
        self._threshold = threshold
        self._min_area = min_area

    def _foreground(self, frame: np.ndarray) -> np.ndarray:
        pixels = frame.astype(np.int32)
        border = np.concatenate(
            [pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]], axis=0
        )
        background = np.median(border, axis=0)
        distance = np.abs(pixels - background).sum(axis=-1)
        return distance > self._threshold

    def _detect_frame(self, frame: np.ndarray) -> tuple[Detection, ...]:
        labels = _connected_components(self._foreground(frame))
        detections = []
        for label in np.unique(labels):
            if label == 0:
                continue
            rows, cols = np.where(labels == label)
            if rows.size < self._min_area:
                continue
            detections.append(
                Detection(
                    centroid=(float(rows.mean()), float(cols.mean())),
                    area=int(rows.size),
                )
            )
        return tuple(detections)

    def detect(self, frames: np.ndarray) -> tuple[tuple[Detection, ...], ...]:
        frames = np.asarray(frames)
        if frames.ndim != 4:
            raise ValueError(
                f"frames must be (T, H, W, C) pixel frames, got shape {frames.shape}"
            )
        return tuple(self._detect_frame(frame) for frame in frames)
