"""Tests for the reference BlobTracker."""

from __future__ import annotations

import numpy as np
import pytest

from worldproof.tracking import BlobTracker

SIZE = 32
_COLORS = [(220, 30, 30), (30, 180, 30), (30, 30, 220), (200, 150, 10)]


def scene(centers, *, occluder=None):
    frame = np.full((SIZE, SIZE, 3), 255, np.uint8)
    if occluder is not None:
        y0, y1, x0, x1 = occluder
        frame[y0:y1, x0:x1] = (120, 120, 120)
    for i, (y, x) in enumerate(centers):
        frame[y - 1 : y + 2, x - 1 : x + 2] = _COLORS[i % len(_COLORS)]
    return frame


def test_counts_distinct_objects():
    tracker = BlobTracker()
    detections = tracker.detect(scene([(5, 5), (5, 20), (20, 12)])[None])[0]
    assert len(detections) == 3


def test_reports_centroids():
    tracker = BlobTracker()
    (detection,) = tracker.detect(scene([(10, 15)])[None])[0]
    assert round(detection.centroid[0]) == 10
    assert round(detection.centroid[1]) == 15


def test_empty_scene_has_no_detections():
    assert BlobTracker().detect(scene([])[None])[0] == ()


def test_occluded_object_merges_into_occluder():
    tracker = BlobTracker()

    def small_blobs(detections):
        return [d for d in detections if d.area < 50]

    visible = small_blobs(tracker.detect(scene([(5, 5), (16, 16)])[None])[0])
    # hide the second object behind a bar spanning its row
    occluded = small_blobs(
        tracker.detect(scene([(5, 5), (16, 16)], occluder=(13, 20, 0, SIZE))[None])[0]
    )
    assert len(visible) == 2  # two distinct small objects when unoccluded
    assert len(occluded) == 1  # the hidden object is absorbed into the occluder


def test_filters_tiny_noise():
    frame = np.full((SIZE, SIZE, 3), 255, np.uint8)
    frame[10, 10] = (0, 0, 0)  # single pixel < min_area
    assert BlobTracker(min_area=4).detect(frame[None])[0] == ()


def test_requires_pixel_frames():
    with pytest.raises(ValueError, match=r"\(T, H, W, C\)"):
        BlobTracker().detect(np.zeros((32, 32, 3), np.uint8))
