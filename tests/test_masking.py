"""Tests for dynamic-region masking."""

from __future__ import annotations

import numpy as np
import pytest

from worldproof.masking import dynamic_region_mask


def _moving_square_sequence(t_steps=4, size=20):
    frames = np.full((t_steps, size, size, 3), 100, dtype=np.uint8)
    for t in range(t_steps):
        frames[t, t : t + 4, t : t + 4] = 255  # a square slides diagonally
    return frames


def test_mask_shape_and_dtype():
    mask = dynamic_region_mask(_moving_square_sequence())
    assert mask.shape == (4, 20, 20)
    assert mask.dtype == bool


def test_mask_marks_motion_not_static_background():
    mask = dynamic_region_mask(_moving_square_sequence())
    assert mask.any()  # the moving square is caught
    assert not mask[:, -1, -1].any()  # a far corner never moves


def test_static_sequence_has_empty_mask():
    frames = np.full((4, 16, 16, 3), 128, dtype=np.uint8)
    assert not dynamic_region_mask(frames).any()


def test_single_frame_is_all_dynamic():
    frames = np.full((1, 8, 8, 3), 128, dtype=np.uint8)
    assert dynamic_region_mask(frames).all()


def test_requires_pixel_frames():
    with pytest.raises(ValueError, match=r"\(T, H, W, C\)"):
        dynamic_region_mask(np.zeros((4, 8, 8), dtype=np.uint8))


def test_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        dynamic_region_mask(np.zeros((4, 8, 8, 3), dtype=np.float32))
