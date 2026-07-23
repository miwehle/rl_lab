import numpy as np
import pytest

from nn_viz.video import _crop_to_visible_alpha, compose_bottom_overlay


def test_compose_bottom_overlay_blends_only_bottom_band():
    frame = np.full((4, 3, 3), 100, dtype=np.uint8)
    overlay = np.zeros((2, 3, 4), dtype=np.uint8)
    overlay[:, :, 0] = 200
    overlay[:, :, 3] = 255

    composed = compose_bottom_overlay(frame, overlay, alpha=0.5)

    assert composed.dtype == np.uint8
    np.testing.assert_array_equal(composed[:2], frame[:2])
    assert np.all(composed[2:, :, 0] == 150)
    assert np.all(composed[2:, :, 1:] == 50)


def test_compose_bottom_overlay_requires_matching_width():
    frame = np.zeros((4, 3, 3), dtype=np.uint8)
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="overlay width"):
        compose_bottom_overlay(frame, overlay, alpha=0.5)


def test_crop_to_visible_alpha_removes_transparent_margins():
    rgba = np.zeros((5, 6, 4), dtype=np.uint8)
    rgba[1:4, 2:5, 3] = 255

    cropped = _crop_to_visible_alpha(rgba)

    assert cropped.shape == (3, 3, 4)
    assert np.all(cropped[:, :, 3] == 255)
