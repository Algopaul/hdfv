import numpy as np
import pytest

from hdfv.images import frame_rgb, grid_shape, tile_batch

# ---------------------------------------------------------------------------
# frame_rgb
# ---------------------------------------------------------------------------


def test_frame_rgb_2d():
    x = np.linspace(0, 1, 64).reshape(8, 8)
    out = frame_rgb(x)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_frame_rgb_channel():
    x = np.random.rand(8, 8, 3).astype(np.float32)
    out = frame_rgb(x, channel=1)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_frame_rgb_vmin_vmax():
    x = np.ones((4, 4)) * 0.5
    out = frame_rgb(x, vmin=0.0, vmax=1.0)
    # all pixels should be the same color
    assert np.all(out == out[0, 0])


def test_frame_rgb_3d_no_channel_raises():
    x = np.random.rand(8, 8, 3).astype(np.float32)
    with pytest.raises(ValueError, match="--channel"):
        frame_rgb(x, channel=None, rgb=False)


def test_frame_rgb_rgb_passthrough():
    x = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    out = frame_rgb(x, rgb=True)
    assert np.array_equal(out, x)


# ---------------------------------------------------------------------------
# tile_batch
# ---------------------------------------------------------------------------


def test_tile_batch_shape():
    batch = np.zeros((6, 4, 4))
    out = tile_batch(batch, nrows=2, ncols=3)
    assert out.shape == (8, 12)


def test_tile_batch_with_channels():
    batch = np.zeros((4, 4, 4, 3))
    out = tile_batch(batch, nrows=2, ncols=2)
    assert out.shape == (8, 8, 3)


def test_tile_batch_partial():
    batch = np.ones((3, 4, 4))
    out = tile_batch(batch, nrows=2, ncols=2, pad=0.0)
    assert out.shape == (8, 8)
    # bottom-right tile should be padding
    assert np.all(out[4:, 4:] == 0.0)


# ---------------------------------------------------------------------------
# grid_shape
# ---------------------------------------------------------------------------


def test_grid_shape_perfect_square():
    nrows, ncols = grid_shape(9)
    assert nrows * ncols >= 9
    assert ncols == 3


def test_grid_shape_non_square():
    nrows, ncols = grid_shape(7)
    assert nrows * ncols >= 7
