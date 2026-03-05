import numpy as np
import pytest
import h5py
import zarr
from typer.testing import CliRunner

from hdfv.cli import app
from hdfv.histogram_videos import (
    _dot_offsets,
    _to_pixel,
    histogram_frames,
    trace_frames,
    angle_color_coded_frames,
)
from hdfv.images import frame_rgb, tile_batch, grid_shape

runner = CliRunner()

# ---------------------------------------------------------------------------
# _dot_offsets
# ---------------------------------------------------------------------------

def test_dot_offsets_radius_0():
    assert _dot_offsets(0) == [(0, 0)]


def test_dot_offsets_radius_1():
    offsets = _dot_offsets(1)
    # radius-1 circle: (0,0), (±1,0), (0,±1) — corners excluded
    assert (0, 0) in offsets
    assert (1, 0) in offsets
    assert (-1, 0) in offsets
    assert (0, 1) in offsets
    assert (0, -1) in offsets
    assert (1, 1) not in offsets  # diagonal: sqrt(2) > 1


def test_dot_offsets_symmetry():
    for r in range(4):
        offsets = _dot_offsets(r)
        mirror = {(-dy, -dx) for dy, dx in offsets}
        assert mirror == set(map(tuple, offsets))


# ---------------------------------------------------------------------------
# _to_pixel
# ---------------------------------------------------------------------------

def test_to_pixel_center():
    px, py = _to_pixel(np.array([0.0]), np.array([0.0]), (-1, 1), (-1, 1), 101)
    assert px[0] == 50
    assert py[0] == 50


def test_to_pixel_corners():
    x = np.array([-1.0, 1.0])
    y = np.array([-1.0, 1.0])
    px, py = _to_pixel(x, y, (-1, 1), (-1, 1), 100)
    assert px[0] == 0 and px[1] == 99
    assert py[0] == 0 and py[1] == 99


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


# ---------------------------------------------------------------------------
# histogram_frames
# ---------------------------------------------------------------------------

def test_histogram_frames_shape_and_dtype():
    data = np.random.uniform(-1, 1, (5, 200, 2))
    frames = list(histogram_frames(data, n_bins=64))
    assert len(frames) == 5
    assert frames[0].shape == (64, 64, 3)
    assert frames[0].dtype == np.uint8


def test_histogram_frames_colorscheme():
    data = np.random.uniform(-1, 1, (2, 100, 2))
    f_viridis = list(histogram_frames(data, n_bins=32, colorscheme="viridis"))
    f_plasma = list(histogram_frames(data, n_bins=32, colorscheme="plasma"))
    # different colorschemes should produce different output
    assert not np.array_equal(f_viridis[0], f_plasma[0])


# ---------------------------------------------------------------------------
# trace_frames
# ---------------------------------------------------------------------------

def test_trace_frames_shape_and_dtype():
    data = np.random.uniform(-1, 1, (8, 50, 2))
    frames = list(trace_frames(data, resolution=64))
    assert len(frames) == 8
    assert frames[0].shape == (64, 64, 3)
    assert frames[0].dtype == np.uint8


def test_trace_frames_trail_accumulates():
    data = np.zeros((5, 10, 2))  # all particles at origin
    frames = list(trace_frames(data, resolution=64, trail_decay=1.0, dot_intensity=1.0))
    # brightness should be non-decreasing at origin pixel
    brightness = [f[32, 32].sum() for f in frames]
    assert all(b2 >= b1 for b1, b2 in zip(brightness, brightness[1:]))


def test_trace_frames_dot_radius():
    data = np.zeros((1, 1, 2))  # single particle at origin
    frames_r0 = list(trace_frames(data, resolution=64, dot_radius=0))
    frames_r3 = list(trace_frames(data, resolution=64, dot_radius=3))
    # larger radius → more lit pixels
    lit_r0 = np.count_nonzero(frames_r0[0].sum(axis=-1))
    lit_r3 = np.count_nonzero(frames_r3[0].sum(axis=-1))
    assert lit_r3 > lit_r0


# ---------------------------------------------------------------------------
# angle_color_coded_frames
# ---------------------------------------------------------------------------

def test_angle_color_coded_frames_shape_and_dtype():
    data = np.random.uniform(-1, 1, (6, 40, 2))
    source = np.random.uniform(-1, 1, (40, 2))
    frames = list(angle_color_coded_frames(data, source, resolution=64))
    assert len(frames) == 6
    assert frames[0].shape == (64, 64, 3)
    assert frames[0].dtype == np.uint8


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

@pytest.fixture
def h5_particles(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f["particles"] = np.random.uniform(-1, 1, (10, 100, 2))
        f["field"] = np.random.rand(20, 32, 32)
    return path


@pytest.fixture
def zarr_particles(tmp_path):
    path = tmp_path / "data.zarr"
    store = zarr.open_group(str(path), mode="w")
    store["particles"] = np.random.uniform(-1, 1, (10, 100, 2))  # type: ignore[index]
    store["field"] = np.random.rand(20, 32, 32)  # type: ignore[index]
    return path


def test_histvid(h5_particles, tmp_path):
    out = tmp_path / "out.mp4"
    result = runner.invoke(app, [
        "histvid", str(h5_particles), "particles", str(out),
        "--n-bins", "32", "--xlim", "-1,1", "--ylim", "-1,1", "--vmax", "5",
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_histims(h5_particles, tmp_path):
    out_base = tmp_path / "hist"
    result = runner.invoke(app, [
        "histims", str(h5_particles), "particles", str(out_base),
        "--n-bins", "32", "--vmax", "3",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "hist_000.png").exists()


def test_video(h5_particles, tmp_path):
    out = tmp_path / "out.mp4"
    result = runner.invoke(app, [
        "video", str(h5_particles), "field", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_imshow(h5_particles, tmp_path):
    out_base = tmp_path / "frame"
    result = runner.invoke(app, [
        "imshow", str(h5_particles), "field", str(out_base),
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "frame_000.png").exists()


def test_tracevid(h5_particles, tmp_path):
    out = tmp_path / "trace.mp4"
    result = runner.invoke(app, [
        "tracevid", str(h5_particles), "particles", str(out),
        "--resolution", "64",
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_anglevid(h5_particles):
    # source_field must be (n_particles, 2) — store separately
    path = h5_particles.parent / "data2.h5"
    with h5py.File(path, "w") as f:
        f["particles"] = np.random.uniform(-1, 1, (10, 100, 2))
        f["source"] = np.random.uniform(-1, 1, (100, 2))
    out = h5_particles.parent / "angle.mp4"
    result = runner.invoke(app, [
        "anglevid", str(path), "particles", "source", str(out),
        "--resolution", "64",
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_histvid_zarr(zarr_particles, tmp_path):
    out = tmp_path / "out.mp4"
    result = runner.invoke(app, [
        "histvid", str(zarr_particles), "particles", str(out),
        "--n-bins", "32",
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_unsupported_format(tmp_path):
    bad = tmp_path / "data.csv"
    bad.write_text("fake")
    result = runner.invoke(app, ["video", str(bad), "field", str(tmp_path / "out.mp4")])
    assert result.exit_code != 0
