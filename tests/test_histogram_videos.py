import numpy as np

from hdfv.histogram_videos import (
    _dot_offsets,
    _to_pixel,
    angle_color_coded_frames,
    histogram_frames,
    trace_frames,
)

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
# histogram_frames
# ---------------------------------------------------------------------------


def test_histogram_frames_shape_and_dtype():
    data = np.random.uniform(-1, 1, (5, 200, 2))
    frames = list(histogram_frames(data, n_bins=64))
    assert len(frames) == 5
    assert frames[0].data.shape == (64, 64, 3)
    assert frames[0].data.dtype == np.uint8


def test_histogram_frames_colorscheme():
    data = np.random.uniform(-1, 1, (2, 100, 2))
    f_viridis = list(histogram_frames(data, n_bins=32, colorscheme="viridis"))
    f_plasma = list(histogram_frames(data, n_bins=32, colorscheme="plasma"))
    # different colorschemes should produce different output
    assert not np.array_equal(f_viridis[0].data, f_plasma[0].data)


# ---------------------------------------------------------------------------
# trace_frames
# ---------------------------------------------------------------------------


def test_trace_frames_shape_and_dtype():
    data = np.random.uniform(-1, 1, (8, 50, 2))
    frames = list(trace_frames(data, resolution=64))
    assert len(frames) == 8
    assert frames[0].data.shape == (64, 64, 3)
    assert frames[0].data.dtype == np.uint8


def test_trace_frames_trail_accumulates():
    data = np.zeros((5, 10, 2))  # all particles at origin
    frames = list(trace_frames(data, resolution=64, trail_decay=1.0, dot_intensity=1.0))
    # brightness should be non-decreasing at origin pixel
    brightness = [f.data[32, 32].sum() for f in frames]
    assert all(b2 >= b1 for b1, b2 in zip(brightness, brightness[1:]))


def test_trace_frames_dot_radius():
    data = np.zeros((1, 1, 2))  # single particle at origin
    frames_r0 = list(trace_frames(data, resolution=64, dot_radius=0))
    frames_r3 = list(trace_frames(data, resolution=64, dot_radius=3))
    # larger radius → more lit pixels
    lit_r0 = np.count_nonzero(frames_r0[0].data.sum(axis=-1))
    lit_r3 = np.count_nonzero(frames_r3[0].data.sum(axis=-1))
    assert lit_r3 > lit_r0


# ---------------------------------------------------------------------------
# angle_color_coded_frames
# ---------------------------------------------------------------------------


def test_angle_color_coded_frames_shape_and_dtype():
    data = np.random.uniform(-1, 1, (6, 40, 2))
    source = np.random.uniform(-1, 1, (40, 2))
    frames = list(angle_color_coded_frames(data, source, resolution=64))
    assert len(frames) == 6
    assert frames[0].data.shape == (64, 64, 3)
    assert frames[0].data.dtype == np.uint8
