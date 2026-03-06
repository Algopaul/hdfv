import h5py
import numpy as np
from typer.testing import CliRunner

from hdfv.cli import app

runner = CliRunner()


def test_histvid(h5_particles, tmp_path):
    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "histvid",
            str(h5_particles),
            "particles",
            str(out),
            "--n-bins",
            "32",
            "--xlim",
            "-1,1",
            "--ylim",
            "-1,1",
            "--vmax",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_histims(h5_particles, tmp_path):
    out_base = tmp_path / "hist"
    result = runner.invoke(
        app,
        [
            "histims",
            str(h5_particles),
            "particles",
            str(out_base),
            "--n-bins",
            "32",
            "--vmax",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "hist_000.png").exists()


def test_video(h5_particles, tmp_path):
    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "video",
            str(h5_particles),
            "field",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_imshow(h5_particles, tmp_path):
    out_base = tmp_path / "frame"
    result = runner.invoke(
        app,
        [
            "imshow",
            str(h5_particles),
            "field",
            str(out_base),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "frame_000.png").exists()


def test_tracevid(h5_particles, tmp_path):
    out = tmp_path / "trace.mp4"
    result = runner.invoke(
        app,
        [
            "tracevid",
            str(h5_particles),
            "particles",
            str(out),
            "--resolution",
            "64",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_anglevid(h5_particles):
    # source_field must be (n_particles, 2) — store separately
    path = h5_particles.parent / "data2.h5"
    with h5py.File(path, "w") as f:
        f["particles"] = np.random.uniform(-1, 1, (10, 100, 2))
        f["source"] = np.random.uniform(-1, 1, (100, 2))
    out = h5_particles.parent / "angle.mp4"
    result = runner.invoke(
        app,
        [
            "anglevid",
            str(path),
            "particles",
            "source",
            str(out),
            "--resolution",
            "64",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_histvid_zarr(zarr_particles, tmp_path):
    out = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        [
            "histvid",
            str(zarr_particles),
            "particles",
            str(out),
            "--n-bins",
            "32",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_unsupported_format(tmp_path):
    bad = tmp_path / "data.csv"
    bad.write_text("fake")
    result = runner.invoke(app, ["video", str(bad), "field", str(tmp_path / "out.mp4")])
    assert result.exit_code != 0
