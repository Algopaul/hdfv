from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import typer
import zarr
from hdfx.cli import parse_slice
from rich.console import Console

from hdfv.histogram_videos import (
    angle_color_coded_video,
    histogram_video,
    mhistims,
    trace_video,
)
from hdfv.images import simshow, svideo

app = typer.Typer(help="Unix-style tools for working with HDF5 / Zarr")
console = Console()
err_console = Console(stderr=True)


@app.command(rich_help_panel="Particles")
def histvid(
    file: Path,
    field: str,
    outfile,
    *,
    fps: int = 30,
    n_bins: int = 256,
    xlim: str = "-1,1",
    ylim: str = "-1,1",
    vmax: float = 4.0,
    colorscheme: str = "viridis",
):
    """Particle density histogram video. FIELD shape: (n_frames, n_particles, 2)."""
    with open_dataset(file, field) as dset:
        histogram_video(
            dset,
            outfile,
            fps=fps,
            n_bins=n_bins,
            xlim=_parse_lim(xlim),
            ylim=_parse_lim(ylim),
            vmax=vmax,
            colorscheme=colorscheme,
        )


@app.command(rich_help_panel="Particles")
def histims(
    file: Path,
    field: str,
    outfile_base,
    *,
    n_bins: int = 256,
    xlim: str = "-1,1",
    ylim: str = "-1,1",
    vmax: float = 3.0,
    colorscheme: str = "viridis",
):
    """Particle density histogram images. FIELD shape: (n_frames, n_particles, 2). Writes OUTFILE_BASE_NNN.png."""
    with open_dataset(file, field) as dset:
        mhistims(
            dset,
            outfile_base,
            n_bins=n_bins,
            xlim=_parse_lim(xlim),
            ylim=_parse_lim(ylim),
            vmax=vmax,
            colorscheme=colorscheme,
        )


@app.command(rich_help_panel="Fields")
def imshow(
    file: Path,
    field: str,
    outfile_base: Path,
    *,
    channel: int = 0,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = "viridis",
    slice: Optional[str] = None,
    colorbar: bool = False,
    frame_number: bool = False,
):
    """Save field frames as images. FIELD shape: (n_frames, H, W) or (n_frames, H, W, C). Writes OUTFILE_BASE_NNN.png."""
    with open_dataset(file, field) as dset:
        if slice:
            sel = parse_slice(slice)
            dset = dset[sel]  # type: ignore[index]
        simshow(
            dset,
            outfile_base,
            channel=channel,
            scale_factor=scale_factor,
            vmin=vmin,
            vmax=vmax,
            colorscheme=colorscheme,
            colorbar=colorbar,
            frame_number=frame_number,
        )


def _parse_lim(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


@contextmanager
def open_dataset(filename: Path, field: str):
    p = Path(filename)
    suffix = p.suffix.lower()
    if suffix == ".zarr":
        store = zarr.open_group(p, mode="r")
        yield store[field]
    elif suffix in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            yield f[field]
    else:
        raise ValueError(
            f'Unsupported format "{suffix}". Expected .zarr, .h5, or .hdf5'
        )


@app.command(rich_help_panel="Fields")
def video(
    file: Path,
    field: str,
    outfile_base: Path,
    *,
    rgb: bool = False,
    grid: bool = False,
    batch: bool = False,
    ncols: Optional[int] = None,
    channel: Optional[int] = None,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = "viridis",
    fps: int = 30,
    perm: Optional[str] = None,
    slice: Optional[str] = None,
    colorbar: bool = False,
    frame_number: bool = False,
):
    """Render field data as a video. FIELD shape: (n_frames, H, W) or (n_frames, H, W, C). Use --grid for batched data (B, n_frames, H, W) as a tiled grid, or --batch to write one MP4 per batch item."""
    with open_dataset(file, field) as dset:
        if perm is not None:
            p = tuple(int(i) for i in perm.split(","))
            dset = Permuted(dset, p)
        if slice:
            sel = parse_slice(slice)
            dset = dset[sel]  # type: ignore[index]

        svideo(
            dset,
            outfile_base,
            rgb=rgb,
            grid=grid,
            batch=batch,
            ncols=ncols,
            channel=channel,
            scale_factor=scale_factor,
            vmin=vmin,
            vmax=vmax,
            colorscheme=colorscheme,
            fps=fps,
            colorbar=colorbar,
            frame_number=frame_number,
        )


class Permuted:
    def __init__(self, dset, perm):
        self.dset = dset
        self.perm = tuple(perm)
        self.inv = tuple(np.argsort(self.perm))
        self.shape = tuple(dset.shape[p] for p in self.perm)
        self.ndim = len(self.shape)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        # pad with full slices
        idx = idx + (slice(None),) * (self.ndim - len(idx))

        # map back to original axes
        src = tuple(idx[self.inv[i]] for i in range(self.ndim))
        return self.dset[src]


@app.command(rich_help_panel="Particles")
def tracevid(
    file: Path,
    field: str,
    outfile: Path,
    *,
    resolution: int = 512,
    xlim: str = "-1,1",
    ylim: str = "-1,1",
    trail_decay: float = 0.92,
    dot_intensity: float = 1.0,
    dot_radius: int = 0,
    fps: int = 30,
):
    """Particle trace video with fading trails. data: (n_time, n_particles, 2)"""
    with open_dataset(file, field) as dset:
        trace_video(
            dset,
            outfile,
            resolution=resolution,
            xlim=_parse_lim(xlim),
            ylim=_parse_lim(ylim),
            trail_decay=trail_decay,
            dot_intensity=dot_intensity,
            dot_radius=dot_radius,
            fps=fps,
        )


@app.command(rich_help_panel="Particles")
def anglevid(
    file: Path,
    field: str,
    source_field: str,
    outfile: Path,
    *,
    resolution: int = 512,
    xlim: str = "-1,1",
    ylim: str = "-1,1",
    dot_radius: int = 0,
    fps: int = 30,
):
    """Particle video colored by angle of a reference vector field. data: (n_time, n_particles, 2)"""
    with open_dataset(file, field) as dset:
        with open_dataset(file, source_field) as src:
            angle_color_coded_video(
                dset,
                src,
                outfile,
                resolution=resolution,
                xlim=_parse_lim(xlim),
                ylim=_parse_lim(ylim),
                dot_radius=dot_radius,
                fps=fps,
            )


def main():
    app()
