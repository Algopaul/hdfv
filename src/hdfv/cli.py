from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import typer
import zarr
from hdfx.cli import parse_slice
from rich.console import Console

from hdfv.histogram_videos import (angle_color_coded_video, histogram_video,
                                   mhistims, trace_video)
from hdfv.images import simshow, svideo

app = typer.Typer(help="Unix-style tools for working with HDF5")
console = Console()
err_console = Console(stderr=True)


@app.command()
def histvid(file: Path,
            field: str,
            outfile,
            *,
            fps: int = 30,
            colorscheme: str = 'viridis'):
  with open_dataset(file, field) as dset:
    histogram_video(dset, outfile, fps=fps, colorscheme=colorscheme)


@app.command()
def histims(file: Path,
            field: str,
            outfile_base,
            *,
            colorscheme: str = 'viridis'):
  with open_dataset(file, field) as dset:
    mhistims(dset, outfile_base, colorscheme=colorscheme)


@app.command()
def imshow(
    file: Path,
    field: str,
    outfile_base: Path,
    *,
    channel: int = 0,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = 'viridis',
    slice: Optional[str] = None,
):
  with open_dataset(file, field) as dset:
    if slice:
      sel = parse_slice(slice)
      dset = dset[sel]
    simshow(
        dset,
        outfile_base,
        channel=channel,
        scale_factor=scale_factor,
        vmin=vmin,
        vmax=vmax,
        colorscheme=colorscheme)


def _parse_lim(s: str) -> tuple[float, float]:
  a, b = s.split(",")
  return float(a), float(b)


@contextmanager
def open_dataset(filename: Path, field: str):
  p = Path(filename)
  suffix = p.suffix.lower()
  if suffix == '.zarr':
    store = zarr.open(p, mode='r')
    yield store[field]
  elif suffix in ('.h5', '.hdf5'):
    with h5py.File(p, 'r') as f:
      yield f[field]
  else:
    raise ValueError(
        f'Unsupported format "{suffix}". Expected .zarr, .h5, or .hdf5')


@app.command()
def video(
    file: Path,
    field: str,
    outfile_base: Path,
    *,
    rgb: bool = False,
    grid: bool = False,
    ncols: Optional[int] = None,
    channel: Optional[int] = None,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = 'viridis',
    fps: int = 30,
    perm: Optional[str] = None,
    slice: Optional[str] = None,
):
  with open_dataset(file, field) as dset:
    if perm is not None:
      p = tuple(int(i) for i in perm.split(","))
      dset = Permuted(dset, p)
    if slice:
      sel = parse_slice(slice)
      dset = dset[sel]  # pyright: ignore

    svideo(
        dset,
        outfile_base,
        rgb=rgb,
        grid=grid,
        ncols=ncols,
        channel=channel,
        scale_factor=scale_factor,
        vmin=vmin,
        vmax=vmax,
        colorscheme=colorscheme,
        fps=fps,
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


@app.command()
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
        fps=fps,
    )


@app.command()
def anglevid(
    file: Path,
    field: str,
    source_field: str,
    outfile: Path,
    *,
    resolution: int = 512,
    xlim: str = "-1,1",
    ylim: str = "-1,1",
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
          fps=fps,
      )


def main():
  app()
