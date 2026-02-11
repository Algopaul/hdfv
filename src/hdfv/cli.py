from pathlib import Path
from typing import Optional, cast

import h5py
import numpy as np
import typer
from hdfx.cli import parse_slice
from rich.console import Console

from hdfv.histogram_videos import histogram_video, mhistims
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
  with h5py.File(file, 'r') as f:
    histogram_video(f[field], outfile, fps=fps, colorscheme=colorscheme)


@app.command()
def histims(file: Path,
            field: str,
            outfile_base,
            *,
            colorscheme: str = 'viridis'):
  with h5py.File(file, 'r') as f:
    mhistims(f[field], outfile_base, colorscheme=colorscheme)


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
  with h5py.File(file, 'r') as f:
    dset = cast(h5py.Dataset, f[field])
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
  with h5py.File(file, 'r') as f:
    dset = cast(h5py.Dataset, f[field])
    if perm is not None:
      p = tuple(int(i) for i in perm.split(","))
      dset = Permuted(dset, p)
    if slice:
      sel = parse_slice(slice)
      dset = dset[sel]

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


def main():
  app()
