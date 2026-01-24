from pathlib import Path
from typing import Optional

import h5py
import typer
from rich.console import Console

from hdfv.histogram_videos import histogram_video
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
):
  with h5py.File(file, 'r') as f:
    simshow(
        f[field],
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
    channel: int = 0,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = 'viridis',
    fps: int = 30,
):
  with h5py.File(file, 'r') as f:
    svideo(
        f[field],
        outfile_base,
        channel=channel,
        scale_factor=scale_factor,
        vmin=vmin,
        vmax=vmax,
        colorscheme=colorscheme,
        fps=fps,
    )


def main():
  app()
