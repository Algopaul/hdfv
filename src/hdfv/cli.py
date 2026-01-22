from pathlib import Path

import h5py
import typer
from rich.console import Console

from hdfv.histogram_videos import histogram_video

app = typer.Typer(help="Unix-style tools for working with HDF5")
console = Console()
err_console = Console(stderr=True)


@app.command()
def histvid(file: Path, field: str, outfile, *, fps: int = 60):
  with h5py.File(file, 'r') as f:
    histogram_video(f[field], outfile, fps=fps)


@app.command()
def extra(file: Path, field: str, outfile, *, fps: int):
  with h5py.File(file, 'r') as f:
    histogram_video(f[field], outfile, fps=fps)


def main():
  app()
