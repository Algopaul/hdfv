import math
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps


def tile_batch(batch: np.ndarray, nrows, ncols, pad=0.0):
  """
  batch: (B, X, Y) or (B, X, Y, C)
  """
  B = batch.shape[0]
  X, Y = batch.shape[1], batch.shape[2]
  C = () if batch.ndim == 3 else (batch.shape[3],)

  out = np.full((nrows * X, ncols * Y, *C), pad, dtype=batch.dtype)

  for i in range(min(nrows * ncols, B)):
    r = i // ncols
    c = i % ncols
    out[r * X:(r + 1) * X, c * Y:(c + 1) * Y, ...] = batch[i]

  return out


def _frame_rgb(
    x,
    channel,
    scale_factor,
    vmin,
    vmax,
    cmap,
    rgb,
    grid,
    nrows,
    ncols,
):
  if channel is not None and rgb:
    raise ValueError("Use either --channel or --rgb, not both.")

  if grid:
    x = tile_batch(x, nrows, ncols)

  if rgb:
    return x
  else:
    x = scale_factor * np.array(x)
    vvmin = np.min(x) if vmin is None else vmin
    vvmax = np.max(x) if vmax is None else vmax
    x = np.clip(x, vvmin, vvmax)
    x -= vvmin
    x /= (vvmax - vvmin)
    if channel is not None and channel >= 0:
      x = x[..., channel]
      rgb = (255 * cmap(x)[..., :3]).astype(np.uint8)
    elif x.ndim == 2:
      rgb = (255 * cmap(x)[..., :3]).astype(np.uint8)
    else:
      assert rgb
      rgb = x
    return rgb


def simshow(
    data,
    outfile_base,
    *,
    channel: int = 0,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = 'viridis',
    rgb: bool = False,
):
  cmap = colormaps[colorscheme]
  dir = Path(outfile_base).parent
  dir.mkdir(exist_ok=True, parents=True)
  for i, x in enumerate(data):
    frame = _frame_rgb(
        x,
        channel,
        scale_factor,
        vmin,
        vmax,
        cmap,
        rgb,
        grid=False,
        nrows=1,
        ncols=1,
    )
    imageio.imwrite(str(outfile_base) + f'_{i:03d}.png', frame)


def grid_shape(B: int) -> tuple[int, int]:
  ncols = int(np.ceil(np.sqrt(B)))
  nrows = int(np.ceil(B / ncols))
  return nrows, ncols


def svideo(
    data,
    outfile,
    *,
    rgb: bool = False,
    grid: bool = False,
    ncols: Optional[int] = None,
    channel: Optional[int] = None,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = 'viridis',
    fps: int = 20,
):
  if ncols is None:
    nrows, ncols = grid_shape(data.shape[1])
    print(data.shape[1])
  else:
    nrows = int(np.ceil(data.shape[0] / ncols))
  cmap = colormaps[colorscheme]
  writer = imageio.get_writer(outfile, fps=fps)
  for x in data:
    frame = _frame_rgb(
        x,
        channel,
        scale_factor,
        vmin,
        vmax,
        cmap,
        rgb,
        grid,
        nrows,
        ncols,
    )
    writer.append_data(frame)
  writer.close()
