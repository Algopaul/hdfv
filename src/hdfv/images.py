from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps


def _frame_rgb(x, channel, scale_factor, vmin, vmax, cmap):
  x = scale_factor * np.array(x)
  vvmin = np.min(x) if vmin is None else vmin
  vvmax = np.max(x) if vmax is None else vmax
  x = np.clip(x, vvmin, vvmax)
  x -= vvmin
  x /= (vvmax - vvmin)
  if channel >= 0:
    x = x[..., channel]
    rgb = (255 * cmap(x)[..., :3]).astype(np.uint8)
  elif x.ndim == 2:
    rgb = (255 * cmap(x)[..., :3]).astype(np.uint8)
  else:
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
):
  cmap = colormaps[colorscheme]
  dir = Path(outfile_base).parent
  dir.mkdir(exist_ok=True, parents=True)
  for i, x in enumerate(data):
    rgb = _frame_rgb(x, channel, scale_factor, vmin, vmax, cmap)
    imageio.imwrite(str(outfile_base) + f'_{i:03d}.png', rgb)


def svideo(
    data,
    outfile,
    *,
    channel: int = 0,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = 'viridis',
    fps: int = 30,
):
  cmap = colormaps[colorscheme]
  writer = imageio.get_writer(outfile, fps=fps)
  for x in data:
    rgb = _frame_rgb(x, channel, scale_factor, vmin, vmax, cmap)
    writer.append_data(rgb)
  writer.close()
