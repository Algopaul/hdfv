import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps


def histogram_frames(
    data,
    *,
    n_bins: int = 256,
    xlim=(-1, 1),
    ylim=(-1, 1),
    vmax: float = 4.0,
    colorscheme: str = "viridis",
):
  cmap = colormaps[colorscheme]
  frames = []

  for t in data:
    hist = np.histogram2d(
        t[:, 1],
        t[:, 0],
        bins=n_bins,
        range=[xlim, ylim],
    )[0]

    img = np.clip(hist / vmax, 0, 1)
    rgb = (255 * cmap(img)[..., :3]).astype(np.uint8)
    frames.append(rgb)

  return np.stack(frames)


def histogram_video(data,
                    outfile,
                    *,
                    n_bins: int = 256,
                    xlim=(-1, 1),
                    ylim=(-1, 1),
                    vmax: float = 4.0,
                    fps: int = 30,
                    colorscheme: str = 'viridis'):

  vid_writer = imageio.get_writer(outfile, fps=fps, codec="libx264")
  frames = histogram_frames(
      data,
      n_bins=n_bins,
      xlim=xlim,
      ylim=ylim,
      vmax=vmax,
      colorscheme=colorscheme,
  )
  for f in frames:
    vid_writer.append_data(f)

  vid_writer.close()


def mhistims(data, outfile_base, *, colorscheme: str = 'viridis'):
  cmap = colormaps[colorscheme]
  for i, t in enumerate(data):
    n_bins = 256
    traj = np.histogram2d(t[:, 1], t[:, 0], n_bins, [[-1, 1], [-1, 1]])[0]
    img = traj.reshape((n_bins, n_bins))
    img = np.clip(img / 3, 0, 1)
    rgb = (255 * cmap(img)[..., :3]).astype(np.uint8)
    imageio.imwrite(str(outfile_base) + f'_{i:03d}.png', rgb)
