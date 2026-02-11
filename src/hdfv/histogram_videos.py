import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps


def histogram_video(data,
                    outfile,
                    *,
                    fps: int = 30,
                    colorscheme: str = 'viridis'):

  vid_writer = imageio.get_writer(outfile, fps=fps, codec="libx264")
  cmap = colormaps[colorscheme]
  for t in data:
    n_bins = 256
    traj = np.histogram2d(t[:, 1], t[:, 0], n_bins, [[-5, 5], [-5, 5]])[0]
    img = traj.reshape((n_bins, n_bins))
    img = np.clip(img / 4, 0, 1)
    rgb = (255 * cmap(img)[..., :3]).astype(np.uint8)
    vid_writer.append_data(rgb)

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
