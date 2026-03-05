# hdfv

Command-line tools for visualizing HDF5 and Zarr datasets as images and videos.

## Installation

```bash
pip install git+ssh://git@github.com/Algopaul/hdfv.git
```

Or clone and install locally:

```bash
git clone git@github.com:Algopaul/hdfv.git
cd hdfv
pip install -e .
```

After installation, shell completion can be enabled with:

```bash
hdfv --install-completion
```

## Commands

All commands accept `.h5`, `.hdf5`, and `.zarr` files.

### `video` — field data to video

Renders a dataset of shape `(n_frames, H, W)` as an MP4.

```bash
hdfv video data.h5 field_name output.mp4
```

| Option | Default | Description |
|---|---|---|
| `--fps INT` | 30 | Frames per second |
| `--vmin FLOAT` | auto | Colormap lower bound |
| `--vmax FLOAT` | auto | Colormap upper bound |
| `--colorscheme STR` | `viridis` | Matplotlib colormap name |
| `--channel INT` | — | Select a channel from the last axis |
| `--rgb` | false | Treat last axis as RGB directly |
| `--grid` | false | Tile a batch dimension; data shape `(B, n_frames, H, W)` |
| `--ncols INT` | auto | Columns when using `--grid` |
| `--scale-factor FLOAT` | 1.0 | Multiply values before normalization |
| `--perm STR` | — | Permute axes before rendering, e.g. `1,0,2` |
| `--slice STR` | — | NumPy-style slice applied before rendering |

### `imshow` — field data to images

Saves each frame as a numbered PNG: `outfile_base_000.png`, `outfile_base_001.png`, ...

```bash
hdfv imshow data.h5 field_name output/frame
```

Accepts the same options as `video` (except `--fps`, `--grid`, `--perm`).

### `histvid` — particle histogram video

Renders particle trajectories of shape `(n_frames, n_particles, 2)` as a 2D density histogram video.

```bash
hdfv histvid data.h5 particles output.mp4
```

| Option | Default | Description |
|---|---|---|
| `--fps INT` | 30 | Frames per second |
| `--n-bins INT` | 256 | Histogram resolution |
| `--xlim STR` | `-1,1` | x-axis range |
| `--ylim STR` | `-1,1` | y-axis range |
| `--vmax FLOAT` | 4.0 | Particle count mapped to full brightness |
| `--colorscheme STR` | `viridis` | Matplotlib colormap name |

### `histims` — particle histogram images

Saves one histogram image per frame as numbered PNGs.

```bash
hdfv histims data.h5 particles output/hist
```

Accepts the same options as `histvid` (except `--fps`). Default `--vmax` is 3.0.

### `tracevid` — particle trace video

Renders particles as bright dots with a fading trail effect. Data shape: `(n_frames, n_particles, 2)`.

```bash
hdfv tracevid data.h5 particles output.mp4
```

| Option | Default | Description |
|---|---|---|
| `--fps INT` | 30 | Frames per second |
| `--resolution INT` | 512 | Output image size in pixels |
| `--xlim STR` | `-1,1` | x-axis range |
| `--ylim STR` | `-1,1` | y-axis range |
| `--trail-decay FLOAT` | 0.92 | Per-frame brightness decay for the trail (0–1) |
| `--dot-intensity FLOAT` | 1.0 | Brightness added per particle per frame |
| `--dot-radius INT` | 0 | Filled-circle radius for each dot in pixels |

### `anglevid` — angle-colored particle video

Renders particles colored by the angle of a reference vector field (e.g. initial velocity direction), using the HSV colormap. Requires a position dataset of shape `(n_frames, n_particles, 2)` and a reference vector dataset of shape `(n_particles, 2)` in the same file.

```bash
hdfv anglevid data.h5 particles source_field output.mp4
```

| Option | Default | Description |
|---|---|---|
| `--fps INT` | 30 | Frames per second |
| `--resolution INT` | 512 | Output image size in pixels |
| `--xlim STR` | `-1,1` | x-axis range |
| `--ylim STR` | `-1,1` | y-axis range |
| `--dot-radius INT` | 0 | Filled-circle radius for each dot in pixels |

## Development

```bash
pip install -e ".[dev]"
pytest
```
