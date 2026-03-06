# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test
pytest tests/test_cli.py::test_video

# Run the CLI directly
hdfv --help

# Lint and format
ruff check --fix .
ruff format .
```

Linter and formatter: **ruff** (config in `pyproject.toml` under `[tool.ruff]`). Rules: E, F, I.

## Architecture

`hdfv` is a CLI tool (entry point `hdfv.cli:main`) built with [Typer](https://typer.tiangolo.com/). All commands live in `src/hdfv/cli.py`; rendering logic lives in two modules:

- **`src/hdfv/images.py`** — field/image rendering: `frame_rgb` (normalise → colormap → uint8), `simshow` (images), `svideo` (video), `annotate_frame` (colorbar + frame number overlays via PIL), `tile_batch` (grid layout).
- **`src/hdfv/histogram_videos.py`** — particle rendering: histogram frames/video (`histogram_frames`, `histogram_video`, `mhistims`), trace frames/video (`trace_frames`, `trace_video`), angle-colored frames/video (`angle_color_coded_frames`, `angle_color_coded_video`).

**File format abstraction** is handled entirely by `open_dataset(file, field)` in `cli.py` — a `@contextmanager` that opens `.zarr` via `zarr.open_group` or `.h5`/`.hdf5` via `h5py.File`. All commands use this; neither rendering module touches h5py or zarr directly.

**Data conventions:**
- Field commands (`video`, `imshow`): shape `(n_frames, H, W)` or `(n_frames, H, W, C)`, optionally `(B, n_frames, H, W)` with `--grid`
- Particle commands (`histvid`, `histims`, `tracevid`, `anglevid`): shape `(n_frames, n_particles, 2)`

**CLI panels** (Typer `rich_help_panel`): "Fields" groups `video`/`imshow`; "Particles" groups `histvid`/`histims`/`tracevid`/`anglevid`.

**`frame_rgb` return value** is `(rgb_array, vmin_used, vmax_used)` — callers need to unpack all three. When `rgb=True`, `vmin_used` and `vmax_used` are `None`.

**External dependency**: `hdfx` (private git package) provides `parse_slice` used for `--slice` argument parsing.

## Tests

`tests/conftest.py` provides two fixtures — `h5_particles` and `zarr_particles` — that create small in-memory temp files with `particles` `(10, 100, 2)` and `field` `(20, 32, 32)` datasets. CLI tests use `typer.testing.CliRunner` and invoke `app` directly.
