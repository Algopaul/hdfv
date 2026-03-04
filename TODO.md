# TODO

## Medium priority

### Type hints on core functions
`frame_rgb`, `simshow`, `svideo`, `histogram_frames`, `histogram_video` have no parameter or
return type annotations. Add them for IDE support and static analysis.

### Fix `assert` in `frame_rgb`
`images.py` line 64 has `assert rgb` in production flow. Replace with:
```python
raise ValueError("Data has >2 dims and rgb=False — pass rgb=True or select a channel.")
```

### Logarithmic / power-law colormap scaling
Scientific data often spans orders of magnitude. Add a `--scale` flag:
- `--scale log` — log-normalize before colormap
- `--scale power 0.5` — power-law normalization

### Max / mean projections along an axis
Very common need: collapse 3D → 2D before visualizing. Add a `--project` flag:
```
hdfv video file.h5 field out.mp4 --project max --project-axis 2
```

## Lower priority

### Config file / presets
Save commonly-used `--slice`, `--perm`, `--vmin/vmax` combos to a YAML preset file,
selectable with `--preset <name>`.

### More output formats
- GIF (easier to share than MP4)
- WebP for images

### Interactive explore mode
A `hdfv explore` command using `matplotlib.widgets.Slider` for interactive time-axis
scrubbing without writing a file first.

### Batch mode
Process multiple fields from one file in a single command:
```
hdfv video file.h5 field1 field2 out_dir/
```

### Video annotations
Overlay colorbar, axis labels, and frame number on the video output.

### Extend zarr/HDF5 support to `imshow`, `histvid`, `histims`
Currently only the `video` command supports zarr. The other three commands use h5py
directly. Refactor them to use the `open_dataset` context manager.
