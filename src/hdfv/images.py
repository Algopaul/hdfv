from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


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
        out[r * X : (r + 1) * X, c * Y : (c + 1) * Y, ...] = batch[i]

    return out


_DEFAULT_FONT = None


def _font():
    global _DEFAULT_FONT
    if _DEFAULT_FONT is None:
        try:
            _DEFAULT_FONT = ImageFont.load_default(size=14)
        except TypeError:
            _DEFAULT_FONT = ImageFont.load_default()
    return _DEFAULT_FONT


def _colorbar_strip(
    cmap,
    vmin: float,
    vmax: float,
    height: int,
    bar_width: int = 16,
    label_width: int = 52,
) -> np.ndarray:
    """Returns (height, bar_width + label_width, 3) uint8 colorbar."""
    gradient = np.linspace(1.0, 0.0, height)[:, np.newaxis]  # top = vmax
    strip = (255 * cmap(gradient)[..., :3]).astype(np.uint8)
    strip = np.repeat(strip, bar_width, axis=1)

    label_panel = np.zeros((height, label_width, 3), dtype=np.uint8)
    cb = np.concatenate([strip, label_panel], axis=1)

    img = Image.fromarray(cb)
    draw = ImageDraw.Draw(img)
    font = _font()
    draw.text((bar_width + 2, 2), f"{vmax:.3g}", fill=(220, 220, 220), font=font)
    draw.text(
        (bar_width + 2, height - 16), f"{vmin:.3g}", fill=(220, 220, 220), font=font
    )
    mid_y = height // 2 - 7
    mid_v = (vmin + vmax) / 2
    draw.text((bar_width + 2, mid_y), f"{mid_v:.3g}", fill=(160, 160, 160), font=font)
    return np.array(img)


@dataclass
class Frame:
    data: np.ndarray
    vrange: tuple[float, float] | None
    cmap: Optional[Colormap] = None

    def annotated(
        self,
        *,
        colorbar: bool = False,
        frame_number: int | None = None,
    ) -> np.ndarray:
        arr = self.data
        if colorbar and self.cmap is not None and self.vrange is not None:
            vmin, vmax = self.vrange
            h = arr.shape[0]
            cb = _colorbar_strip(self.cmap, vmin, vmax, h)
            sep = np.zeros((h, 2, 3), dtype=np.uint8)
            arr = np.concatenate([arr, sep, cb], axis=1)
        if frame_number is not None:
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            font = _font()
            text = f"t={frame_number:04d}"
            draw.text((5, 5), text, fill=(0, 0, 0), font=font)
            draw.text((4, 4), text, fill=(255, 255, 255), font=font)
            arr = np.array(img)
        return arr

    def save(self, path, *, colorbar: bool = False, frame_number: int | None = None):
        imageio.imwrite(
            str(path), self.annotated(colorbar=colorbar, frame_number=frame_number)
        )

    def append_to(
        self, writer, *, colorbar: bool = False, frame_number: int | None = None
    ):
        writer.append_data(self.annotated(colorbar=colorbar, frame_number=frame_number))


def frame_rgb(
    x,
    *,
    channel: Optional[int] = None,
    scale_factor=1.0,
    vmin=None,
    vmax=None,
    cmap=None,
    rgb=None,
    grid=False,
    nrows=-1,
    ncols=-1,
) -> Frame:
    if cmap is None:
        cmap = colormaps["viridis"]
    if channel is not None and rgb:
        raise ValueError("Use either --channel or --rgb, not both.")

    if grid:
        x = tile_batch(x, nrows, ncols)

    if rgb:
        return Frame(x, None)
    else:
        x = scale_factor * np.array(x)
        vvmin = np.min(x) if vmin is None else vmin
        vvmax = np.max(x) if vmax is None else vmax
        x = np.clip(x, vvmin, vvmax)
        x -= vvmin
        x /= vvmax - vvmin
        if channel is not None and channel >= 0:
            x = x[..., channel]
            out = (255 * cmap(x)[..., :3]).astype(np.uint8)
        elif x.ndim == 2:
            out = (255 * cmap(x)[..., :3]).astype(np.uint8)
        else:
            raise ValueError(
                "3D data requires either --channel to select a channel or --rgb for direct RGB output."
            )
        return Frame(out, (vvmin, vvmax), cmap)


def simshow(
    data,
    outfile_base,
    *,
    channel: int = 0,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = "viridis",
    rgb: bool = False,
    colorbar: bool = False,
    frame_number: bool = False,
):
    if len(data.shape) == 2:
        data = data[np.newaxis]
    cmap = colormaps[colorscheme]
    dir = Path(outfile_base).parent
    dir.mkdir(exist_ok=True, parents=True)
    for i, x in enumerate(data):
        f = frame_rgb(
            x,
            channel=channel,
            scale_factor=scale_factor,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            rgb=rgb,
        )
        f.save(
            str(outfile_base) + f"_{i:03d}.png",
            colorbar=colorbar,
            frame_number=i if frame_number else None,
        )


def grid_shape(B: int) -> tuple[int, int]:
    ncols = int(np.ceil(np.sqrt(B)))
    nrows = int(np.ceil(B / ncols))
    return nrows, ncols


def _write_video(frames, outfile, *, n_frames, cmap, channel, scale_factor, vmin, vmax, rgb, grid, nrows, ncols, fps, colorbar, frame_number):
    writer = imageio.get_writer(outfile, fps=fps)
    for i, x in enumerate(tqdm(frames, total=n_frames, desc=f"Writing {Path(outfile).name}")):
        f = frame_rgb(
            x,
            channel=channel,
            scale_factor=scale_factor,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            rgb=rgb,
            grid=grid,
            nrows=nrows,
            ncols=ncols,
        )
        f.append_to(writer, colorbar=colorbar, frame_number=i if frame_number else None)
    writer.close()


def svideo(
    data,
    outfile,
    *,
    rgb: bool = False,
    grid: bool = False,
    batch: bool = False,
    ncols: Optional[int] = None,
    channel: Optional[int] = None,
    scale_factor: float = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorscheme: str = "viridis",
    fps: int = 30,
    colorbar: bool = False,
    frame_number: bool = False,
):
    if grid and batch:
        raise ValueError("Use either --grid or --batch, not both.")
    cmap = colormaps[colorscheme]
    common = dict(
        cmap=cmap,
        channel=channel,
        scale_factor=scale_factor,
        vmin=vmin,
        vmax=vmax,
        rgb=rgb,
        fps=fps,
        colorbar=colorbar,
        frame_number=frame_number,
    )

    if batch:
        n_batch = data.shape[0]
        n_time = data.shape[1]
        outfile = Path(outfile)
        stem = outfile.with_suffix("").name
        suffix = outfile.suffix or ".mp4"
        for b in range(n_batch):
            out = outfile.parent / f"{stem}_{b:03d}{suffix}"
            _write_video(
                iter(data[b]),
                str(out),
                n_frames=n_time,
                grid=False,
                nrows=1,
                ncols=1,
                **common,
            )
    elif grid:
        n_batch = data.shape[0]
        n_time = data.shape[1]
        if ncols is None:
            nrows, ncols = grid_shape(n_batch)
        else:
            nrows = int(np.ceil(n_batch / ncols))
        _write_video(
            (data[:, t] for t in range(n_time)),
            outfile,
            n_frames=n_time,
            grid=True,
            nrows=nrows,
            ncols=ncols,
            **common,
        )
    else:
        _write_video(
            iter(data),
            outfile,
            n_frames=data.shape[0],
            grid=False,
            nrows=1,
            ncols=1,
            **common,
        )
