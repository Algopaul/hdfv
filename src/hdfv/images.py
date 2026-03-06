from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps
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
):
    if cmap is None:
        cmap = colormaps["viridis"]
    if channel is not None and rgb:
        raise ValueError("Use either --channel or --rgb, not both.")

    if grid:
        x = tile_batch(x, nrows, ncols)

    if rgb:
        return x, None, None
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
        return out, vvmin, vvmax


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
    cmap = colormaps[colorscheme]
    dir = Path(outfile_base).parent
    dir.mkdir(exist_ok=True, parents=True)
    for i, x in enumerate(data):
        frame, vmin_used, vmax_used = frame_rgb(
            x,
            channel=channel,
            scale_factor=scale_factor,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            rgb=rgb,
            grid=False,
            nrows=1,
            ncols=1,
        )
        frame = annotate_frame(
            frame,
            frame_idx=i,
            cmap=cmap,
            vmin=vmin_used,
            vmax=vmax_used,
            colorbar=colorbar,
            frame_number=frame_number,
        )
        imageio.imwrite(str(outfile_base) + f"_{i:03d}.png", frame)


def grid_shape(B: int) -> tuple[int, int]:
    ncols = int(np.ceil(np.sqrt(B)))
    nrows = int(np.ceil(B / ncols))
    return nrows, ncols


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


def annotate_frame(
    rgb: np.ndarray,
    *,
    frame_idx: Optional[int] = None,
    cmap=None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = False,
    frame_number: bool = False,
    bar_width: int = 16,
    label_width: int = 52,
) -> np.ndarray:
    """Append colorbar and/or overlay frame counter onto an RGB frame."""
    frame = rgb
    if colorbar and cmap is not None and vmin is not None and vmax is not None:
        h = frame.shape[0]
        cb = _colorbar_strip(
            cmap, vmin, vmax, h, bar_width=bar_width, label_width=label_width
        )
        sep = np.zeros((h, 2, 3), dtype=np.uint8)
        frame = np.concatenate([frame, sep, cb], axis=1)

    if frame_number and frame_idx is not None:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        font = _font()
        text = f"t={frame_idx:04d}"
        draw.text((5, 5), text, fill=(0, 0, 0), font=font)
        draw.text((4, 4), text, fill=(255, 255, 255), font=font)
        frame = np.array(img)

    return frame


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
    colorscheme: str = "viridis",
    fps: int = 30,
    colorbar: bool = False,
    frame_number: bool = False,
):
    if grid:
        n_batch = data.shape[0]
        n_time = data.shape[1]
        if ncols is None:
            nrows, ncols = grid_shape(n_batch)
        else:
            nrows = int(np.ceil(n_batch / ncols))
        frames = (data[:, t] for t in range(n_time))
    else:
        nrows, ncols = 1, 1
        frames = iter(data)
    cmap = colormaps[colorscheme]
    writer = imageio.get_writer(outfile, fps=fps)
    for i, x in enumerate(
        tqdm(
            frames,
            total=data.shape[1] if grid else data.shape[0],
            desc="Writing frames",
        )
    ):
        frame, vmin_used, vmax_used = frame_rgb(
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
        frame = annotate_frame(
            frame,
            frame_idx=i,
            cmap=cmap,
            vmin=vmin_used,
            vmax=vmax_used,
            colorbar=colorbar,
            frame_number=frame_number,
        )
        writer.append_data(frame)
    writer.close()
