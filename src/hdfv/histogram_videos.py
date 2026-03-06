import imageio.v2 as imageio
import matplotlib
import numpy as np
from matplotlib import colormaps
from tqdm import tqdm


def _to_pixel(x, y, xlim, ylim, resolution):
    xmin, xmax = xlim
    ymin, ymax = ylim
    px = ((x - xmin) / (xmax - xmin) * (resolution - 1)).astype(int)
    py = ((y - ymin) / (ymax - ymin) * (resolution - 1)).astype(int)
    return px, py


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

    for t in data:
        hist = np.histogram2d(
            t[:, 1],
            t[:, 0],
            bins=n_bins,
            range=[xlim, ylim],
        )[0]

        img = np.clip(hist / vmax, 0, 1)
        yield (255 * cmap(img)[..., :3]).astype(np.uint8)


def histogram_video(
    data,
    outfile,
    *,
    n_bins: int = 256,
    xlim=(-1, 1),
    ylim=(-1, 1),
    vmax: float = 4.0,
    fps: int = 30,
    colorscheme: str = "viridis",
):

    vid_writer = imageio.get_writer(outfile, fps=fps, codec="libx264")
    frames = histogram_frames(
        data,
        n_bins=n_bins,
        xlim=xlim,
        ylim=ylim,
        vmax=vmax,
        colorscheme=colorscheme,
    )
    for f in tqdm(frames, total=data.shape[0], desc="Writing frames"):
        vid_writer.append_data(f)

    vid_writer.close()


def mhistims(
    data,
    outfile_base,
    *,
    n_bins: int = 256,
    xlim=(-1, 1),
    ylim=(-1, 1),
    vmax: float = 3.0,
    colorscheme: str = "viridis",
):
    cmap = colormaps[colorscheme]
    for i, t in enumerate(data):
        traj = np.histogram2d(t[:, 1], t[:, 0], n_bins, [list(xlim), list(ylim)])[0]
        img = np.clip(traj / vmax, 0, 1)
        rgb = (255 * cmap(img)[..., :3]).astype(np.uint8)
        imageio.imwrite(str(outfile_base) + f"_{i:03d}.png", rgb)


def _dot_offsets(dot_radius: int):
    return [
        (dy, dx)
        for dy in range(-dot_radius, dot_radius + 1)
        for dx in range(-dot_radius, dot_radius + 1)
        if dy * dy + dx * dx <= dot_radius * dot_radius
    ]


def trace_frames(
    data,
    *,
    resolution: int = 512,
    xlim=(-1, 1),
    ylim=(-1, 1),
    trail_decay: float = 0.92,
    dot_intensity: float = 1.0,
    dot_radius: int = 0,
):
    """
    data: (n_time, n_particles, 2)
    Yields one RGB frame per timestep with a fading trail effect.
    """
    frame = np.zeros((resolution, resolution, 3), dtype=np.float32)
    offsets = np.array(_dot_offsets(dot_radius), dtype=int)  # (n_offsets, 2): (dy, dx)
    color = np.array([0.1, 0.8, 1.0], dtype=np.float32) * dot_intensity

    for t in range(data.shape[0]):
        frame *= trail_decay

        xy = np.array(data[t])
        px, py = _to_pixel(xy[:, 0], xy[:, 1], xlim, ylim, resolution)

        qy = py[np.newaxis, :] + offsets[:, 0, np.newaxis]  # (n_offsets, n_particles)
        qx = px[np.newaxis, :] + offsets[:, 1, np.newaxis]
        qy, qx = qy.ravel(), qx.ravel()
        mask = (qx >= 0) & (qx < resolution) & (qy >= 0) & (qy < resolution)
        np.add.at(frame, (qy[mask], qx[mask]), color)

        yield (255 * np.clip(frame, 0, 1)).astype(np.uint8)


def trace_video(
    data,
    outfile,
    *,
    resolution: int = 512,
    xlim=(-1, 1),
    ylim=(-1, 1),
    trail_decay: float = 0.92,
    dot_intensity: float = 1.0,
    dot_radius: int = 0,
    fps: int = 30,
):
    writer = imageio.get_writer(outfile, fps=fps, codec="libx264")
    frames = trace_frames(
        data,
        resolution=resolution,
        xlim=xlim,
        ylim=ylim,
        trail_decay=trail_decay,
        dot_intensity=dot_intensity,
        dot_radius=dot_radius,
    )
    for f in tqdm(frames, total=data.shape[0], desc="Writing frames"):
        writer.append_data(f)
    writer.close()


def angle_color_coded_frames(
    data,
    source_data,
    *,
    resolution: int = 512,
    xlim=(-1, 1),
    ylim=(-1, 1),
    dot_radius: int = 0,
):
    """
    data:        (n_time, n_particles, 2)  — positions over time
    source_data: (n_particles, 2)          — reference vectors for color (e.g. initial velocity)
    Yields one RGB frame per timestep, particles colored by angle of source_data.
    """
    cmap = matplotlib.colormaps["hsv"]
    angles = np.arctan2(np.array(source_data[:, 1]), np.array(source_data[:, 0]))
    colors = cmap((np.pi + angles) / (2 * np.pi))[:, :3].astype(
        np.float32
    )  # (n_particles, 3)
    offsets = np.array(_dot_offsets(dot_radius), dtype=int)  # (n_offsets, 2): (dy, dx)
    n_offsets = len(offsets)

    for t in range(data.shape[0]):
        frame = np.zeros((resolution, resolution, 3), dtype=np.float32)

        xy = np.array(data[t])
        px, py = _to_pixel(xy[:, 0], xy[:, 1], xlim, ylim, resolution)

        qy = py[np.newaxis, :] + offsets[:, 0, np.newaxis]  # (n_offsets, n_particles)
        qx = px[np.newaxis, :] + offsets[:, 1, np.newaxis]
        qy, qx = qy.ravel(), qx.ravel()
        colors_flat = np.tile(colors, (n_offsets, 1))  # (n_offsets * n_particles, 3)
        mask = (qx >= 0) & (qx < resolution) & (qy >= 0) & (qy < resolution)
        np.add.at(frame, (qy[mask], qx[mask]), colors_flat[mask])

        yield (255 * np.clip(frame, 0, 1)).astype(np.uint8)


def angle_color_coded_video(
    data,
    source_data,
    outfile,
    *,
    resolution: int = 512,
    xlim=(-1, 1),
    ylim=(-1, 1),
    dot_radius: int = 0,
    fps: int = 30,
):
    writer = imageio.get_writer(outfile, fps=fps, codec="libx264")
    frames = angle_color_coded_frames(
        data,
        source_data,
        resolution=resolution,
        xlim=xlim,
        ylim=ylim,
        dot_radius=dot_radius,
    )
    for f in tqdm(frames, total=data.shape[0], desc="Writing frames"):
        writer.append_data(f)
    writer.close()
