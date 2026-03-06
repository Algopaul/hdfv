"""Generate small test files with jittering particles for hdfv smoke tests."""

import h5py
import numpy as np
import zarr

RNG = np.random.default_rng(42)


def jitter(start, n_time, noise=0.02):
    """Random walk from starting positions."""
    pos = start.copy()
    traj = [pos.copy()]
    for _ in range(n_time - 1):
        pos = np.clip(pos + RNG.normal(0, noise, pos.shape), -1, 1)
        traj.append(pos.copy())
    return np.stack(traj)  # (n_time, n_particles, 2)


# --- scenario A: single cluster, many particles ---
n_time, n_particles = 200, 500
start_a = RNG.uniform(-0.2, 0.2, (n_particles, 2))
traj_a = jitter(start_a, n_time)

# --- scenario B: two clusters drifting apart ---
n_particles_b = 300
start_b = np.vstack(
    [
        RNG.uniform(-0.5, -0.3, (n_particles_b // 2, 2)),
        RNG.uniform(0.3, 0.5, (n_particles_b // 2, 2)),
    ]
)
# add a slow drift
drift = np.linspace(0, 0.4, n_time)[:, None, None] * np.array([-1, 1])[None, None, :]
traj_b = jitter(start_b, n_time, noise=0.01) + drift

# --- scenario C: ring of particles ---
n_particles_c = 400
theta = np.linspace(0, 2 * np.pi, n_particles_c, endpoint=False)
start_c = 0.6 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
traj_c = jitter(start_c, n_time, noise=0.015)

# write HDF5
with h5py.File("test_particles.h5", "w") as f:
    f.create_dataset("single_cluster", data=traj_a)  # (200, 500, 2)
    f.create_dataset("two_clusters", data=traj_b)  # (200, 300, 2)
    f.create_dataset("ring", data=traj_c)  # (200, 400, 2)
    # initial positions as reference vectors for anglevid
    f.create_dataset("ring_init", data=start_c)  # (400, 2)
    f.create_dataset("cluster_init", data=start_a)  # (500, 2)

print("wrote test_particles.h5")

# write Zarr
store = zarr.open("data/test_particles.zarr", mode="w")
store["single_cluster"] = traj_a
store["two_clusters"] = traj_b
store["ring"] = traj_c
store["ring_init"] = start_c
store["cluster_init"] = start_a

print("wrote test_particles.zarr")
print()
print("Example commands:")
print("  hdfv tracevid test_particles.h5 ring out_ring.mp4")
print("  hdfv tracevid test_particles.h5 two_clusters out_drift.mp4 --trail-decay 0.85")
print("  hdfv anglevid test_particles.h5 ring ring_init out_ring_angle.mp4")
print("  hdfv histvid  test_particles.h5 single_cluster out_hist.mp4")
