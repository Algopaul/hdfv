import h5py
import numpy as np
import pytest
import zarr


@pytest.fixture
def h5_particles(tmp_path):
  path = tmp_path / "data.h5"
  with h5py.File(path, "w") as f:
    f["particles"] = np.random.uniform(-1, 1, (10, 100, 2))
    f["field"] = np.random.rand(20, 32, 32)
  return path


@pytest.fixture
def zarr_particles(tmp_path):
  path = tmp_path / "data.zarr"
  store = zarr.open_group(str(path), mode="w")
  store["particles"] = np.random.uniform(-1, 1,
                                         (10, 100, 2))  # type: ignore[index]
  store["field"] = np.random.rand(20, 32, 32)  # type: ignore[index]
  return path
