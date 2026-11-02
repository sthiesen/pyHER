"""Readers for the Jura example data shipped with the MATLAB HERs repository."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.io


@dataclass
class Dataset:
    """Calibration/validation split of a point data set. All arrays are 1-D."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    x_cal: np.ndarray
    y_cal: np.ndarray
    z_cal: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    z_val: np.ndarray
    idx_cal: np.ndarray  # 0-based indices into x, y, z (MATLAB files store them 1-based)
    idx_val: np.ndarray
    txt: str
    # only the synthetic HER (HESS 2020) data sets have a separate test set
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    z_test: np.ndarray | None = None
    idx_test: np.ndarray | None = None


def _read_v73(path: Path) -> dict[str, np.ndarray]:
    """Read numeric variables of a MATLAB v7.3 (HDF5) file with MATLAB's axis order."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key, item in f.items():
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "fiu":
                out[key] = np.asarray(item).T  # HDF5 stores MATLAB arrays transposed
    return out


def load_mat(path: str | Path) -> dict[str, np.ndarray]:
    """Load a .mat file of either format (v7.3 via h5py, older via scipy)."""
    path = Path(path)
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return _read_v73(path)


def _chars(path: Path, key: str) -> str:
    with h5py.File(path, "r") as f:
        return "".join(chr(c) for c in np.asarray(f[key]).ravel())


def load_jura_dataset(path: str | Path) -> Dataset:
    """Load ``Jura_dataset_Pb_log10.mat`` (calibration set: 259 points, validation: 100)."""
    path = Path(path)
    m = load_mat(path)
    v = {k: np.ravel(m[k]).astype(float) for k in
         ("x", "y", "z", "x_cal", "y_cal", "z_cal", "x_val", "y_val", "z_val")}
    idx_cal = np.ravel(m["idx_cal"]).astype(int) - 1
    idx_val = np.ravel(m["idx_val"]).astype(int) - 1
    try:
        txt = _chars(path, "txt")
    except (OSError, KeyError):
        txt = str(m.get("txt", ""))
    return Dataset(idx_cal=idx_cal, idx_val=idx_val, txt=txt, **v)


def load_jura_basin(shp_path: str | Path) -> list[np.ndarray]:
    """Return the basin outline as a list of (n, 2) polygon-part arrays."""
    import shapefile  # pyshp

    sf = shapefile.Reader(str(shp_path))
    parts: list[np.ndarray] = []
    for shp in sf.shapes():
        pts = np.asarray(shp.points)
        bounds = list(shp.parts) + [len(pts)]
        parts.extend(pts[a:b] for a, b in zip(bounds[:-1], bounds[1:]))
    return parts


def load_her_synthetic(path: str | Path) -> Dataset:
    """Load one of the synthetic data sets of Thiesen et al. (2020, HESS), e.g.
    ``LR1_input_randomfield_cal600.mat`` (100 x 100 grid, calibration / validation / test split).

    ``x, y, z`` are the full grid flattened like MATLAB's ``x(:)`` (column-major); the ``idx_*`` arrays
    are 0-based indices into them and ``x_cal = x[idx_cal]`` etc. are checked on loading."""
    m = load_mat(path)
    full = {k: np.asarray(m[k], dtype=float).ravel(order="F") for k in ("x", "y", "z")}
    sets = {}
    for part in ("cal", "val", "test"):
        idx = np.ravel(m[f"idx_{part}"]).astype(int) - 1
        sets[part] = idx
        for k in ("x", "y", "z"):
            got = np.ravel(m[f"{k}_{part}"]).astype(float)
            if not np.allclose(full[k][idx], got):
                raise ValueError(f"{k}_{part} does not match {k}(idx_{part}) in {path}")
    ds = Dataset(
        **full, x_cal=full["x"][sets["cal"]], y_cal=full["y"][sets["cal"]], z_cal=full["z"][sets["cal"]],
        x_val=full["x"][sets["val"]], y_val=full["y"][sets["val"]], z_val=full["z"][sets["val"]],
        idx_cal=sets["cal"], idx_val=sets["val"], txt=str(m.get("txt", "")),
        x_test=full["x"][sets["test"]], y_test=full["y"][sets["test"]], z_test=full["z"][sets["test"]],
        idx_test=sets["test"])
    return ds
