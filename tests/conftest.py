from pathlib import Path

import pytest
import scipy.io

GOLDEN = Path(__file__).parent / "golden"
REPO = Path(__file__).resolve().parents[2]
JURA_MAT = REPO / "HERs" / "1.0.raw data" / "Jura_dataset_Pb_log10.mat"
JURA_SHP = REPO / "HERs" / "1.0.raw data" / "jura_basin.shp"


@pytest.fixture(scope="session")
def golden():
    def load(name):
        return scipy.io.loadmat(GOLDEN / f"{name}.mat", squeeze_me=True, struct_as_record=False)
    return load
