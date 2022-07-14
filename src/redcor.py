import numpy as np
from tqdm import tqdm

from src.models import StarCluster

# Relative extinction coefficients for Gaia's phot (Wang et al. 2019)
RE_AG_AV = 0.789
RE_AGBP_AV = 1.002
RE_AGRP_AV = 0.589


def reddening_correction(
    cl: StarCluster, magnitude_col: str = "Gmag", color_col: str = "BP-RP"
) -> None:
    """Distance and color correction for a cluster."""

    A_V = cl.paramtable["AVNN"]
    dist = cl.paramtable["DistPc"]

    dm = 5 * np.log10(dist) - 5
    color_excess = (RE_AGBP_AV - RE_AGRP_AV) * A_V
    A_G = RE_AG_AV * A_V

    cl.paramtable["E(GBP - GRP)"] = color_excess
    cl.paramtable["A_G"] = A_G

    cl.membertable[f"{magnitude_col}_corr"] = cl.membertable[magnitude_col] - dm - A_G
    cl.membertable[f"{color_col}_corr"] = cl.membertable[color_col] - color_excess


def apply_reddening_correction(
    clusters: dict[str, StarCluster],
    magnitude_col: str = "Gmag",
    color_col: str = "BP-RP",
) -> None:
    """Apply distance and color correction to all clusters."""

    print("Appling reddening correction...")
    for cl in tqdm(clusters.values()):
        reddening_correction(cl, magnitude_col, color_col)
