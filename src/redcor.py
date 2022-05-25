import numpy as np
from tqdm import tqdm

from src.models import StarCluster

# Relative extinction coefficients for Gaia's phot (Wang et al. 2019)
RE_AG_AV = 0.789
RE_AGBP_AV = 1.002
RE_AGRP_AV = 0.589


def reddening_correction(cl: StarCluster) -> None:
    """Distance and color correction for a cluster."""

    A_V = cl.paramtable["AVNN"]
    dist = cl.paramtable["DistPc"]

    dm = 5 * np.log10(dist) - 5
    color_excess = (RE_AGBP_AV - RE_AGRP_AV) * A_V

    cl.datatable["Gmag_corr"] = cl.datatable["Gmag"] - dm - RE_AG_AV * A_V
    cl.datatable["BP-RP_corr"] = cl.datatable["BP-RP"] - color_excess


def apply_reddening_correction(clusters: dict[str, StarCluster]) -> None:
    """Apply distance and color correction to all clusters."""

    print("Appling reddening correction...")
    for cl in tqdm(clusters.values()):
        reddening_correction(cl)
