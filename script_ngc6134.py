from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np

from src.difred import differential_reddening_correction
from src.models import StarCluster
from src.param_loader import DifRedClusterParams
from src.settings import Config
from src.plots import plot_dif_dist_corrected


CLUSTER_NAME = "NGC_6134_NIR"

# Read data
members = Table.read(Config.RAW_DATA / "NGC_6134.csv", format="csv")
members["mj-mk"] = members["mj"] - members["mk"]
members["RA_ICRS"] = members["ra"]  # Rename columns
members["DE_ICRS"] = members["dec"]

# Information from Peña Ramirez et al. (2022)
dict_of_columns = {
    "name": [CLUSTER_NAME],
    "RA_ICRS": [246.941],
    "DE_ICRS": [-49.16],
    "d": [1141],
    "e_d": [36],
    "Av": [1.49],
    "e_Av": [0.05],
    "log_age": [8.93],
    "e_log_age": [0.05],
    "E(J-Ks)": [0.23],
    "A_Ks": [0.18],
    "mass": [681],
    "e_mass": [19],
    "N": [594],
}

params = Table(dict_of_columns)

# Create star cluster object
coords = SkyCoord(
    params["RA_ICRS"].data.data,
    params["DE_ICRS"].data.data,
    frame="icrs",
    unit="deg",
)
cl_ngc6134 = StarCluster(CLUSTER_NAME, coords, members, params)

# from src.param_loader import DifRedClusterParams
drcp = DifRedClusterParams(
    cluster_name=CLUSTER_NAME,
    ms_region=(0.27, 0.9, 11.27, 15.0),
    origin=(0.27, 11.27),
    ref_stars_range=(0.2, 1.8),
    bin_width=0.2,
)

# Reddening correction
t = differential_reddening_correction(
    star_cluster=cl_ngc6134,
    drparams=drcp,
    epochs=2,
    color_excess="E(J-Ks)",
    extinction="A_Ks",
    color="mj-mk",
    magnitude="mk",
)

# Distance correction
dm = 5 * np.log10(params["d"]) - 5
t["mk_dered_corr"] = t["mk_dered"] - dm - params["A_Ks"]
t["mj-mk_dered_corr"] = t["mj-mk_dered"] - params["E(J-Ks)"]

plot_dif_dist_corrected(t, CLUSTER_NAME, color_col="mj-mk_dered_corr", magnitude_col="mk_dered_corr")

t.write(
    str(Config.PROC_DATA / f"{CLUSTER_NAME}_dered_dist.vot"),
    overwrite=True,
    format="votable",
)


