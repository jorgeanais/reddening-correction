"""
This script creates the final CMDs plots corrected for distance.
"""

from src.data_loader import DataLoader
from src.param_loader import DifRedParamLoader
from src.redcor import apply_reddening_correction
from src.difred import apply_differential_reddening_correction
from src.plots import plot_dif_dist_corrected


# Load clusters from catalogs
dl = DataLoader()
catalogs = dl.run()
clusters = {k: v for c in catalogs for k, v in c.get_all_clusters().items()}

# First time to calculate reddening vector
apply_reddening_correction(clusters)

# Apply differential reddening correction
pl = DifRedParamLoader()
difred_params = pl.run()
corrected = apply_differential_reddening_correction(difred_params, clusters)

# Create a dict with processed clusters
processed_clusters = {}
for cluster_name, table in corrected.items():
    print("cluster_name")
    clusters[cluster_name].membertable = table  # Update info
    processed_clusters[cluster_name] = clusters[cluster_name]

# Apply distance correction on top of differential reddening correction
apply_reddening_correction(processed_clusters, "Gmag_dered", "BP-RP_dered")

for cluster_name, cluster in processed_clusters.items():
    plot_dif_dist_corrected(cluster.membertable, cluster_name)
    print(f"{cluster_name} done")


# Load files
# files = glob.glob(str(Config.PROC_DATA / "*.vot"))

# file = files[0]
# table = Table.read(file)

# "Gmag_dered", "BP-RP_dered"
# "mk_dered", "mj-mk_dered"
