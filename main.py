from src.data_loader import DataLoader
from src.param_loader import DifRedParamLoader
from src.redcor import apply_reddening_correction
from src.difred import apply_differential_reddening_correction
from src.settings import Config


def main():
    # Load data from catalogs
    dl = DataLoader()
    catalogs = dl.run()
    clusters = {k: v for c in catalogs for k, v in c.get_all_clusters().items()}

    # Get reddening info
    apply_reddening_correction(clusters)

    # Get differential reddening params for each cluster
    pl = DifRedParamLoader()
    difred_params = pl.run()

    # Apply Differential Reddening correction
    corrected = apply_differential_reddening_correction(difred_params, clusters)

    return catalogs, clusters, difred_params, corrected


if __name__ == "__main__":

    catalogs, clusters, difred_params, corrected = main()
    for cluster_name, table in corrected.items():
        table.write(
            str(Config.PROC_DATA / f"{cluster_name}_dered.vot"),
            overwrite=True,
            format="votable",
        )
