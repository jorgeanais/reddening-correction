from src.data_loader import DataLoader
from src.param_loader import DifRedParamLoader
from src.redcor import apply_reddening_correction
from src.difred import apply_differential_reddening_correction

def main():
    dl = DataLoader()
    catalogs = dl.run()
    clusters = {k: v for c in catalogs for k, v in c.get_all_clusters().items()}
    apply_reddening_correction(clusters)

    # Differential reddening
    pl = DifRedParamLoader()
    cluster_selection = pl.run()

    return catalogs, clusters, cluster_selection



if __name__ == "__main__":
    catalogs, clusters, cluster_selection = main()