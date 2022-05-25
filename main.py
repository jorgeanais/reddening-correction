from src.data_loader import DataLoader
from src.redcor import apply_reddening_correction

def main():
    dl = DataLoader()
    catalogs = dl.run()
    clusters = {k: v for c in catalogs for k, v in c.get_all_clusters().items()}
    apply_reddening_correction(clusters)
    return catalogs, clusters


if __name__ == "__main__":
    catalogs, clusters = main()