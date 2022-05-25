from dataclasses import field
from pathlib import Path
import pickle
import pydantic
from pydantic.dataclasses import dataclass
import yaml

from astroquery.utils.commons import TableList
from astroquery.vizier import Vizier

from src.settings import Config
from src.models import Catalog


Vizier.ROW_LIMIT = -1


@dataclass
class StagingCatalog:
    """Staging class for validation of catalogs in yaml file."""

    name: str
    cds_id: str
    author: str
    table_names: dict[str, str]
    tablelist_path: Path = field(init=False, default_factory=Path)

    @pydantic.validator("table_names")
    def check_tables_key(cls, value):
        table_keys = ("clusters_params", "cluster_members")
        if not all(key in value for key in table_keys):
            raise ValueError(f"Catalog must include {', '.join(table_keys)}.")
        return value

    def __post_init_post_parse__(self):
        self.tablelist_path = Config.RAW_DATA / (
            self.name + self.cds_id.replace("/", "_") + ".pkl"
        )
        # Add cds_id to table_names
        self.table_names = {
            k: self.cds_id + "/" + v for k, v in self.table_names.items()
        }


class DataLoader:
    """Loads data from Vizier and writes to disk."""

    def __init__(self) -> None:
        self.staging_list: list[StagingCatalog] = []
        self.catalogs: list[Catalog] = []

    def read_input_catalog_file(self) -> None:
        with open(Config.CATALOGS) as file:
            file_content = yaml.load(file, Loader=yaml.FullLoader)
            self.staging_list = [StagingCatalog(**c) for c in file_content]

    def download_tablelist(self, staging_catalog: StagingCatalog) -> TableList:
        print(f"Downloading {staging_catalog.name}...")
        return Vizier.get_catalogs(staging_catalog.cds_id)

    def write_tablelist(self, table_list: TableList, path: Path) -> None:
        with open(str(path), "wb") as file:
            print(f"Writing to {path}...")
            pickle.dump(table_list, file, protocol=pickle.HIGHEST_PROTOCOL)

    def read_tablelist(self, path: Path) -> TableList:
        with open(str(path), "rb") as file:
            print(f"Reading {path}...")
            return pickle.load(file)

    def load_catalogs(self) -> None:
        for scat in self.staging_list:
            if scat.tablelist_path.is_file():
                tl = self.read_tablelist(scat.tablelist_path)
            else:
                tl = self.download_tablelist(scat)
                self.write_tablelist(tl, scat.tablelist_path)

            p_table = tl[scat.table_names["clusters_params"]]
            m_table = tl[scat.table_names["cluster_members"]]
            cat = Catalog(scat.name, scat.cds_id, scat.author, p_table, m_table)
            self.catalogs.append(cat)

    def run(self) -> list[Catalog]:
        self.read_input_catalog_file()
        self.load_catalogs()
        return self.catalogs
