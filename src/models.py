from dataclasses import dataclass, field
from tqdm import tqdm

from astropy.coordinates import SkyCoord
from astropy import table


@dataclass(frozen=True, slots=True)
class StarCluster:
    """A Star Cluster representation."""    

    name: str
    coordinates: SkyCoord = field(repr=False)
    membertable: table.Table = field(repr=False)
    paramtable: table.Table = field(repr=False)


@dataclass(slots=True)
class Catalog:
    """A catalog representation."""

    name: str
    cds_id: str
    author: str
    params_table: table.Table = field(repr=False)
    members_table: table.Table = field(repr=False)
    star_clusters: dict[str, StarCluster] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self.create_star_clusters()

    def create_star_clusters(self) -> None:
        clusters = table.unique(self.members_table, keys="Cluster")["Cluster"].data.data
        print(f"Creating {self.name} star clusters...")
        for cl in tqdm(clusters):
            members = self.members_table[self.members_table["Cluster"] == cl]
            params = self.params_table[self.params_table["Cluster"] == cl]
            coords = SkyCoord(
                params["RA_ICRS"].data.data,
                params["DE_ICRS"].data.data,
                frame="icrs",
                unit="deg",
            )
            sc = StarCluster(cl, coords, members, params)
            self.star_clusters[cl] = sc

    def get_all_clusters(self) -> dict[StarCluster]:
        return self.star_clusters

    def list_cluster(self) -> None:
        for cl in self.star_clusters.values():
            print(f"{cl.name}")

    def get_cluster(self, cluster_name: str) -> StarCluster:
        return self.star_clusters[cluster_name]
