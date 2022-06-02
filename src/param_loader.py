import pydantic
from pydantic.dataclasses import dataclass
import yaml

from src.settings import Config


@dataclass
class DifRedClusterParams:
    """Class for store and validate differential reddening params."""

    cluster_name: str
    ms_region: tuple[float, float, float, float]
    origin: tuple[float, float]
    ref_stars_range: tuple[float, float]

    @pydantic.validator("ms_region")
    def check_ms_region(cls, value):
        if value[0] > value[1]:
            raise ValueError("MS region must be in increasing order.")
        if value[2] > value[3]:
            raise ValueError("MS region must be in increasing order.")
        return value

    @pydantic.validator("ref_stars_range")
    def check_ref_stars_range(cls, value):
        if value[0] > value[1]:
            raise ValueError("Ref stars range must be in increasing order.")
        return value


class DifRedParamLoader:
    """Loads cluster params for differential reddening stage."""

    def __init__(self) -> None:
        self.cluster_params: list[DifRedClusterParams] = []

    def read_input_param_file(self) -> None:
        with open(Config.CL_SELECTION) as file:
            file_content = yaml.load(file, Loader=yaml.FullLoader)
            self.cluster_params = [DifRedClusterParams(**c) for c in file_content]

    def run(self) -> list[DifRedClusterParams]:
        self.read_input_param_file()
        return self.cluster_params
