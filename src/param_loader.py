import pydantic
from pydantic.dataclasses import dataclass
import yaml

from src.settings import Config

@dataclass
class DifRedClusterParams:
    """Class for validation of differntial reddening params."""

    cluster_name: str  # Name of cluster
    ms_region: list[float, float, float, float]  # color_min, color_max, mag_min, mag_max
    origin: list[float, float]  # (abscissa, ordinate)
    ref_stars_range: list[float, float]  # ordinate_min, ordinate_max

    @pydantic.validator("ms_region")
    def check_ms_region(cls, value):
        if not len(value) == 4:
            raise ValueError("MS region must be a list of 4 floats.")
        if value[0] > value[1]:
            raise ValueError("MS region must be in increasing order.")
        if value[2] > value[3]:
            raise ValueError("MS region must be in increasing order.")
        return value
    
    @pydantic.validator("ref_stars_range")
    def check_ref_stars_range(cls, value):
        if not len(value) == 2:
            raise ValueError("Ref stars range must be a list of 2 floats.")
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