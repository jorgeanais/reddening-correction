"""
https://numpy.org/doc/stable/reference/typing.html#module-numpy.typing
"""
from astropy.table import Table
import numpy as np
import numpy.typing as npt
import pickle

from src.settings import Config


def _cols2array(table: Table, columns: list[str]) -> np.ndarray:
    """Convert a list of columns from an astropy table to an array"""

    return np.array([table[col].data for col in columns])


def differential_reddening(
    cmd_data: np.ndarray,
    reddening_vector: tuple[float, float],
    origin: tuple[float, float] | None,
) -> np.ndarray:
    """Differential Reddening correction"""

    # Get reference frame

    # Generation of fiducial line

    # Δ abscissa from fiducial line

    # Selection of reference stars

    # Estimation of differential extinction

    pass


def get_points_within_rectangle_region(
    data: np.ndarray,
    limits: tuple[float, float, float, float],
) -> np.ndarray:
    """Return points that are inside a predefined rectangular region"""

    x1, x2, y1, y2 = limits
    above_inf = np.all(data >= np.array([[x1],[y1]]), axis=0)
    below_sup = np.all(data <= np.array([[x2],[y2]]), axis=0)
    mask = above_inf*below_sup
    return data[:, mask]


def linear_transformation(
    data: np.ndarray,
    origin: tuple[float, float],
    vector: tuple[float, float],
) -> np.ndarray:
    """Perform translation and rotation according to an origin point and vector angle"""

    #  translation
    o = np.array([origin[0], origin[1]]).reshape(2, 1)

    # CMD Rotation
    rot_angle = -np.arctan2(vector[1], vector[0])
    print(f"Rotation angle: {np.rad2deg(rot_angle):.1f}°")
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rotation_matrix = np.array([[c, -s], [s, c]])

    return rotation_matrix @ (data - o)

def get_fiducial_line(
    data: np.ndarray,
    step: float = 0.4,
) -> np.ndarray:
    """Get the fiducial line"""

    # Bin data along the ordinate

    # Get the median of each bin along the abscissa

    # Fit a cubic spline to the median values

    pass
    


if __name__ == "__main__":

    reddening_vector = (0.31, 0.59)
    origin = (0.36, 11.93)
    msr = (0.2, 1.9, 11.5, 18.0)
    test_path = Config.TEST_DATA / "NGC_2099_.pkl"
    with open(str(test_path), "rb") as file:
        cmd_data = pickle.load(file)
    
    rotated_data = linear_transformation(cmd_data, reddening_vector, origin)
