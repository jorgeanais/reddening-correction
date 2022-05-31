"""
https://numpy.org/doc/stable/reference/typing.html#module-numpy.typing
"""
from astropy.table import Table
import numpy as np
import numpy.typing as npt
import pickle
from scipy import stats
from scipy.interpolate import CubicSpline

from src.settings import Config


def _cols2array(table: Table, columns: list[str]) -> np.ndarray:
    """Convert a list of columns from an astropy table to an array"""

    return np.array([table[col].data for col in columns])


def apply_differential_reddening_correction(
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
    
    BIN_STEP = 0.4  # mag
    
    # Remmember (abscissa, ordinate)
    # Check bad values
    # bad_values = np.any(np.isnan(data), axis=0)

    # Bin data along the ordinate
    min_ordinate, max_ordinate = np.nanmin(data[1]), np.nanmax(data[1])
    print(f"Min ordinate: {min_ordinate:.2f}")
    print(f"Max ordinate: {max_ordinate:.2f}")
    
    bins = np.arange(min_ordinate, max_ordinate+BIN_STEP, BIN_STEP)

    # Get the median of each bin along the abscissa
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
    bin_means, bin_edges, binnumber = stats.binned_statistic(data[1], data[0], statistic=np.nanmedian, bins=bins)

    # Fit a cubic spline to the median values
    # cs = CubicSpline(x, y)  <-- return this
    # xs = np.arange(-0.5, 9.6, 0.1)
    # cs(xs)

    # return CubicSpline(x, y)

    pass

def get_delta_abscissa(data: np.ndarray, fiducial_line: CubicSpline) -> np.ndarray:
    """Get the delta abscissa from the fiducial line"""

    # Get the delta abscissa
    delta_abscissa = fiducial_line(data[1]) - data[0]
    return delta_abscissa
    


if __name__ == "__main__":

    reddening_vector = (0.31, 0.59)
    origin = (0.36, 11.93)
    msr = (0.2, 1.9, 11.5, 18.0)
    test_path = Config.TEST_DATA / "NGC_2099_.pkl"
    with open(str(test_path), "rb") as file:
        cmd_data = pickle.load(file)
    
    rotated_data = linear_transformation(cmd_data, reddening_vector, origin)
