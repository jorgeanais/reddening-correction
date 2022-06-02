"""
https://numpy.org/doc/stable/reference/typing.html#module-numpy.typing
"""
from astropy.table import Table, hstack
import numpy as np
import pickle

from scipy import stats
from scipy.interpolate import CubicSpline
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.models import StarCluster
from src.param_loader import DifRedClusterParams
from src.settings import Config


def _cols2array(table: Table, columns: list[str]) -> np.ndarray:
    """Convert a list of columns from an astropy table to an array"""

    return np.array([table[col].data for col in columns])


def _array2cols(array: np.ndarray, columns: list[str]) -> Table:
    """Convert an array to an astropy table"""
    if len(array.shape) == 1:
        array = array.reshape(1, -1)
    return Table([array[i] for i in range(array.shape[0])], names=columns)


def _append_array_to_table(
    table: Table,
    array: np.ndarray,
    columns: list[str],
) -> Table:
    """Append an array to an astropy table"""

    return hstack([table, _array2cols(array, columns)])


def apply_differential_reddening_correction(
    drpl: list[DifRedClusterParams],
    clusters: dict[str, StarCluster],
) -> list[Table]:
    """Apply Differential Reddening correction to all clusters defined in a list."""

    print("Appling differential reddening correction...")
    results = []
    for drparams in tqdm(drpl):
        cl = clusters[drparams.cluster_name]
        r = differential_reddening_correction(cl, drparams)
        results.append(r)

    return results


def differential_reddening_correction(
    star_cluster: StarCluster,
    drparams: DifRedClusterParams,
    epochs: int = 4,
) -> Table:
    """Differential Reddening correction workflow"""

    # Get params
    ms_region = drparams.ms_region
    origin = drparams.origin
    ref_stars_range = drparams.ref_stars_range
    reddening_vector = tuple(
        np.squeeze(_cols2array(star_cluster.paramtable, ["E(GBP - GRP)", "A_G"]))
    )

    # Get cluster's data (astropy table)
    membertable = star_cluster.membertable.copy()

    # Select only data from stars within the MS CMD region
    cmd_data = _cols2array(membertable, ["BP-RP", "Gmag"])
    ms_data = replace_points_outside_rectangle_region_with_nan(cmd_data, ms_region)
    membertable = _append_array_to_table(membertable, ms_data, ["ms_BP-RP", "ms_Gmag"])

    # Apply linear transformation
    rotated_data = linear_transformation(ms_data, origin, reddening_vector)
    membertable = _append_array_to_table(
        membertable, rotated_data, ["abscissa", "ordinate"]
    )

    # Generation of fiducial line
    fiducial_line = get_fiducial_line(rotated_data)

    # Δ abscissa from fiducial line
    delta_abscissa = get_delta_abscissa(rotated_data, fiducial_line)
    membertable = _append_array_to_table(
        membertable, delta_abscissa, ["delta_abscissa"]
    )

    # Selection of reference stars
    ref_stars = get_points_within_range(rotated_data, ref_stars_range)
    membertable = _append_array_to_table(membertable, ref_stars, ["ref_stars"])

    # Estimation of differential extinction
    diffred_estimation(membertable)

    return membertable


def get_points_within_range(
    data: np.ndarray,
    limits: tuple[float, float],
) -> np.ndarray:
    """Return bolean array indicating points that are inside a range"""

    x1, x2 = limits
    mask = np.all(data >= np.array([[x1]]), axis=0)
    mask *= np.all(data <= np.array([[x2]]), axis=0)
    return mask


def replace_points_outside_rectangle_region_with_nan(
    data: np.ndarray,
    limits: tuple[float, float, float, float],
) -> np.ndarray:
    """Replace points that are outside a predefined rectangular region with NaN"""

    x1, x2, y1, y2 = limits
    above_inf = np.all(data >= np.array([[x1], [y1]]), axis=0)
    below_sup = np.all(data <= np.array([[x2], [y2]]), axis=0)
    mask = above_inf * below_sup
    data[:, ~mask] = np.nan
    return data


def linear_transformation(
    data: np.ndarray,
    origin: tuple[float, float],
    vector: tuple[float, float],
) -> np.ndarray:
    """Perform translation and rotation according to an origin point and vector angle"""

    #  translation
    o = np.array([origin[0], origin[1]]).reshape(2, 1)

    # CMD Rotation
    rot_angle = -np.arctan(vector[1] / vector[0])
    print(f"Rotation angle: {np.rad2deg(rot_angle):.1f}°")
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rotation_matrix = np.array([[c, -s], [s, c]])

    return rotation_matrix @ (data - o)


def get_fiducial_line(
    data: np.ndarray,
    step: float = 0.4,
) -> np.ndarray:
    """Get the fiducial line of MS stars along the rotated CMD"""

    BIN_STEP = 0.4  # mag

    # Remmember (abscissa, ordinate)
    # Check bad values
    # bad_values = np.any(np.isnan(data), axis=0)

    # Bin data along the ordinate
    min_ordinate, max_ordinate = np.nanmin(data[1]), np.nanmax(data[1])
    print(f"Min ordinate: {min_ordinate:.2f}")
    print(f"Max ordinate: {max_ordinate:.2f}")

    bins_ordinate = np.arange(min_ordinate, max_ordinate + BIN_STEP, BIN_STEP)

    # Get the median of each bin along the abscissa and ordinate axis
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
    median_abscissa, _edges, _number = stats.binned_statistic(
        data[1], data[0], statistic=np.nanmedian, bins=bins_ordinate
    )
    median_ordinate, _edges, _number = stats.binned_statistic(
        data[1], data[1], statistic=np.nanmedian, bins=bins_ordinate
    )

    # Fit a cubic spline to the median values
    # cs = CubicSpline(x, y)  <-- return this
    # xs = np.arange(-0.5, 9.6, 0.1)
    # cs(xs)

    # Notice that we want the abscissa value as function of the ordinate value (x = x(y))
    return CubicSpline(median_ordinate, median_abscissa)


def get_delta_abscissa(data: np.ndarray, fiducial_line: CubicSpline) -> np.ndarray:
    """Get the delta abscissa from the fiducial line"""

    # Get the delta abscissa
    delta_abscissa = fiducial_line(data[1]) - data[0]
    return delta_abscissa


def diffred_estimation(table: Table, k: float = 35) -> np.ndarray:
    """Get the median color residual along the reddening vector (Δ abscissa) from k nearest neighbors"""

    # from sklearn.metrics import pairwise_distances
    # dist_matrix = pairwise_distances(coords.T, metric='haversine')

    # check that table lenght is greater than k
    if len(table) < k:
        raise ValueError(f"Table length must be greater than k ({k})")

    # Get coordinates
    coords = np.deg2rad(_cols2array(table, ["RA_ICRS", "DE_ICRS"]))

    reference_star_table = table[table["ref_stars"]]
    coords_ref_stars = np.deg2rad(
        _cols2array(reference_star_table, ["RA_ICRS", "DE_ICRS"])
    )

    # Find k-nearest reference stars
    nn = NearestNeighbors(n_neighbors=k, metric="haversine").fit(coords_ref_stars.T)
    indxs = nn.kneighbors(coords.T, return_distance=False)

    # Get median Δ abscissa values from k nearest reference stars
    delta_abscissa = _cols2array(reference_star_table, ["delta_abscissa"])
    median_delta_abscissa = np.nanmedian(np.take(delta_abscissa, indxs), axis=1)
    table["difredest"] = median_delta_abscissa
    table["abscissa_corrected"] = table["abscissa"] - median_delta_abscissa


if __name__ == "__main__":

    reddening_vector = (0.31, 0.59)
    origin = (0.36, 11.93)
    msr = (0.2, 1.9, 11.5, 18.0)
    test_path = Config.TEST_DATA / "NGC_2099_.pkl"
    with open(str(test_path), "rb") as file:
        cmd_data = pickle.load(file)

    rotated_data = linear_transformation(cmd_data, origin, reddening_vector)
    fiducial_line = get_fiducial_line(rotated_data)
    delta_abscissa = get_delta_abscissa(rotated_data, fiducial_line)
