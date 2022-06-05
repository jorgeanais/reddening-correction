from astropy.table import Table, hstack
import numpy as np
import pickle

from astropy.stats import sigma_clip
from scipy import stats
from scipy.interpolate import CubicSpline
from sklearn.neighbors import NearestNeighbors

from src.models import StarCluster
from src.param_loader import DifRedClusterParams
from src.settings import Config
from src.plots import plot_cmd_reddening_vector, plot_rotated_cmd, plot_dereddened_cmd


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
    epoch: int | None = None,
) -> Table:
    """Append an array to an astropy table. If epoch is given, then column name are renamed: colname_epoch"""

    if epoch is not None:
        columns = [c + f"_{epoch}" for c in columns]
    return hstack([table, _array2cols(array, columns)])


def _sigmaclip_median(array: np.ndarray) -> float:
    """Get the median of a data array, removing outliers"""
    
    x = array.copy()
    x = x[~np.isnan(x)]
    clipped = sigma_clip(x, cenfunc='median', stdfunc="mad_std", masked=False)
    return np.median(clipped)

def apply_differential_reddening_correction(
    drpl: list[DifRedClusterParams],
    clusters: dict[str, StarCluster],
) -> list[Table]:
    """Apply Differential Reddening correction to all clusters defined in a list"""

    print("Appling differential reddening correction...")
    results = []
    for drparams in drpl:
        cl = clusters[drparams.cluster_name]
        r = differential_reddening_correction(cl, drparams)
        results.append(r)

    return results


def differential_reddening_correction(
    star_cluster: StarCluster,
    drparams: DifRedClusterParams,
    epochs: int = 12,
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
    plot_cmd_reddening_vector(star_cluster.membertable, origin, reddening_vector, star_cluster.name)

    # Select only data from stars within the MS CMD region
    cmd_data = _cols2array(membertable, ["BP-RP", "Gmag"])
    ms_data = replace_points_outside_rectangle_region_with_nan(cmd_data, ms_region)
    membertable = _append_array_to_table(membertable, ms_data, ["ms_BP-RP", "ms_Gmag"])

    # Apply linear transformation
    abs_ord_data = linear_transformation(ms_data, origin, reddening_vector)
    membertable = _append_array_to_table(
        membertable,
        abs_ord_data,
        ["abscissa", "ordinate"],
        epoch=0,
    )

    for epoch in range(epochs):

        print(f"Epoch: {epoch}")

        # Fit fiducial line
        fiducial_line, _, _ = fit_fiducial_line(abs_ord_data)

        # Get `Δ abscissa` from fiducial line
        delta_abscissa = get_delta_abscissa(abs_ord_data, fiducial_line)
        membertable = _append_array_to_table(
            membertable, delta_abscissa, ["delta_abscissa"], epoch
        )

        # Selection of reference stars
        ref_stars = get_points_within_range(abs_ord_data[1], ref_stars_range)
        membertable = _append_array_to_table(
            membertable, ref_stars, ["ref_stars"], epoch
        )

        # Estimation of differential extinction
        abscissa_corrected = diffred_estimation(membertable, epoch)

        abs_ord_data = np.array([abscissa_corrected, abs_ord_data[1]])
        membertable = _append_array_to_table(
            membertable, abs_ord_data, ["abscissa", "ordinate"], epoch + 1
        )

        plot_rotated_cmd(
            membertable, fiducial_line, ref_stars_range, star_cluster.name, epoch
        )

    # Back to original coordinates
    dereddened_cmd = linear_transformation(
        abs_ord_data,
        origin,
        reddening_vector,
        inverse=True,
    )
    membertable = _append_array_to_table(
        membertable, dereddened_cmd, ["BP-RP_dered", "Gmag_dered"]
    )

    plot_dereddened_cmd(membertable, star_cluster.name)

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
    """Replace points that are outside a predefined rectangular region with NaNs"""

    data = data.copy()
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
    inverse: bool = False,
) -> np.ndarray:
    """Perform translation and rotation according to an origin point and vector angle"""

    #  translation
    o = np.array([origin[0], origin[1]]).reshape(2, 1)

    # CMD Rotation
    rot_angle = -np.arctan(vector[1] / vector[0])

    if inverse:
        rot_angle = -rot_angle

    print(f"Rotation angle: {np.rad2deg(rot_angle):.1f}°")
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    rotation_matrix = np.array([[c, -s], [s, c]])

    if inverse:
        result = rotation_matrix @ data + o
    else:
        result = rotation_matrix @ (data - o)

    return result


def fit_fiducial_line(
    data: np.ndarray,
    bin_width: float = 0.4,
) -> tuple[CubicSpline, np.ndarray, np.ndarray]:
    """Get the fiducial line of MS stars along the rotated CMD"""


    # Bin data along the ordinate (y axis)
    min_ordinate, max_ordinate = np.nanmin(data[1]), np.nanmax(data[1])
    bins_ordinate = np.arange(min_ordinate, max_ordinate + bin_width, bin_width)

    # Get the median of each bin along the abscissa and ordinate axis
    median_abscissa, _, _ = stats.binned_statistic(
        data[1], data[0], statistic=_sigmaclip_median, bins=bins_ordinate
    )

    median_ordinate, _, _ = stats.binned_statistic(
        data[1], data[1], statistic=_sigmaclip_median, bins=bins_ordinate
    )

    # Notice that we want the abscissa value as function of the ordinate value (x = x(y))
    return CubicSpline(median_ordinate, median_abscissa), median_abscissa, median_ordinate


def get_delta_abscissa(data: np.ndarray, fiducial_line: CubicSpline) -> np.ndarray:
    """Get the distance `Δ abscissa` from the fiducial line along the reddening direction"""

    return data[0] - fiducial_line(data[1])


def diffred_estimation(table: Table, epoch: int, k: float = 35) -> np.ndarray:
    """Get the median color residual along the reddening vector (Δ abscissa) from k nearest neighbors"""

    # from sklearn.metrics import pairwise_distances
    # dist_matrix = pairwise_distances(coords.T, metric='haversine')

    # check that table lenght is greater than k
    if len(table) < k:
        raise ValueError(f"Table length must be greater than k ({k})")

    # Get coordinates
    coords = np.deg2rad(_cols2array(table, ["RA_ICRS", "DE_ICRS"]))

    reference_star_table = table[table[f"ref_stars_{epoch}"]]
    coords_ref_stars = np.deg2rad(
        _cols2array(reference_star_table, ["RA_ICRS", "DE_ICRS"])
    )

    # Find k-nearest reference stars
    nn = NearestNeighbors(n_neighbors=k, metric="haversine").fit(coords_ref_stars.T)
    ang_dist, indxs = nn.kneighbors(coords.T, return_distance=True)
    ang_dist = np.rad2deg(ang_dist)

    # Get median Δ abscissa values from k nearest reference stars
    delta_abscissa = _cols2array(reference_star_table, [f"delta_abscissa_{epoch}"])
    median_delta_abscissa = np.nanmedian(np.take(delta_abscissa, indxs), axis=1)
    table["difredest"] = median_delta_abscissa
    print("Average Δ median abscissa: ", np.mean(median_delta_abscissa))
    abscissa_corrected = table[f"abscissa_{epoch}"] - median_delta_abscissa
    table["abscissa_corrected"] = abscissa_corrected

    return abscissa_corrected.data

