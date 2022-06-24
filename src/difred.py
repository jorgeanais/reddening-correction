from astropy.table import Table, hstack
import numpy as np
import pickle

from astropy.stats import sigma_clip
from scipy import stats
from scipy.interpolate import CubicSpline
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

from src.models import StarCluster
from src.param_loader import DifRedClusterParams
from src.settings import Config
from src.plots import (
    plot_cmd_reddening_vector,
    plot_rotated_cmd,
    plot_dereddened_cmd,
    plot_difred_test,
    plot_dereddened_cmd_for_report
)


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
    clipped = sigma_clip(
        x, sigma_lower=3, sigma_upper=2, cenfunc="median", masked=False
    )
    return np.median(clipped)


def apply_differential_reddening_correction(
    drpl: list[DifRedClusterParams],
    clusters: dict[str, StarCluster],
) -> dict[str, Table]:
    """Apply Differential Reddening correction to all clusters defined in a list"""

    print("Appling differential reddening correction...")
    results = dict()
    for drparams in drpl:
        cl = clusters[drparams.cluster_name]
        r = differential_reddening_correction(cl, drparams)
        results[drparams.cluster_name] = r

    return results


def differential_reddening_correction(
    star_cluster: StarCluster,
    drparams: DifRedClusterParams,
    epochs: int = 2,
) -> Table:
    """Differential Reddening correction workflow"""

    # Get params
    print(star_cluster.name)
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
    ms_mask, ms_data = replace_points_outside_rectangle_region_with_nan(
        cmd_data, ms_region
    )
    membertable = _append_array_to_table(membertable, ms_data, ["ms_BP-RP", "ms_Gmag"])
    membertable = _append_array_to_table(membertable, ms_mask, ["ms_mask"])
    plot_cmd_reddening_vector(membertable, origin, reddening_vector, star_cluster.name)

    # Apply linear transformation
    abs_ord_data = linear_transformation(cmd_data, origin, reddening_vector)
    membertable = _append_array_to_table(
        membertable,
        abs_ord_data,
        ["abscissa", "ordinate"],
        epoch=0,
    )

    # Rotation
    abs_ord_data_ms = linear_transformation(ms_data, origin, reddening_vector)
    membertable = _append_array_to_table(
        membertable, abs_ord_data, ["abscissa_ms", "ordinate_ms"]
    )

    # Get fiducial line (only for the first epoch)
    fiducial_line, median_abscissa, median_ordinate = fit_fiducial_line(
        abs_ord_data_ms, bin_width=drparams.bin_width
    )

    for epoch in range(epochs):

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

        # Get main sequence abscissa and ordinate for each epoch.
        # Not in use since fiducial line is only calculated for the first epoch.
        # abs_ord_data_ms = _cols2array(
        #     membertable[membertable["ms_mask"]],
        #     [f"abscissa_{epoch + 1}", f"ordinate_{epoch + 1}"],
        # )

        plot_rotated_cmd(
            membertable,
            fiducial_line,
            ref_stars_range,
            median_abscissa,
            median_ordinate,
            star_cluster.name,
            epoch,
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
    plot_dereddened_cmd_for_report(membertable, star_cluster.name, reddening_vector)

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
) -> tuple[np.ndarray, np.ndarray]:
    """Replace points that are outside a predefined rectangular region with NaNs"""

    data = data.copy()
    x1, x2, y1, y2 = limits
    above_inf = np.all(data >= np.array([[x1], [y1]]), axis=0)
    below_sup = np.all(data <= np.array([[x2], [y2]]), axis=0)
    mask = above_inf * below_sup
    data[:, ~mask] = np.nan
    return mask, data


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

    if np.isnan(rot_angle):
        # In case of negligible reddening vector, set it to -62.4 deg (Gaia DR2's value)
        rot_angle = np.deg2rad(-62.4)

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
    bin_width: float = 0.1,
) -> tuple[CubicSpline, np.ndarray, np.ndarray]:
    """Get the fiducial line of MS stars along the rotated CMD applying a moving window"""

    # Bin data along the ordinate (y axis)
    min_ordinate, max_ordinate = np.nanmin(data[1]), np.nanmax(data[1])

    # Compute shifted bins
    bins_ordinate = np.arange(min_ordinate, max_ordinate, bin_width)

    # Get the median of each bin along the abscissa and ordinate axis
    median_ordinate, _, _ = stats.binned_statistic(
        data[1], data[1], statistic=np.nanmedian, bins=bins_ordinate
    )

    median_abscissa, _, _ = stats.binned_statistic(
        data[1], data[0], statistic=_sigmaclip_median, bins=bins_ordinate
    )

    # Notice that we want the abscissa value as function of the ordinate value
    return (
        CubicSpline(median_ordinate, median_abscissa),
        median_abscissa,
        median_ordinate,
    )


def get_delta_abscissa(data: np.ndarray, fiducial_line: CubicSpline) -> np.ndarray:
    """Get the distance `Δ abscissa` from the fiducial line along the reddening direction"""

    return data[0] - fiducial_line(data[1])


def diffred_estimation(
    table: Table, epoch: int, k: float = 35, plots: bool = False
) -> np.ndarray:
    """Get the median color residual along the reddening vector (Δ abscissa) from k nearest neighbors"""

    # from sklearn.metrics import pairwise_distances
    # dist_matrix = pairwise_distances(coords.T, metric='haversine')

    # check that table lenght is greater than k
    if len(table) < k:
        raise ValueError(f"Table length must be greater than k ({k})")

    # Get coordinates
    coords = np.deg2rad(_cols2array(table, ["RA_ICRS", "DE_ICRS"]))

    ref_star_mask = table[f"ref_stars_{epoch}"]
    reference_star_table = table[ref_star_mask]
    coords_ref_stars = np.deg2rad(
        _cols2array(reference_star_table, ["RA_ICRS", "DE_ICRS"])
    )

    # Find k-nearest reference stars metric: haversine
    nn = NearestNeighbors(n_neighbors=k, metric="minkowski").fit(coords_ref_stars.T)
    dist, indxs = nn.kneighbors(coords.T, return_distance=True)

    # Get median Δ abscissa values from k nearest reference stars
    delta_abscissa = _cols2array(reference_star_table, [f"delta_abscissa_{epoch}"])
    neighbors_deltas = np.take(delta_abscissa, indxs)

    # check that the distance to a reference star is less than X arcminutes
    # dist = np.rad2deg(dist) * 60

    # Replace with nan the values that are not nearby
    # neighbors_deltas_n = np.where(dist < 3.0, neighbors_deltas, np.nan)
    # neighbors_deltas_n = np.where(dist > 0.001, neighbors_deltas, np.nan)
    # print("Avg. non nan values", np.mean(np.sum(~np.isnan(neighbors_deltas_n), axis=1)))

    # median_delta_abscissa = np.nanmedian(neighbors_deltas, axis=1)
    median_delta_abscissa = np.nanmedian(
        sigma_clip(
            neighbors_deltas,
            sigma_lower=3,
            sigma_upper=3,
            cenfunc="median",
            axis=1,
            masked=False,
        ),
        axis=1,
    )

    # Replace nan values with 0
    median_delta_abscissa = np.nan_to_num(median_delta_abscissa)

    # Test Plots (slow!)
    ref_ordinates = _cols2array(reference_star_table, [f"ordinate_{epoch}"])
    if plots:
        plot_difred_test(
            cluster_coords=coords,
            cluster_delta_abscissa=_cols2array(table, [f"delta_abscissa_{epoch}"]),
            cluster_ordinates=_cols2array(table, [f"ordinate_{epoch}"]),
            ref_coords=coords_ref_stars,
            ref_delta_abscissa=delta_abscissa,
            ref_ordinates=ref_ordinates,
            nn_coords=np.array(
                [
                    np.take(coords_ref_stars[0], indxs),
                    np.take(coords_ref_stars[1], indxs),
                ]
            ),
            nn_delta_abscissa=neighbors_deltas,
            nn_ordinates=np.take(ref_ordinates, indxs),
            median_values=median_delta_abscissa,
            object_name="Test",
            epoch=epoch,
        )

    # Compute correction
    print(
        f"epoch:{epoch}, Δ:{np.mean(median_delta_abscissa):.5f}",
    )
    table[f"median_delta_abscissa_{epoch}"] = median_delta_abscissa
    abscissa_corrected = table[f"abscissa_{epoch}"] - median_delta_abscissa
    table["abscissa_corrected"] = abscissa_corrected

    return abscissa_corrected.data
