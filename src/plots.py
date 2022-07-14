from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.interpolate import CubicSpline

from src.settings import Config


RELABEL = {
    "mk": r"$K_{s}$",
    "mj-mk": r"$J - K_{s}$",
    "Gmag": r"$G$",
    "BP-RP": r"$G_{BP} - G_{RP}$",
    "Gmag_dered": r"$G$",
    "BP-RP_dered": r"$G_{BP} - G_{RP}$",
}


def plot_cmd_reddening_vector(
    table: Table,
    origin: tuple[float, float],
    reddening_vector: tuple[float, float],
    object_name: str,
    color_col: str = "BP-RP",
    magnitude_col: str = "Gmag",
) -> None:
    """Plot CMD with reddening vector"""

    color = table[color_col]
    magnitude = table[magnitude_col]
    ms_color = table[f"ms_{color_col}"]
    ms_magnitude = table[f"ms_{magnitude_col}"]

    # Plot CMD
    plt.figure(figsize=(6, 10))
    plt.scatter(color, magnitude, c="C1", s=10, alpha=0.3, label="All")
    plt.scatter(ms_color, ms_magnitude, c="C0", s=12, alpha=0.5, label="MS selection")
    plt.quiver(
        origin[0],
        origin[1],
        reddening_vector[0],
        reddening_vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.03,
        color="red",
        zorder=10,
    )
    plt.title(f"CMD {object_name}")
    plt.xlabel(RELABEL[color_col] if color_col in RELABEL else color_col)
    plt.ylabel(RELABEL[magnitude_col] if magnitude_col in RELABEL else magnitude_col)
    plt.legend()
    plt.gca().set_aspect("equal")
    fname = Config.DIFREDDIR / f"{object_name}_cmd_reddening_vector.png"
    plt.savefig(fname)
    plt.clf()
    plt.close()


def plot_dereddened_cmd(
    table: Table,
    object_name: str,
    color_col: str = "BP-RP",
    magnitude_col: str = "Gmag",
) -> None:
    """Plot CMD with reddening vector"""

    color = table[color_col]
    magnitude = table[magnitude_col]
    color_dered = table[f"{color_col}_dered"]
    magnitude_dered = table[f"{magnitude_col}_dered"]

    # Plot CMD
    plt.figure(figsize=(6, 10))
    plt.scatter(color, magnitude, c="C3", s=10, alpha=0.3, label="original")
    plt.scatter(
        color_dered, magnitude_dered, c="C0", s=10, alpha=0.3, label="dereddened"
    )
    plt.title(f"De-reddended CMD {object_name}")
    plt.xlabel(RELABEL[color_col] if color_col in RELABEL else color_col)
    plt.ylabel(RELABEL[magnitude_col] if magnitude_col in RELABEL else magnitude_col)
    plt.legend()
    # plt.gca().set_aspect("equal")
    fname = Config.DIFREDDIR / f"{object_name}_dereddened_cmd.png"
    plt.savefig(fname)
    plt.clf()
    plt.close()


def plot_dereddened_cmd_for_report(
    table: Table,
    object_name: str,
    reddening_vector: tuple[float, float],
    color_col: str = "BP-RP",
    magnitude_col: str = "Gmag",
) -> None:
    """Plot CMD with reddening vector for report"""

    color = table[color_col]
    magnitude = table[magnitude_col]
    color_dered = table[f"{color_col}_dered"]
    magnitude_dered = table[f"{magnitude_col}_dered"]

    # Plot CMD
    plt.figure(figsize=(10, 8))
    plt.suptitle(object_name.replace("_", " "))

    ax1 = plt.subplot(121)
    plt.scatter(color, magnitude, c="C0", s=15, alpha=0.3, label="original")
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    plt.quiver(
        xmin + (xmax - xmin) * 0.1,
        ymin + (ymax - ymin) * 0.1,
        reddening_vector[0],
        reddening_vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.01,
        color="red",
        zorder=10,
    )
    ax1.invert_yaxis()
    plt.xlabel(RELABEL[color_col] if color_col in RELABEL else color_col)
    plt.ylabel(RELABEL[magnitude_col] if magnitude_col in RELABEL else magnitude_col)
    plt.title("Original")

    ax2 = plt.subplot(122, sharey=ax1)
    plt.scatter(
        color_dered, magnitude_dered, c="C0", s=15, alpha=0.3, label="Corrected"
    )
    plt.xlabel(RELABEL[color_col] if color_col in RELABEL else color_col)
    plt.ylabel(RELABEL[magnitude_col] if magnitude_col in RELABEL else magnitude_col)
    plt.title("Corrected")

    plt.tight_layout()
    fname = Config.DIFREDDIR / f"{object_name}_dereddened_cmd_report.png"
    plt.savefig(fname)
    plt.clf()
    plt.close()


def plot_rotated_cmd(
    table: Table,
    fiducial_line: CubicSpline,
    ref_stars_range: tuple[float, float],
    median_abscissa: np.ndarray,
    median_ordinate: np.ndarray,
    object_name: str,
    epoch: int,
) -> None:
    """Plot rotated CMD"""

    abscissa = table[f"abscissa_{epoch}"]
    ordinate = table[f"ordinate_{epoch}"]
    delta_abscissa = table[f"delta_abscissa_{epoch}"]
    refstars_mask = table[f"ref_stars_{epoch}"]
    print(np.sum(refstars_mask))

    # Rotated MS
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Rotated CMD {object_name} epoch {epoch}")

    # Rotated CMD ----------------
    ax = plt.subplot(121)
    plt.scatter(abscissa, ordinate, s=10, alpha=0.2, label="All")
    plt.scatter(
        abscissa[refstars_mask],
        ordinate[refstars_mask],
        s=10,
        alpha=0.5,
        label="Reference stars",
    )

    # Fiducial line
    ys = np.linspace(median_ordinate.min(), median_ordinate.max(), 100)
    plt.plot(fiducial_line(ys), ys, label="Fiducial line", zorder=10, c="C3")

    # Median values
    plt.plot(median_abscissa, median_ordinate, "o", color="yellow")

    plt.legend()
    plt.xlabel("Abscissa")
    plt.ylabel("Ordinate")
    plt.xlim(-2.0, 7.0)  # TODO: remove limits

    # Delta abscissa ----------------
    plt.subplot(122, sharey=ax)
    plt.scatter(delta_abscissa, ordinate, s=10, alpha=0.3, label="MS")
    plt.scatter(
        delta_abscissa[refstars_mask],
        ordinate[refstars_mask],
        s=10,
        alpha=0.5,
        label="Reference stars",
    )
    plt.axvline(x=0.00, c="grey", linestyle="--", alpha=0.5)
    plt.axhline(y=ref_stars_range[0], color="grey", linestyle="--", alpha=0.5)
    plt.axhline(y=ref_stars_range[1], color="grey", linestyle="--", alpha=0.5)

    plt.legend()
    plt.xlabel("$\Delta$ Abscissa")
    plt.xlim(-1.2, 1.2)  # TODO: remove limits

    fname = Config.DIFREDDIR / f"{object_name}_rotated_cmd_e{epoch}.png"
    plt.savefig(fname)
    plt.close()


def plot_difred_test(
    cluster_coords: np.ndarray,
    cluster_delta_abscissa: np.ndarray,
    cluster_ordinates: np.ndarray,
    ref_coords: np.ndarray,
    ref_delta_abscissa: np.ndarray,
    ref_ordinates: np.ndarray,
    nn_coords: np.ndarray,
    nn_delta_abscissa: np.ndarray,
    nn_ordinates: np.ndarray,
    median_values: np.ndarray,
    object_name: str,
    epoch: int,
) -> None:
    """Plot diferential reddening calculation for each star and iteration"""

    def _individual_plot(
        object_name: str,
        epoch: int,
        star_ra: float,
        star_dec: float,
        star_ord: float,
        star_delta_abs: float,
        cl_ra: np.ndarray,
        cl_dec: np.ndarray,
        cl_ord: np.ndarray,
        cl_delta_abs: np.ndarray,
        ref_ra: np.ndarray,
        ref_dec: np.ndarray,
        ref_ord: np.ndarray,
        ref_delta_abs: np.ndarray,
        nn_ra: np.ndarray,
        nn_dec: np.ndarray,
        nn_ord: np.ndarray,
        nn_delta_abs: np.ndarray,
        median_value: float,
        iteration: int,
    ) -> None:
        """Plot diferential reddening calculation for each individual star"""
        plt.figure(figsize=(18, 6))
        plt.suptitle(f"Rotated CMD {object_name} epoch {epoch}")

        # Spatial plot
        plt.subplot(121)
        plt.scatter(
            cl_ra,
            cl_dec,
            s=20,
            alpha=0.3,
            label="Cluster stars",
            marker=".",
            color="C7",
        )
        plt.scatter(
            ref_ra,
            ref_dec,
            s=40,
            alpha=0.3,
            label="Reference stars",
            marker="x",
            color="C0",
        )
        plt.scatter(
            nn_ra,
            nn_dec,
            s=40,
            alpha=0.7,
            label="Nearest neighbors",
            marker="^",
            color="C1",
        )
        plt.scatter(
            star_ra,
            star_dec,
            s=70,
            alpha=1.0,
            label="Target star",
            marker="*",
            color="C3",
        )
        plt.xlabel("RA (deg)")
        plt.ylabel("DEC (deg)")
        plt.legend()
        plt.gca().set_aspect("equal")

        # ordinate vs Δ abscissa plot
        plt.subplot(122)
        plt.scatter(
            cl_delta_abs,
            cl_ord,
            s=20,
            alpha=0.3,
            label="Cluster stars",
            marker=".",
            color="C7",
        )
        plt.scatter(
            ref_delta_abs,
            ref_ord,
            s=40,
            alpha=0.3,
            label="Reference stars",
            marker="x",
            color="C0",
        )
        plt.scatter(
            nn_delta_abs,
            nn_ord,
            s=40,
            alpha=0.7,
            label="Nearest neighbors",
            marker="^",
            color="C1",
        )
        plt.scatter(
            star_delta_abs,
            star_ord,
            s=70,
            alpha=1.0,
            label="Target star",
            marker="*",
            color="C3",
        )
        plt.axvline(
            x=median_value,
            linestyle="--",
            alpha=0.9,
            label="Median",
            color="C2",
        )

        ymin, ymax = plt.ylim()
        plt.xlim(-2.0, 2.0)
        plt.ylim(0.0, ymax)
        plt.xlabel(r"$\Delta$ abscissa")
        plt.ylabel("Ordinate")
        plt.legend()

        # Save
        fname = Config.DIFREDDIR / f"difred_{object_name}_{epoch}_{iteration}.png"
        plt.savefig(fname)
        plt.close()

    # For each row in nn variable, plot the diferential reddening
    for i in range(nn_coords.shape[1]):
        if i % 100 == 0:
            print(i)
        _individual_plot(
            object_name=object_name,
            epoch=epoch,
            star_ra=cluster_coords[0, i],
            star_dec=cluster_coords[1, i],
            star_ord=cluster_ordinates[0, i],
            star_delta_abs=cluster_delta_abscissa[0, i],
            cl_ra=cluster_coords[0],
            cl_dec=cluster_coords[1],
            cl_ord=cluster_ordinates,
            cl_delta_abs=cluster_delta_abscissa,
            ref_ra=ref_coords[0],
            ref_dec=ref_coords[1],
            ref_ord=ref_ordinates,
            ref_delta_abs=ref_delta_abscissa,
            nn_ra=nn_coords[0, i, :],
            nn_dec=nn_coords[1, i, :],
            nn_ord=nn_ordinates[i],
            nn_delta_abs=nn_delta_abscissa[i],
            median_value=median_values[i],
            iteration=i,
        )


def plot_dif_dist_corrected(
    table: Table,
    object_name: str,
    color_col: str = "BP-RP_dered_corr",
    magnitude_col: str = "Gmag_dered_corr",
):
    """Plot differential correction and distance correction"""

    color = table[color_col]
    magnitude = table[magnitude_col]

    plt.scatter(color, magnitude, c="C0", s=15, alpha=0.3, label="Corrected")
    plt.gca().invert_yaxis()
    plt.title(f"CMD {object_name.replace('_', ' ')}")
    plt.xlabel(RELABEL[color_col] if color_col in RELABEL else color_col)
    plt.ylabel(RELABEL[magnitude_col] if magnitude_col in RELABEL else magnitude_col)
    fname = Config.DIFDISTCOR / f"{object_name}_cmd.png"
    plt.savefig(fname)
    plt.clf()
    plt.close()
