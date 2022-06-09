from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.interpolate import CubicSpline

from src.settings import Config


def plot_cmd_reddening_vector(
    table: Table,
    origin: tuple[float, float],
    reddening_vector: tuple[float, float],
    object_name: str,
) -> None:
    """Plot CMD with reddening vector"""

    color = table["BP-RP"]
    magnitude = table["Gmag"]
    ms_color = table["ms_BP-RP"]
    ms_magnitude = table["ms_Gmag"]

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
    plt.xlabel("$G_{BP} - G_{RP}$")
    plt.ylabel("G")
    plt.legend()
    plt.gca().set_aspect("equal")
    fname = Config.PLOTDIR / f"{object_name}_cmd_reddening_vector.png"
    plt.savefig(fname)
    plt.clf()
    plt.close()


def plot_dereddened_cmd(
    table: Table,
    object_name: str,
) -> None:
    """Plot CMD with reddening vector"""

    color = table["BP-RP"]
    magnitude = table["Gmag"]
    color_dered = table["BP-RP_dered"]
    magnitude_dered = table["Gmag_dered"]

    # Plot CMD
    plt.figure(figsize=(6, 10))
    plt.scatter(color, magnitude, c="C3", s=10, alpha=0.3, label="original")
    plt.scatter(
        color_dered, magnitude_dered, c="C0", s=10, alpha=0.3, label="dereddened"
    )
    plt.title(f"De-reddended CMD {object_name}")
    plt.xlabel("$G_{BP} - G_{RP}$")
    plt.ylabel("G")
    plt.legend()
    # plt.gca().set_aspect("equal")
    fname = Config.PLOTDIR / f"{object_name}_dereddened_cmd.png"
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

    fname = Config.PLOTDIR / f"{object_name}_rotated_cmd_e{epoch}.png"
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
            c="grey",
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

    print("")
    print(f"cluster_coords {cluster_coords.shape}")
    print(f"cluster_delta_abscissa {cluster_delta_abscissa.shape}")
    print(f"cluster_ordinates {cluster_ordinates.shape}")
    print(f"ref_coords {ref_coords.shape}")
    print(f"ref_delta_abscissa {ref_delta_abscissa.shape}")
    print(f"ref_ordinates {ref_ordinates.shape}")
    print(f"nn_coords {nn_coords.shape}")
    print(f"nn_delta_abscissa {nn_delta_abscissa.shape}")
    print(f"nn_ordinates {nn_ordinates.shape}")
    print(f"median_values {median_values.shape}")

    # For each row in nn variable, plot the diferential reddening
    for i in range(nn_coords.shape[1]):
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


# Deprecated
def _plot_cmd(
    cmd_data: np.ndarray,
    reddening_vector: tuple[float, float],
    origin: tuple[float, float],
    object_name: str,
) -> None:
    """Plot CMD"""

    dpi = 60
    plt.figure(figsize=(920 / dpi, 720 / dpi), dpi=dpi)

    slope = reddening_vector[1] / reddening_vector[0]

    plt.scatter(cmd_data[0], cmd_data[1], alpha=0.5, s=10)
    plt.axline(origin, slope=slope, color="black", linestyle=(0, (5, 5)))
    plt.axline(origin, slope=-1.0 / slope, color="black", linestyle=(0, (5, 5)))
    plt.quiver(
        origin[0],
        origin[1],
        reddening_vector[0],
        reddening_vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.025,
        color="red",
        zorder=10,
    )

    theta_ab = -np.arctan2(reddening_vector[1], reddening_vector[0])
    ab_factor = 1.5
    ab_label = [
        (origin[0] + 0.1) + ab_factor * np.cos(theta_ab),
        (origin[1] + 0.1) - ab_factor * np.sin(theta_ab),
    ]
    plt.text(
        ab_label[0],
        ab_label[1],
        "Abscissa",
        fontsize=14,
        rotation=np.rad2deg(theta_ab),
        rotation_mode="anchor",
    )

    theta_or = np.pi * 0.5 + theta_ab
    or_factor = 0.4
    or_label = [
        (origin[0] + 0.2) + or_factor * np.cos(theta_or),
        (origin[1] + 0.1) - or_factor * np.sin(theta_or),
    ]
    plt.text(
        or_label[0],
        or_label[1],
        "Ordinate",
        fontsize=14,
        rotation=np.rad2deg(theta_or),
        rotation_mode="anchor",
    )
    plt.text(
        origin[0] - 0.15,
        origin[1] + 0.2,
        "$O$",
        fontsize=14,
    )

    plt.xlabel("$G_{BP} - G_{RP}$")
    plt.ylabel("$G$")
    ymin, ymax = plt.ylim()
    plt.ylim(ymax, ymin)

    plt.text(
        origin[0],
        ymin + 0.5,
        object_name.replace("_", " "),
        fontsize=14,
    )

    plt.gca().set_aspect("equal")
    plt.show()
    plt.clf()
    plt.close()
