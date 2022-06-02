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
    #plt.gca().set_aspect("equal")
    fname = Config.PLOTDIR / f"{object_name}_dereddened_cmd.png"
    plt.savefig(fname)
    plt.clf()


def plot_rotated_cmd(
    table: Table,
    fiducial_line: CubicSpline,
    ref_stars_range: tuple[float, float],
    object_name: str,
    epoch: int = 3,
) -> None:
    """Plot rotated CMD"""

    abscissa = table["abscissa"]
    ordinate = table["ordinate"]
    delta_abscissa = table[f"delta_abscissa_{epoch}"]
    refstars_mask = table[f"ref_stars_{epoch}"]
    print(np.sum(refstars_mask))

    # Rotated MS
    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Rotated CMD {object_name}")

    # Rotated CMD ----------------
    ax = plt.subplot(121)
    plt.scatter(abscissa, ordinate, s=10, alpha=0.3, label="MS selection")
    plt.scatter(
        abscissa[refstars_mask],
        ordinate[refstars_mask],
        s=10,
        alpha=0.5,
        label="Reference stars",
    )

    # Fiducial line
    ys = np.linspace(np.nanmin(ordinate), np.nanmax(ordinate), 100)
    plt.plot(fiducial_line(ys), ys, label="Fiducial line", zorder=10, c="red")

    plt.legend()
    plt.xlabel("Abscissa")
    plt.ylabel("Ordinate")
    plt.xlim(-1.0, 7.0)  # TODO: remove limits

    # Delta abscissa ----------------
    plt.subplot(122, sharey=ax)
    plt.scatter(delta_abscissa, ordinate, s=10, alpha=0.3, label="MS selection")
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
    plt.xlim(-2.0, 1.0)  # TODO: remove limits

    fname = Config.PLOTDIR / f"{object_name}_rotated_cmd_e{epoch}.png"
    plt.savefig(fname)


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
