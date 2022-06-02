import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from src.settings import Config

def plot_cmd_reddening_vector(
    all_data: np.ndarray,
    ms_data: np.ndarray,
    origin: tuple[float, float],
    reddening_vector: tuple[float, float],
    object_name: str,
) -> None:
    """Plot CMD with reddening vector"""

    # Plot CMD
    plt.figure(figsize=(6, 10))
    plt.scatter(all_data[0], all_data[1], s=10, alpha=0.3, label="All")
    plt.scatter(ms_data[0], ms_data[1], s=12, alpha=0.5, label="MS selection")
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


def plot_rotated_cmd(
    rot_data: np.ndarray,
    fiducial_line: CubicSpline,
    object_name: str,
) -> None:
    """Plot rotated CMD"""
    plt.scatter(rot_data[0], rot_data[1], s=10, alpha=0.3, label="Rotated data")
    # TODO: include fiducial line in the plot

    plt.title(f"Rotated CMD {object_name}")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.xlabel("Abscissa")
    plt.ylabel("Ordinate")
    fname = Config.PLOTDIR / f"{object_name}_rotated_cmd.png"
    plt.savefig(fname)

    

# Deprecated
def plot_cmd(
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
