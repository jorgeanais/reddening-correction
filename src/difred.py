"""
https://numpy.org/doc/stable/reference/typing.html#module-numpy.typing
"""
from astropy.table import Table
import numpy as np
import numpy.typing as npt
import pickle

from src.settings import Config


def differential_reddening(
    cmd_data: np.ndarray,
    reddening_vector: tuple[float, float],
    origin: tuple[float, float] | None,
) -> np.ndarray:
    """Differential Reddening correction"""

    # CMD translation
    o = np.array([origin[0], origin[1]]).reshape(2, 1)
    data = cmd_data - o

    # CMD Rotation
    theta = -np.arctan2(reddening_vector[1], reddening_vector[0])
    print(f"Rotation angle: {np.rad2deg(theta):.1f}°")
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    rotated_data = rotation_matrix @ data
    return rotated_data

    # Generation of fiducial line

    # Δ abscissa from fiducial line

    # Selection of reference stars

    # Estimation of differential extinction


def cols2array(table: Table, columns: list[str]) -> np.ndarray:
    """Convert a list of columns to an array."""

    return np.array([table[col].data for col in columns])


def plot_cmd(
    cmd_data: np.ndarray,
    reddening_vector: tuple[float, float],
    origin: tuple[float, float],
    object_name: str,
) -> None:
    """Plot CMD"""

    import matplotlib.pyplot as plt

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


if __name__ == "__main__":

    reddening_vector = (0.31, 0.59)
    origin = (0.36, 11.93)
    test_path = Config.TEST_DATA / "NGC_2099_.pkl"
    with open(str(test_path), "rb") as file:
        cmd_data = pickle.load(file)

    plot_cmd(cmd_data, reddening_vector, origin, "NGC_2099")
    
    rotated_data = differential_reddening(cmd_data, reddening_vector, origin)
