import torch
import os
from pathlib import Path
from scipy.spatial import Delaunay
import yaml
import numpy as np
import math
from matplotlib.patches import Ellipse, Rectangle, Polygon
from matplotlib import cm

import functorch

from utils.config import Config

config = Config()

# from functorch import jacrev, jacfwd, vmap


def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)


def label_to_color(label):
    n_points = label.shape[0]
    color = np.zeros((n_points, 3))

    # color template (2021 pantone color: orbital)
    rgb = np.zeros((11, 3))
    rgb[0, :] = [253, 134, 18]
    rgb[1, :] = [106, 194, 217]
    rgb[2, :] = [111, 146, 110]
    rgb[3, :] = [153, 0, 17]
    rgb[4, :] = [179, 173, 151]
    rgb[5, :] = [245, 228, 0]
    rgb[6, :] = [255, 0, 0]
    rgb[7, :] = [0, 255, 0]
    rgb[8, :] = [0, 0, 255]
    rgb[9, :] = [18, 134, 253]
    rgb[10, :] = [155, 155, 155]  # grey

    for idx_color in range(10):
        color[label == idx_color, :] = rgb[idx_color, :]
    return color


def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def PD_metric_to_ellipse(G, center, scale, **kwargs):
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # find angle of ellipse
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    # draw ellipse
    width, height = 2 * scale * np.sqrt(eigvals)
    return Ellipse(
        xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs
    )


def rectangle_scatter(size, center, color):
    return Rectangle(
        xy=(center[0] - size[0] / 2, center[1] - size[1] / 2),
        width=size[0],
        height=size[1],
        facecolor=color,
    )


def triangle_scatter(size, center, color):
    return Polygon(
        (
            (center[0], center[1] + size[1] / 2),
            (center[0] - size[0] / 2, center[1] - size[1] / 2),
            (center[0] + size[0] / 2, center[1] - size[1] / 2),
        ),
        fc=color,
    )


def minmax(item):
    return (item) / (torch.max(item) - torch.min(item)) + (
        torch.max(item) - torch.min(item)
    ) / 2


def cmap_labels(labels, cmap=cm.turbo):
    """
    convert labels
    """
    # apply cmap and change base
    new_labels = (cmap(labels) * 255).astype(int)
    # remove opacity channel from rgba
    new_labels = torch.tensor(new_labels[:, :-1])

    return new_labels


def batch_jacobian(f, input):
    """
    Compute the diagonal entries of the jacobian of f with respect to x
    :param f: the function
    :param x: where it is to be evaluated
    :return: diagonal of df/dx. First dimension is the derivative
    """

    # compute vectorized jacobian. For curvature because of nested derivatives, for some of the functions
    # the forward mode AD is not implemented
    # if input.ndim == 1:
    #    try:
    #        jac = jacfwd(f)(input)
    #    except NotImplementedError:
    #        jac = jacrev(f)(input)

    # else:

    try:
        jac = functorch.vmap(functorch.jacrev(f), in_dims=(0,))(input)
    except NotImplementedError:
        jac = torch.func.vmap(torch.func.jacrev(f), in_dims=(0,))(input)

    return jac


def get_sc_kwargs():
    sc_kwargs = {
        "marker": ".",
        "alpha": 0.5,
        "s": 20,
        "edgecolors": None,
        "linewidth": 0.0,
    }

    return sc_kwargs


def get_saving_kwargs():
    kwargs = {"format": "png", "pad_inches": 0, "dpi": 100}  # 40

    return kwargs


def get_saving_dir(model_name, dataset_name, filename):
    root = Path(os.path.join(config["output_path"], dataset_name, model_name))
    root.mkdir(parents=True, exist_ok=True)
    return os.path.join(root.resolve(), filename)


def get_hull(points):
    """
    Calculates the Delaunay hull for points
    :param points:
    :return:
    """

    return Delaunay(points)


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    return hull.find_simplex(p) >= 0


from scipy.spatial import KDTree

def get_nearest_grid_points(data, num_steps=20):
    """
    Extracts m x n points from a point cloud that are closest to a regular m x n grid.

    Args:
    - point_cloud (numpy.ndarray): Array of shape (N, 2) representing the point cloud in R^2.
    - m (int): Number of rows in the grid.
    - n (int): Number of columns in the grid.

    Returns:
    - numpy.ndarray: Array of shape (m * n, 2) containing the extracted points.
    """

    x_min = torch.min(data[:, 0]).item()
    x_max = torch.max(data[:, 0]).item()
    y_min = torch.min(data[:, 1]).item()
    y_max = torch.max(data[:, 1]).item()

    factor = 0.6

    if num_steps != None:
        num_steps_x = num_steps
        num_steps_y = math.ceil((y_max - y_min) / (x_max - x_min) * num_steps_x)

    # Create grid points
    grid_points = np.array([[i, j] for i in torch.linspace(x_min, x_max, num_steps_x) for j in torch.linspace(y_min, y_max, num_steps_y)])

    # Build KD-tree for efficient nearest neighbor search
    kdtree = KDTree(data)

    # Query the KD-tree for nearest neighbors to each grid point
    _, nearest_indices = kdtree.query(grid_points)

    return np.unique(nearest_indices)

def get_coordinates(
    latent_activations,
    grid=None,
    num_steps=20,
    coords0=None,
    model_name=None,
    dataset_name=None,
):
    """
    Get indicatrix positions
    Args:
        latent_activations: the embedding considered
        grid: the type of grid we consider
        num_steps: the number of steps in the orizontal direction
        coords0: one fixed coordinate that should be part of thhe grid
        model_name: the name of the model which created the embedding
        dataset_name: the name of the dataset considered

    Returns:
        None
    """

    x_min = torch.min(latent_activations[:, 0]).item()
    x_max = torch.max(latent_activations[:, 0]).item()
    y_min = torch.min(latent_activations[:, 1]).item()
    y_max = torch.max(latent_activations[:, 1]).item()

    factor = 0.6

    if num_steps != None:
        num_steps_x = num_steps
        print(latent_activations)
        print(latent_activations.shape, num_steps_x, x_max, x_min)
        num_steps_y = math.ceil((y_max - y_min) / (x_max - x_min) * num_steps_x)

        step_size_x = (x_max - x_min) / (num_steps_x)
        step_size_y = (y_max - y_min) / (num_steps_y)

    if grid == "dataset":
        coordinates = latent_activations
    elif grid == "on_data":
        if coords0 is not None:
            x_0 = coords0[0].item()
            y_0 = coords0[1].item()

            num_steps_left = int((x_0 - x_min) / (x_max - x_min) * num_steps_x)
            num_steps_right = num_steps - num_steps_left

            num_steps_up = int((y_max - y_0) / (y_max - y_min) * num_steps_y)
            num_steps_down = num_steps_y - num_steps_up

            x_left = x_0 - np.arange(num_steps_left) * step_size_x
            x_right = x_0 + np.arange(num_steps_right) * step_size_x

            y_down = y_0 - np.arange(num_steps_down) * step_size_y
            y_up = y_0 + np.arange(num_steps_up) * step_size_y

            x_left = np.flip(np.array(x_left))[:-1]
            x_right = np.array(x_right)
            y_up = np.array(y_up)
            y_down = np.flip(np.array(y_down))[:-1]

            xs = torch.from_numpy(np.concatenate((x_left, x_right))).float()
            ys = torch.from_numpy(np.concatenate((y_down, y_up))).float()

        else:
            xs = torch.linspace(x_min, x_max, steps=num_steps_x)
            ys = torch.linspace(y_min, y_max, steps=num_steps_y)

        num_tiles = len(xs) * len(ys)
        mean_data_per_tile = len(latent_activations) / num_tiles

        coordinates = []
        num_xs = len(xs)
        num_ys = len(ys)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                mask_x = torch.logical_and(
                    latent_activations[:, 0] >= x - step_size_x / 2,
                    latent_activations[:, 0] <= x + step_size_x / 2,
                )
                mask_y = torch.logical_and(
                    latent_activations[:, 1] >= y - step_size_y / 2,
                    latent_activations[:, 1] <= y + step_size_y / 2,
                )

                mask = torch.logical_and(mask_x, mask_y)
                in_tile = latent_activations[mask].shape[0]

                required_data_per_tile = factor * mean_data_per_tile
                if (i == 0 or i == num_xs - 1) or (j == 0 or j == num_ys - 1):
                    if (i == 0 or i == num_xs - 1) and (j == 0 or j == num_ys - 1):
                        required_data_per_tile = required_data_per_tile / 4
                    else:
                        required_data_per_tile = required_data_per_tile / 2

                if in_tile >= required_data_per_tile:
                    coordinates.append(torch.tensor([x, y]))

        coordinates = torch.stack(coordinates)
    elif grid == "convex_hull":
        coordinates = []
        xs = torch.linspace(x_min, x_max, steps=num_steps_x)
        ys = torch.linspace(y_min, y_max, steps=num_steps_y)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                coordinates.append(torch.tensor([x, y]))

        coordinates = torch.stack(coordinates)
    else:
        coordinates = None

    hull = get_hull(latent_activations)
    coordinates = coordinates[in_hull(coordinates, hull)]

    return coordinates


def generate_unit_vectors(n, base_point, metric):
    """
    calculate polygon lengths using metric
    :param n: number of vectors
    :param base_point: the base point
    :return: array of norms
    """

    # the angles
    phi = torch.linspace(0.0, 2 * np.pi, n)

    # generate circular vector patch
    raw_vectors = torch.stack([torch.sin(phi), torch.cos(phi)])

    # normalize vectors in pullback metric
    # norm_vectors = metric.norm(raw_vectors, matrix=metric)

    norm_vectors = torch.einsum("ijk,kl->ijl", metric, raw_vectors)
    norm_vectors = torch.einsum("mn,imn->in", raw_vectors, norm_vectors)
    norm_vectors = torch.sqrt(norm_vectors)

    norm_vectors = norm_vectors.unsqueeze(2).expand(
        *norm_vectors.shape, raw_vectors.shape[0]
    )

    # reshape the raw vectors
    raw_vectors = raw_vectors.unsqueeze(2).expand(
        *raw_vectors.shape, base_point.shape[0]
    )
    raw_vectors = torch.transpose(raw_vectors, dim0=0, dim1=2)

    # normalize the vector patches
    unit_vectors = torch.where(
        norm_vectors != 0, raw_vectors / norm_vectors, torch.zeros_like(raw_vectors)
    )

    return unit_vectors, norm_vectors


def random_metric_field_generator(
    num_samples, dim, sigma, local_coordinates="exponential"
):
    if local_coordinates == "exponential":
        S = torch.randn(num_samples, dim, dim) * sigma
        S = (S + S.permute(0, 2, 1)) / 2
        eigh = torch.linalg.eigh(S)
        random_G = (
            eigh.eigenvectors
            @ torch.diag_embed(torch.exp(eigh.eigenvalues))
            @ (eigh.eigenvectors.permute(0, 2, 1))
        )
    elif local_coordinates == "cholesky":
        S = torch.randn(num_samples, dim, dim) * sigma
        S = torch.tril(S)
        random_G = S @ S.permute(0, 2, 1)
    return random_G


def get_significant(val):
    i = 0
    in_sig = False
    while i < len(str(val)):
        if str(val)[i] not in [".", "0"]:
            in_sig = True

        if in_sig:
            if int(str(val)[i]) <= 2:
                return i + 1
            else:
                return i

        i += 1


def get_next_digit(val, i):
    val_10 = float(val) * 10 ** (i)
    return int(str(val_10).split(".")[1][0])


def round_significant(data, errors):
    """
    Round to first significant digit
    """
    results = []

    i = 0
    while i < len(data):
        dist = get_significant(errors[i])

        if errors[i] != 0.0:
            dist = int(math.floor(math.log10(abs(errors[i]))))
        else:
            dist = 1

        if get_next_digit(errors[i], -dist) <= 2.0:
            dist -= 1

        if errors[i] != 0:
            err = round(errors[i], -dist)
            val = round(data[i], -dist)
        else:
            err = round(errors[i], 1)
            val = round(data[i], 2)

        if err.is_integer():
            err = int(err)

        if val.is_integer():
            val = int(val)

        results.append(f"{val}" + " $\pm$ " + f"{err}")

        i += 1

    return results
