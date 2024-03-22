from matplotlib import pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


from experiments.util import generate_tradeoff_data, get_dl, normalize

from models import load_pretrained
from geometry import get_pullbacked_Riemannian_metric, get_pushforwarded_Riemannian_metric

from experiments.util import (
    load_model,
    reg_strength_data,
    determine_scaling_fn,
    values_in_quantile,
)

from utils.config import Config
from utils.utils import (
    get_sc_kwargs,
    get_saving_kwargs,
    get_saving_dir,
    get_coordinates,
    generate_unit_vectors,
)


config = Config()


def reg_strength_plot(dataset, metric, results):
    plt.figure(figsize=(5, 3))

    for model in config["models"]:
        if model == "ae":
            continue

        regs, mean_vals, std_vals = reg_strength_data(model, dataset, metric, results)

        plt.errorbar(regs, mean_vals, yerr=std_vals, fmt="o--", label=model)

    # ae baseline
    baseline = np.array(results[dataset]["ae"][""][f"{metric}_"])

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)

    x_range = plt.xlim()
    xs = np.linspace(x_range[0], x_range[1], 3)

    plt.plot(
        xs,
        baseline_mean * np.ones_like(xs, dtype=float),
        color="black",
        label="vanilla AE",
        linestyle="--",
    )
    plt.fill_between(
        x=xs,
        y1=baseline_mean + baseline_std,
        y2=baseline_mean - baseline_std,
        color="black",
        alpha=0.4,
    )

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig(
        get_saving_dir("tradeoff", dataset, f"regs_vs_{metric}.png"),
    )

    if config["show"]:
        plt.show()
        plt.close()


def tradeoff_plot(dataset, metric, results):
    fig = plt.figure(figsize=(5, 3))
    # plt.title(f"{metric} vs {x} @ {dataset}", fontsize=15)

    tomse = config["type_of_mse"]

    max_vals = -float("inf")
    max_means = -float("inf")
    for model in config["models"]:
        if model == "ae":
            continue

        mean_mses, mean_vals, std_mses, _, _ = generate_tradeoff_data(
            results[dataset][model], metric
        )

        # normalize xs with values of ae
        mean_mses = mean_mses / np.mean(results[dataset]["ae"][""][f"{tomse}_"])
        std_mses = std_mses / np.mean(results[dataset]["ae"][""][f"{tomse}_"])

        plt.errorbar(
            mean_vals, mean_mses, yerr=std_mses, fmt="o--", label=model
        )  # xerr=std_vals
        # if metric == METRIC_FOR_MODEL[model]:
        #    plt.errorbar(vals[idx], xs[idx], yerr=std_xs[idx], fmt="o", color="red")
        if metric == f"{config['METRIC_FOR_MODEL'][model]}":
            if np.max(mean_vals) > max_vals:
                max_vals = np.max(mean_vals)
            if np.max(mean_mses) > max_means:
                max_xs = np.max(mean_mses)

    # ae baseline
    baseline = np.array(results[dataset]["ae"][""][f"{tomse}_"])
    baseline_mean = 1.0
    baseline_std = np.std(baseline) / np.mean(baseline)

    if metric != "VP":
        x_max = min(20.0, 11 / 10 * max_vals)
    else:
        x_max = min(0.5e7, 11 / 10 * max_vals)

    DPI = fig.get_dpi()
    total_width_pixels = DPI * fig.get_size_inches()[0]
    x_min = -(10 / total_width_pixels) * x_max

    if dataset != "earth":
        y_min = 0.9
    else:
        y_min = 0.5

    y_max = min(5, max_xs + 0.1)

    plt.gca().set_xlim(x_min, x_max)
    plt.gca().set_ylim(y_min, y_max)

    # plt.gca().set_xlim(-max_vals / 10, max_vals + max_vals / 10)
    # plt.gca().set_ylim(0.9, 3/2*max_xs)

    x_range = plt.xlim()
    xs = np.linspace(x_range[0], x_range[1], 3)

    plt.plot(
        xs,
        baseline_mean * np.ones_like(xs),
        color="black",
        label="vanilla AE",
        linestyle="--",
    )
    plt.fill_between(
        x=xs,
        y1=baseline_mean + baseline_std,
        y2=baseline_mean - baseline_std,
        color="black",
        alpha=0.4,
    )

    plt.legend()

    plt.savefig(
        get_saving_dir("tradeoff", dataset, f"mses_vs_{metric}.png"),
    )

    if config["show"]:
        plt.show()
        plt.close()


"""
Indicatrix Plots
"""


def latent_plot(model, model_name, dataset_name, reg, test_dl, train_dl):
    data = torch.cat((test_dl.dataset.data, train_dl.dataset.data))
    targets = torch.cat((test_dl.dataset.targets, train_dl.dataset.targets))
    Z = model.encode(data).detach().cpu()  # .numpy()

    if reg == "":
        reg = 0

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.0)
    plt.margins(0.01, 0.01)

    c = targets.detach().cpu()

    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=c,
        **get_sc_kwargs(),
        zorder=0,
    )
    # plt.title(f"{model_name} @ {dataset_name} (reg={float(reg)})")
    plt.savefig(
        get_saving_dir(model_name, dataset_name, f"latents_reg{float(reg)}.png"),
        **get_saving_kwargs(),
    )

    if config["show"]:
        plt.show()


def latent_plots(dataset_name, model_regs):
    seed = 1

    for model_name, reg in model_regs:
        model, cfg = load_model(model_name, dataset_name, seed, reg)
        train_dl = get_dl(cfg, dataset_name, split="train")
        test_dl = get_dl(cfg, dataset_name)

        print(dataset_name, model_name)

        latent_plot(model, model_name, dataset_name, reg, test_dl, train_dl)

    
def indicatrix_plot(model, model_name, dataset_name, reg, test_dl, train_dl):
    # TODO: extract model_name and datase_name from model and test_dl
    data = torch.cat((test_dl.dataset.data, train_dl.dataset.data))
    targets = torch.cat((test_dl.dataset.targets, train_dl.dataset.targets))
    Z = model.encode(data).detach().cpu()  # .numpy()

    if reg == "":
        reg = 0

    coordinates = get_coordinates(
        data.view(data.shape[0], -1),
        grid="on_data",
        num_steps=15,
        dataset_name=dataset_name,
        model_name=model_name,
    ).to(config["device"])

    # calculate grid step sizes
    x_min = torch.min(Z[:, 0]).item()
    x_max = torch.max(Z[:, 0]).item()
    y_min = torch.min(Z[:, 1]).item()
    y_max = torch.max(Z[:, 1]).item()

    num_steps_x = 10
    num_steps_y = int((y_max - y_min) / (x_max - x_min) * num_steps_x)

    step_size_x = (x_max - x_min) / (num_steps_x)
    step_size_y = (y_max - y_min) / (num_steps_y)
    stepsize = min(step_size_x, step_size_y)

    # Z_pinned_data = model.encode(pinned_data)
    G = get_pushforwarded_Riemannian_metric(model.encode, coordinates)

    vector_patches, _ = generate_unit_vectors(100, coordinates, G)
    vector_norms = torch.linalg.norm(vector_patches.reshape(-1, 2), dim=1)
    max_vector_norm = torch.max(vector_norms[torch.nonzero(vector_norms)])

    normed_vector_patches = (
        vector_patches / max_vector_norm * stepsize / 2
    )  # * scaling_factor  # / 3
    anchored_vector_patches = (
        coordinates.unsqueeze(1).expand(*normed_vector_patches.shape)
        + normed_vector_patches
    )

    # create polygons
    polygons = PatchCollection(
        [Polygon(tuple(vector.tolist()), True) for vector in anchored_vector_patches]
    )
    polygons.set_color([0 / 255, 0 / 255, 0 / 255, 0.3])

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.add_collection(polygons)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.0)
    plt.margins(0.01, 0.01)

    c = targets.detach().cpu()

    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=c,
        **get_sc_kwargs(),
        zorder=0,
    )
    # plt.title(f"{model_name} @ {dataset_name} (reg={float(reg)})")
    plt.savefig(
        get_saving_dir(model_name, dataset_name, f"indicatrix_reg{float(reg)}.png"),
        **get_saving_kwargs(),
    )

    if config["show"]:
        plt.show()


def indicatrix_plots(dataset_name, model_regs):
    seed = 1

    for model_name, reg in model_regs:
        model, cfg = load_model(model_name, dataset_name, seed, reg)
        train_dl = get_dl(cfg, dataset_name, split="train")
        test_dl = get_dl(cfg, dataset_name)

        print(dataset_name, model_name)

        indicatrix_plot(model, model_name, dataset_name, reg, test_dl, train_dl)


def visualize_metrics(result, model_name, dataset_name):
    plt.figure(figsize=(5, 3))
    regs = list(result.keys())
    regs.sort()

    for metric in ["mse", "voG", "CN", "VP"]:
        values = []
        stds = []

        values = np.array(values).astype(float)
        stds = np.array(stds).astype(float)

        plt.errorbar(
            np.array(regs).astype(float),
            normalize(values),
            fmt="o--",
            yerr=normalize(stds, rel=values),
            capsize=5,
        )

    plt.xlabel("regularization coefficient", fontsize=12)
    plt.ylabel("normalized metrics", fontsize=12)
    plt.title(f"{model_name} @ {dataset_name}", fontsize=15)
    plt.legend(
        ["mse", "voR", "CN", "VP"], loc="upper left"
    )  # bbox_to_anchor=(1.05, 1),
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1, 2)
    # plt.gca().xaxis.set_ticklabels([])

    plt.savefig(
        get_saving_dir(model_name, dataset_name, f"metrics.png")
    )  # , **get_saving_kwargs()
    # )

    if config["show"]:
        plt.show()
        plt.close()


"""
Determinant Plot
"""


def determinants_plots(dataset_name, model_regs):
    seed = 1

    for model_name, reg in model_regs:
        model, cfg = load_model(model_name, dataset_name, seed, reg)
        test_dl = get_dl(cfg, dataset_name)

        determinants_plot(model, model_name, dataset_name, reg, test_dl)


def determinants_plot(
    model,
    model_name,
    dataset_name,
    reg,
    test_dl,
    quantile=1.0,
    batch_size=-1,
    scaling="asinh",
    grid="dataset",
    num_steps=15,
):
    print(f"[Analyse] determinants of {model_name} on {dataset_name}...")

    if reg == "":
        reg = 0

    data = test_dl.dataset.data

    latent_activations = model.encode(data).detach().cpu()  # .numpy()

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(data.shape[0], generator=generator)
    data = data[perm]
    latent_activations = latent_activations[perm]

    #coordinates = get_coordinates(
    #    torch.squeeze(data.view(data.shape[0], -1)), grid=grid, num_steps=num_steps
    #)

    coordinates = latent_activations

    # initialize diffgeo objects

    # batch-size is negative use the whole batch, i.e. don't batch. Need to batch for storage reasons
    if batch_size == -1:
        batch_size = coordinates.shape[0]

    # calculate determinants
    G = get_pushforwarded_Riemannian_metric(model.encode, data.view(data.shape[0], -1))
    determinants = torch.det(
        G
    ).sqrt()  # TODO: check if this is done along the correct dimension

    # collapse determinants into quantile
    middle_idx = values_in_quantile(determinants, quantile)
    determinants_in_quantile = determinants[middle_idx]
    min_determinant_in_quantile = torch.min(determinants_in_quantile)
    max_determinant_in_quantile = torch.max(determinants_in_quantile)
    determinants[
        determinants < min_determinant_in_quantile
    ] = min_determinant_in_quantile
    determinants[
        determinants > max_determinant_in_quantile
    ] = max_determinant_in_quantile

    # scale determinants
    scaling = "log"
    scaling_fn, prefix = determine_scaling_fn(scaling)
    dets_scaled_raw = scaling_fn(determinants)

    # remove nan scaled determinants
    nonnan_idx = torch.argwhere(~torch.isnan(dets_scaled_raw)).squeeze()
    dets_scaled_raw = dets_scaled_raw[nonnan_idx]
    coordinates = coordinates[nonnan_idx]

    # dets_scaled[torch.isinf(dets_scaled)] = -44

    # change units and shift
    determinants_relative = dets_scaled_raw / torch.abs(torch.mean(dets_scaled_raw))
    # determinants_relative = dets_scaled_raw / torch.mean(dets_scaled_raw)
    dets_scaled = determinants_relative - 1

    # print(dataset_name, torch.min(dets_scaled), torch.max(dets_scaled))

    """
    PLOTTING
    """

    coordinates = coordinates.detach().cpu()
    dets_scaled = dets_scaled.detach().cpu()

    # plot color-coded determinants
    fig, ax = plt.subplots(figsize=((5, 5)))

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.0)
    plt.margins(0.01, 0.01)

    if torch.mean(dets_scaled_raw) >= 0:
        cmap = "turbo"
    else:
        cmap = "turbo"
        dets_scaled += 2

    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=dets_scaled,
        cmap=cmap,
        **get_sc_kwargs(),
        vmin=-1.8,
        vmax=1.22,
    )

    # if model_name == "GeomReg" and dataset_name == "MNIST":
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("bottom", size="5%", pad=0.05)
    # sm = ScalarMappable()
    # sm.set_cmap("turbo")
    # sm.set_array(dets_scaled)  # determinants
    # sm.set_clim(-1.8, 1.22)
    # cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    # cbar.set_alpha(0.5)
    # cbar.draw_all()

    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(
        get_saving_dir(model_name, dataset_name, f"determinants_{float(reg)}.png"),
        **get_saving_kwargs(),
    )

    if config["show"]:
        plt.show()
    
    plt.close()
