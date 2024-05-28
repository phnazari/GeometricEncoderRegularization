import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd


from experiments.util import generate_tradeoff_data, get_dl, normalize
from utils.utils import random_metric_field_generator, get_output_dir, round_significant

from loader.MNIST_dataset import MNIST
from loader.EARTH_dataset import EARTH
from loader.POLSURF_dataset import POLSURF
from loader.ZILIONIS_dataset import ZILIONIS
from loader.CELEGANS_dataset import CELEGANS
from loader.PBMC_dataset import PBMC

from models import load_pretrained
from geometry import get_flattening_scores, get_Riemannian_metric

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
    get_nearest_grid_points
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

    with_legend = True
    if not with_legend:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        if dataset_name == "mnist":
            fig, ax = plt.subplots(figsize=(30, 5))  # 10
        elif dataset_name == "pbmc":
            fig, ax = plt.subplots(figsize=(30, 5))  # 17
        elif dataset_name == "zilionis":
            fig, ax = plt.subplots(figsize=(30, 5))  # 15
        elif dataset_name == "celegans":
            fig, ax = plt.subplots(figsize=(30, 5))  # 29
        else:
            fig, ax = plt.subplots(figsize=(30, 5))  # 10

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.)
    plt.margins(0.01, 0.01)

    c = targets.detach().cpu()

    scatter= ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=c,
        **get_sc_kwargs(),
        zorder=0,
    )

    if with_legend:
        if dataset_name == "zilionis":
            handles, _ = scatter.legend_elements(num=None)

            chartBox = ax.get_position()
            ax.set_position([0, 0, chartBox.width, chartBox.height])

            string_labels = ZILIONIS().transform_labels(os.path.join(config["data_path"], "ZILIONIS"))
            legend = ax.legend(handles, string_labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=2, fontsize=22,
                               markerscale=2)
        elif dataset_name == "pbmc":
            handles, _ = scatter.legend_elements(num=None)

            chartBox = ax.get_position()
            ax.set_position([0, 0, chartBox.width, chartBox.height])

            string_labels = PBMC().transform_labels(os.path.join(config["data_path"], "PBMC"))

            legend = ax.legend(handles, string_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22,
                               markerscale=2, ncol=2)
        elif dataset_name == "celegans":
            handles, _ = scatter.legend_elements(num=None)

            string_labels = CELEGANS().transform_labels(os.path.join(config["data_path"], "CELEGANS"))

            chartBox = ax.get_position()
            ax.set_position([0, 0, chartBox.width, chartBox.height])

            legend = ax.legend(handles, string_labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=4, fontsize=22,
                               markerscale=2)
            scatter.legend_elements(prop="colors", num=37)
        else:
            chartBox = ax.get_position()
            ax.set_position([0, 0, chartBox.width, chartBox.height])

            legend = ax.legend(*scatter.legend_elements(num=None), loc="center left", bbox_to_anchor=(1.4, 0.5), ncol=2,
                               fontsize=22, markerscale=2)

        for lh in legend.legendHandles:
            lh.set_alpha(1)

    # add legend next to plot

    # plt.title(f"{model_name} @ {dataset_name} (reg={float(reg)})")
    plt.savefig(
        get_saving_dir(model_name, dataset_name, f"latents.png"),  # reg{float(reg)}
        **get_saving_kwargs(),
    )

    if config["show"]:
        plt.show()


def latent_plots(dataset_name, model_regs):
    seed = 3

    for model_name, reg in model_regs:
        model, cfg = load_model(model_name, dataset_name, seed, reg)
        train_dl = get_dl(cfg, dataset_name, split="train")
        test_dl = get_dl(cfg, dataset_name)

        print(dataset_name, model_name)

        latent_plot(model, model_name, dataset_name, reg, test_dl, train_dl)


def cn_table(dataset_name, model_regs):

    # TODO: 
    # average over all three seeds
    # create plot for all datasets

    result = {}
    for name in config["models"]:
        result[name] = dict()

    for seed in config["seeds"]:
        for model_name, reg in model_regs:
            if model_name == "ae":
                continue

            if seed == 1:
                result[model_name] = {"encoder": {"encoder": torch.tensor([]), "decoder": torch.tensor([])},
                                    "decoder": {"encoder": torch.tensor([]), "decoder": torch.tensor([])}}

            for reg_part in ["encoder", "decoder"]:
                raw_model, cfg = load_model(model_name, dataset_name, seed, reg, reg_part=reg_part)
                train_dl = get_dl(cfg, dataset_name, split="train")
                test_dl = get_dl(cfg, dataset_name)

                data = torch.cat((test_dl.dataset.data, train_dl.dataset.data))
                # targets = torch.cat((test_dl.dataset.targets, train_dl.dataset.targets))
                Z = raw_model.encode(data).detach().cpu()

                for vis_part in ["encoder", "decoder"]:
                    coordinate_idx = get_nearest_grid_points(Z, num_steps=64)
                    latent_coordinates = Z[coordinate_idx].to(config["device"])
                    data_coordinates = data[coordinate_idx].to(config["device"])

                    #if 'points' in locals():
                    #    print(points.unique())
                        # points_old = points.clone()

                    if vis_part == "encoder":
                        model = raw_model.encode
                        points = data_coordinates
                    elif vis_part == "decoder":
                        model = raw_model.decode
                        points = latent_coordinates

                    # if 'points' in locals():
                    #     print(points.unique())
                    #     print("\n")
                    #     print(points != points_old)
                    #     print(torch.sum(points_old != points))

                    G = get_Riemannian_metric(model, points.view(points.shape[0], -1), "vis", purpose_part=vis_part)

                    #if vis_part == "encoder" and reg_part == "decoder" and dataset_name == "mnist" and model_name == "geomae":
                        #for name, param in raw_model.named_parameters():
                        #    if param.requires_grad:
                        #        print(name, param.data)
                        # print(cfg)
                        # print(seed, model_name, reg)
                        # print(points.unique())
                        # print(G)
                        # print("\n")

                    # G = (G) / torch.std(G.view(len(G), 4), dim=1)[:, None, None]  #  - torch.mean(G.view(len(G), 4), dim=1)[:, None, None]

                    """
                    if model_name == "confae-log":
                        metric_mode = "condition_number"
                    elif model_name == "geomae":
                        metric_mode = "volume_preserving"
                    elif model_name == "irae":
                        metric_mode = "variance"

                    metric = get_flattening_scores(G, mode=metric_mode)
                    """

                    if model_name == "confae-log":
                        S = torch.svd(G).S
                        metric_inp = torch.abs(1 - S.min(1).values/S.max(1).values)  # inverted condition number -1
                        metric_inp = metric_inp[values_in_quantile(metric_inp, 0.90)]
                        metric = torch.mean(metric_inp)
                    elif model_name == "geomae":
                        Slog = torch.log(torch.svd(G).S)
                        metric_inp = Slog.sum(1)
                        metric_inp = metric_inp[values_in_quantile(metric_inp, 0.90)]
                        metric = torch.std(metric_inp)
                    elif model_name == "irae":
                        S = torch.svd(G).S
                        metric_inp = torch.log(S).flatten()
                        metric_inp = metric_inp[values_in_quantile(metric_inp, 0.90)]
                        metric = torch.std(metric_inp)

                    # metric = metric[values_in_quantile(metric, 0.90)]

                    tmp = result[model_name][reg_part][vis_part]
                    result[model_name][reg_part][vis_part] = torch.cat((tmp, metric.reshape(1)))
                    # result[model_name][reg_part][vis_part].append(metric)

        print(f"seed {seed} done.")

    # average over seeds
    mean_result = {}
    std_result = {}
    for name in config["models"]:
        mean_result[name] = dict()
        std_result[name] = dict()

    for model_name, _ in model_regs:
        if model_name == "ae":
            continue
        mean_result[model_name] = {"encoder": {"encoder": [], "decoder": []},
                                "decoder": {"encoder": [], "decoder": []}}
        std_result[model_name] = {"encoder": {"encoder": [], "decoder": []},
                                "decoder": {"encoder": [], "decoder": []}}

        for reg_part in ["encoder", "decoder"]:
            for vis_part in ["encoder", "decoder"]:
                #mean_result[model_name][reg_part][vis_part] = torch.mean(result[model_name][reg_part][vis_part]).item()
                #std_result[model_name][reg_part][vis_part] = torch.std(result[model_name][reg_part][vis_part]).item()
                mean_result[model_name][reg_part][vis_part] = torch.mean(torch.tensor(result[model_name][reg_part][vis_part])).item()
                std_result[model_name][reg_part][vis_part] = torch.std(torch.tensor(result[model_name][reg_part][vis_part])).item()

    print(mean_result)
    print(std_result)

    table = [[],[]]
    
    for mn in ["geomae", "confae-log", "irae"]:
        for part in ["encoder", "decoder"]:
            table[0].append(round_significant(
                [mean_result[mn][part]["encoder"]],
                [std_result[mn][part]["encoder"]]
            )[0])
            table[1].append(round_significant(
                [mean_result[mn][part]["decoder"]],
                [std_result[mn][part]["decoder"]
            ])[0])

    print(np.array(table).shape)

    """
    table = [
        [
            mean_result["geomae"]["encoder"]["encoder"],
            mean_result["geomae"]["decoder"]["encoder"],
            mean_result["confae-log"]["encoder"]["encoder"],
            mean_result["confae-log"]["decoder"]["encoder"],
            mean_result["irae"]["encoder"]["encoder"],
            mean_result["irae"]["decoder"]["encoder"]
        ],
        [
            mean_result["geomae"]["encoder"]["decoder"],
            mean_result["geomae"]["decoder"]["decoder"],
            mean_result["confae-log"]["encoder"]["decoder"],
            mean_result["confae-log"]["decoder"]["decoder"],
            mean_result["irae"]["encoder"]["decoder"],
            mean_result["irae"]["decoder"]["decoder"] 
        ]
    ]
    """

    rowlabels = np.array(["encoder", "decoder"])
    collabels = np.array(["encoder", "decoder", "encoder", "decoder", "encoder", "decoder"])

    # Convert table to dataframe
    df = pd.DataFrame(table, columns=collabels, index=rowlabels)

    # Save dataframe to latex file
    with open(
        os.path.join(get_output_dir(raw=True), f"latex/comparison/table_{dataset_name}.tex"), "w"
    ) as f:
        f.write(df.to_latex(index=True))

    # print(f"dataset: {dataset_name}, cond_table: {result}")
    print(f"DONE: {dataset_name}")


def indicatrix_plot(model, model_name, dataset_name, reg, test_dl, train_dl):
    # TODO: extract model_name and datase_name from model and test_dl
    data = torch.cat((test_dl.dataset.data, train_dl.dataset.data))
    targets = torch.cat((test_dl.dataset.targets, train_dl.dataset.targets))
    Z = model.encode(data).detach().cpu()  # .numpy()

    if reg == "":
        reg = 0

    # calculate grid step sizes
    x_min = torch.min(Z[:, 0]).item()
    x_max = torch.max(Z[:, 0]).item()
    y_min = torch.min(Z[:, 1]).item()
    y_max = torch.max(Z[:, 1]).item()

    num_steps_x = 8
    num_steps_y = int((y_max - y_min) / (x_max - x_min) * num_steps_x)

    step_size_x = (x_max - x_min) / (num_steps_x)
    step_size_y = (y_max - y_min) / (num_steps_y)
    stepsize = min(step_size_x, step_size_y)

    coordinate_idx = get_nearest_grid_points(Z, num_steps=num_steps_x)

    latent_coordinates = Z[coordinate_idx].to(config["device"])
    data_coordinates = data[coordinate_idx].to(config["device"])

    # Z_pinned_data = model.encode(pinned_data)
    # G = get_pushforwarded_Riemannian_metric(model.encode, data_coordinates.view(data_coordinates.shape[0], -1))
    
    if config["part_of_ae"]["vis"] == "encoder":
        model = model.encode
        points = data_coordinates
    elif config["part_of_ae"]["vis"] == "decoder":
        model = model.decode
        points = latent_coordinates

    G = get_Riemannian_metric(model, points.view(points.shape[0], -1), "vis")
    if config["part_of_ae"]["vis"] == "encoder":
        G = torch.inverse(G)
    # print(G.shape)

    vector_patches, _ = generate_unit_vectors(100, points, G)
    vector_norms = torch.linalg.norm(vector_patches, dim=2)
    vector_norms_maxs = torch.max(vector_norms, dim=1).values
    # max_vector_norm = torch.max(vector_norms[torch.nonzero(vector_norms)])
    # scale in such a way that we ignore outliers

    # get 90% quantile of vector norms maxs
    max_vector_norm = torch.quantile(vector_norms_maxs, 0.9).item()

    #top_vector_norms, _ = torch.topk(vector_norms_maxs, 3)
    #max_vector_norm = top_vector_norms[-1].item()
    # max_vector_norm = torch.mean(vector_norms).item()
    # max_vector_norm = torch.topk(vector_norms, 4).values[3]

    #if model_name == "irae":
    #    if dataset_name == "mnist":
    #        max_vector_norm *= 1/3

    normed_vector_patches = (
        vector_patches / max_vector_norm * stepsize / 2
    )  # * scaling_factor  # / 3
    anchored_vector_patches = (
        latent_coordinates.unsqueeze(1).expand(*normed_vector_patches.shape)
        + normed_vector_patches
    )

    # create polygons
    polygons = PatchCollection(
        [Polygon(tuple(vector.tolist()), True) for vector in anchored_vector_patches]
    )

    if config["part_of_ae"]["vis"] == "decoder":
        polygons.set_color([12 / 255, 0 / 255, 55 / 255, 0.3])  # decoder: blue
    elif config["part_of_ae"]["vis"] == "encoder":
        polygons.set_color([55 / 255, 0 / 255, 17 / 255, 0.3])  # encoder: red
    else:
        raise NotImplementedError

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

    # set axis limits s.t. the scatter plot is visible and a small padding
    #ax.set_xlim(x_min, x_max)
    #ax.set_ylim(y_min, y_max)

    # plt.title(f"{model_name} @ {dataset_name} (reg={float(reg)})")
    plt.savefig(
        get_saving_dir(model_name, dataset_name, f"indicatrix.png"),
        **get_saving_kwargs(),
    )

    if config["show"]:
        plt.show()


def indicatrix_plots(dataset_name, model_regs):
    seed = 3

    for model_name, reg in model_regs:
        model, cfg = load_model(model_name, dataset_name, seed, reg)
        train_dl = get_dl(cfg, dataset_name, split="train")
        test_dl = get_dl(cfg, dataset_name)

        print(dataset_name, model_name, reg)

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
    seed = 3

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
    print(f"[Analyze] determinants of {model_name} on {dataset_name}...")

    if reg == "":
        reg = 0

    data = test_dl.dataset.data

    latent_activations = model.encode(data).detach().cpu()  # .numpy()

    generator = torch.Generator().manual_seed(0)
    perm = torch.randperm(data.shape[0], generator=generator)

    data = data[perm]  # [:num_data]
    latent_activations = latent_activations[perm]  # [:num_data]

    # batch-size is negative use the whole batch, i.e. don't batch. Need to batch for storage reasons
    if batch_size == -1:
        batch_size = latent_activations.shape[0]

    if config["part_of_ae"]["vis"] == "encoder":
        model = model.encode
        points = data
    elif config["part_of_ae"]["vis"] == "decoder":
        model = model.decode
        points = latent_activations

    #G = []
    #num_data = 100
    #for i in range(0, points.shape[0], num_data):
    #    print(i)
    #    batch = points[i:i+num_data]
    #    G_new = get_Riemannian_metric(model, batch.view(batch.shape[0], -1), "vis")
    #    G.append(G_new)
    #G = torch.cat(G)

    G = get_Riemannian_metric(model, points.view(points.shape[0], -1), "vis")

    # calculate determinants
    # G = get_pushforwarded_Riemannian_metric(model.encode, data.view(data.shape[0], -1))
    
    determinants = torch.det(
        G
    ).sqrt()

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
    latent_activations = latent_activations[nonnan_idx]

    # dets_scaled[torch.isinf(dets_scaled)] = -44

    # change units and shift
    determinants_relative = dets_scaled_raw / torch.abs(torch.mean(dets_scaled_raw))
    # determinants_relative = dets_scaled_raw / torch.mean(dets_scaled_raw)
    dets_scaled = determinants_relative - 1

    # print(dataset_name, torch.min(dets_scaled), torch.max(dets_scaled))

    """
    PLOTTING
    """

    latent_activations = latent_activations.detach().cpu()
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
        latent_activations[:, 0],
        latent_activations[:, 1],
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
        get_saving_dir(model_name, dataset_name, f"determinants.png"),  # _{float(reg)}
        **get_saving_kwargs(),
    )

    if config["show"]:
        plt.show()
    
    plt.close()
