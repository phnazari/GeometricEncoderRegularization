import os
import numpy as np
import torch

from loader import get_dataloader
from models import load_pretrained
from utils.config import Config

config = Config()


def save_all_results(model_name, dataset_name):
    root = os.path.join(config["results_path"], f"{dataset_name}_z2/")

    loaded_data = False

    regs = []
    result = {}

    for subdir in os.listdir(os.path.join(root, f"seed1")):
        if subdir.split("_")[0] == model_name:
            reg = subdir.split("_")[-2][3:]
            if reg not in regs:
                regs.append(reg)

                result[reg] = dict()

    regs.sort()

    i = 1
    for j, reg in enumerate(regs):
        seed_results = {}
        seed_results["total_mse_"] = []
        seed_results["train_mse_"] = []
        for k, seed in enumerate(config["seeds"]):
            print(
                f"seed {k+1}/{len(config['seeds'])}, reg {j+1}/{len(regs)}, total {i}/{len(regs) * len(config['seeds'])}"
            )

            model, cfg = load_model(model_name, dataset_name, seed, reg)

            if not loaded_data:
                test_dl = get_dl(cfg, dataset_name)
                all_dl = get_dl(cfg, dataset_name, split="all")
                train_dl = get_dl(cfg, dataset_name, split="train")
                loaded_data = True

            results = model.eval_step(test_dl, device="cpu")
            for metric, value in results.items():
                if metric not in seed_results:
                    seed_results[metric] = []
                seed_results[metric].append(value)

            # compute mse over all data from the all_dl dataloader
            all_out = []
            for x, l in all_dl:
                all_out.append(model(x))
            all_out = torch.cat(all_out)
            total_mse = torch.nn.MSELoss()(all_out, all_dl.dataset.data).item()

            train_out = []
            for x, l in train_dl:
                train_out.append(model(x))
            train_out = torch.cat(train_out)
            train_mse = torch.nn.MSELoss()(train_out, train_dl.dataset.data).item()

            seed_results["total_mse_"].append(total_mse)
            seed_results["train_mse_"].append(train_mse)

            i += 1

        result[reg] = seed_results

    return result


def get_best_model(dataset, results, method="other"):
    model_regs = [("ae", "")]

    tomse = config["type_of_mse"]

    for model in config["models"]:
        if model == "ae":
            continue

        result = results[dataset][model]
        metric = config["METRIC_FOR_MODEL"][model]

        xs, vals, _, _, regs = generate_tradeoff_data(result, metric)

        # normalize xs with respect to vanilla ae
        xs = xs / np.mean(results[dataset]["ae"][""][f"{tomse}_"])

        if method == "mindist":
            graph = np.stack((xs, vals), axis=0)
            graph_norm = np.linalg.norm(graph, axis=0)
            model_regs.append((model, regs[np.argmin(graph_norm)]))
        else:
            # get the last value that is at most 1.1 times the vanilla mse
            idx = np.where(xs <= 10)[0][0]

            model_regs.append((model, regs[idx]))

    return model_regs


def reg_strength_data(model, dataset, metric, results):
    result = results[dataset][model]
    regs = list(result.keys())

    for reg in regs:
        print(result[reg].keys())
        print(metric)
        print("\n")

    vals = np.array([result[reg][f"{metric}_"] for reg in regs])
    std_vals = np.std(vals, axis=1)
    mean_vals = np.mean(vals, axis=1)

    idx = np.argsort(regs)
    regs = np.array(regs)[idx]
    mean_vals = mean_vals[idx]
    std_vals = std_vals[idx]

    return regs, mean_vals, std_vals


def generate_tradeoff_data(result, metric):
    regs = list(result.keys())

    mses = []
    vals = []

    tomse = config["type_of_mse"]

    for reg in regs:
        mses.append(result[reg][f"{tomse}_"])
        vals.append(result[reg][f"{metric}_"])

    std_mses = np.std(np.array(mses), axis=1)
    std_vals = np.std(np.array(vals), axis=1)
    mean_mses = np.mean(np.array(mses), axis=1)
    mean_vals = np.mean(np.array(vals), axis=1)

    idx = np.argsort(mean_vals)
    mean_mses = mean_mses[idx]
    std_mses = std_mses[idx]
    mean_vals = mean_vals[idx]
    std_vals = std_vals[idx]

    return mean_mses, mean_vals, std_mses, std_vals, np.array(regs)[idx]


def load_model(model_name, dataset_name, seed, reg):
    model_name_2 = model_name.split("-")[0]

    if model_name_2 == "ae":
        identifier = f"ae_seed{seed}"
    else:
        identifier = f"{model_name}_reg{reg}_seed{seed}"

    model, cfg = load_pretrained(
        root=os.path.join(config["results_path"], f"{dataset_name}_z2/seed{seed}/"),
        identifier=identifier,
        ckpt_file="model_best.pkl",
        config_file=f"{model_name_2}.yml",
    )

    return model, cfg


def get_dl(cfg, dataset_name, split="test"):
    data_cfg = cfg["data"]
    test_data_cfg = data_cfg["validation"]
    test_data_cfg["split"] = split
    test_data_cfg["path"] = config["data_path"]
    test_data_cfg["root"] = config["data_path"]

    if dataset_name == "earth":
        test_data_cfg["filename"] = os.path.join(
            config["data_path"], "EARTH/landmass.pt"
        )
    elif dataset_name == "polsurf":
        test_data_cfg["filename"] = os.path.join(
            config["data_path"], "POLSURF/surface_data.npy"
        )

    test_dl = get_dataloader(test_data_cfg)

    return test_dl


def normalize(input, rel=None):
    input = np.array(input)
    rel = np.array(rel) if rel is not None else None
    if rel is None:
        input = input - input.min()
        input = input / input.max()
        input += 1
    else:
        input = input / rel.max()
    return input


def values_in_quantile(x, q=0):
    """
    Get alues in q quantile
    """
    if q == 1.0:
        idx = torch.arange(len(x))
    else:
        largest_abs = torch.topk(torch.abs(x), k=int(q * len(x)), largest=True)
        smallest = torch.topk(
            largest_abs.values,
            k=int(len(largest_abs.values) / len(x) * q * len(largest_abs.values)),
            largest=False,
        )

        idx = largest_abs.indices[smallest.indices]

    return idx


def determine_scaling_fn(scaling):
    # determine scaling of curvature values
    scaling_fn = None
    if type(scaling) == str:
        if scaling == "asinh":
            scaling_fn = torch.asinh
        elif scaling == "lin":
            scaling_fn = lambda x: x
        elif scaling == "symlog":
            scaling_fn = symlog
        elif scaling == "log":
            scaling_fn = torch.log10
        else:
            print("TROW CUSTOM ERROR")
    elif callable(scaling):
        scaling_fn = scaling
    else:
        print("THROW CUSTOM ERROR")

    def inverse(x):
        if scaling == "asinh":
            return torch.sinh(x)
        elif scaling == "lin":
            return x
        elif scaling == "symlog":
            return symlog_inv(x)
        elif scaling == "log":
            return torch.pow(10, x)

        return x

    if scaling == "lin":
        prefix = ""
    else:
        prefix = f"{scaling} of "

    return scaling_fn, prefix
