import json
import math
import os
import sys
import numpy as np
import pandas as pd
import torch

from utils.utils import round_significant
from utils.config import Config

from experiments.util import get_best_model

# sys.path.append("../")

config = Config()


# Define the metrics to use
# TODO: then, also decide which model to pick for a given dataset. Not that trivial.

larger_is_better = {
    "density_kl_global_100_": 0,
    "density_kl_global_01_": 0,
    "mean_trustworthiness_": 1,
    "rmse_": 0,
    "mean_knn_recall_": 1,
    "spearman_metric_": 1,
    "stress_": 0,
    "CN_": 0,
    "VP_": 0,
    "voR_": 0,
    "mse_": 0,
    "train_mse_": 0
}

used_metrics = [
    ("density_kl_global_01", "$\KL_{01}$"),
    ("mean_knn_recall", "kNN"),
    ("mean_trustworthiness", "TRUST"),
    ("stress", "STRESS"),
    ("density_kl_global_100", "$\KL_{100}$"),
    ("spearman_metric", "SPEAR"),
    ("mse", "MSE (test)"),
    ("train_mse", "MSE (train)"),
    ("voR", "$\VoR$"),
    ("VP", "$\VP$"),
    ("CN", "$\CN$"),
]

with_std = True

datasets = ["earth", "mnist", "celegans", "zilionis", "pbmc"]

models = [
    ("irae", "IsoAE"),
    ("geomae", "GeomAE"),
    ("confae-log", "ConfAE"),
    ("ae", "VanillaAE"),
]

metrics = [metric + "_" for metric, _ in used_metrics]

ranked_tables = np.empty((len(datasets), len(models), len(metrics)), dtype=float)

# metrics_dir = "metrics1.25"

results = np.load(os.path.join(config["output_path"], "results.npy"), allow_pickle=True).item()

for dataset in datasets:
    # Load the dictionary from the file
    # results_dict = torch.load(os.path.join(config["results_path"], f"{dataset}/{metrics_dir}/result.pth"))

    best_models = get_best_model(dataset, results)
    best_models = dict(best_models)

    # Create an empty table
    final_table = np.empty((len(models), len(metrics)), dtype=object)
    mean_table = np.empty((len(models), len(metrics)), dtype=float)
    std_table = np.empty((len(models), len(metrics)), dtype=float)
    ranked_table = np.empty((len(models), len(metrics)), dtype=float)

    # Iterate through the models and metrics, calculating the mean and standard deviation for each metric for each model
    for i, model in enumerate(models):
        model, _ = model

        for j, metric in enumerate(metrics):
            metric_values = results[dataset][model][best_models[model]][metric]

            mean = np.mean(metric_values)
            std = np.std(metric_values)

            mean_table[i, j] = mean
            std_table[i, j] = std

    for i, col in enumerate(mean_table.T):
        if larger_is_better[metrics[i]]:
            ordered_col = np.flip(np.argsort(col))
        else:
            ordered_col = np.argsort(col)
        ranked_col = ordered_col.argsort().astype("float") + 1

        best_index = ordered_col[0]
        second_best_index = ordered_col[1]

        if with_std:
            output = round_significant(mean_table[:, i], std_table[:, i])
        else:
            output = mean_table[:, i]

        output[best_index] = f"\\underline{{\\textbf{{{output[best_index]}}}}}"
        output[second_best_index] = f"\\textbf{{{output[second_best_index]}}}"

        final_table[:, i] = output
        ranked_table[:, i] = ranked_col

    ranked_tables[datasets.index(dataset)] = ranked_table

    collabels = np.array([metric for _, metric in used_metrics])
    rowlabels = np.array([model for _, model in models])

    # Convert table to dataframe
    df = pd.DataFrame(final_table, columns=collabels, index=rowlabels)

    # Save dataframe to latex file
    with open(
        os.path.join(config["output_path"], f"latex/table_{dataset}.tex"), "w"
    ) as f:
        f.write(df.to_latex(index=True))

# calculate mean of ranked_tables over datasets
# mean_ranked_tables = np.around(np.mean(ranked_tables, axis=0, dtype=float), decimals=1).astype(object)
mean_ranked_tables = np.mean(ranked_tables, axis=0, dtype=float)


mean_local_rank = np.mean(mean_ranked_tables[:, 0:3], axis=1, dtype=float)
mean_global_rank = np.mean(mean_ranked_tables[:, 3:6], axis=1, dtype=float)

mean_ranked_tables = np.insert(mean_ranked_tables, 3, mean_local_rank, axis=1)
mean_ranked_tables = np.insert(mean_ranked_tables, 7, mean_global_rank, axis=1)

collabels = np.insert(collabels, 3, "Local", axis=0)
collabels = np.insert(collabels, 7, "Global", axis=0)

mean_ranked_tables = np.around(mean_ranked_tables, decimals=1).astype(object)

for i, col in enumerate(mean_ranked_tables.T):
    ordered_col = np.argsort(col)
    ranked_col = ordered_col.argsort().astype("float") + 1

    best_index = ordered_col[0]
    second_best_index = ordered_col[1]

    col[best_index] = f"\\underline{{\\textbf{{{col[best_index]}}}}}"
    col[second_best_index] = f"\\textbf{{{col[second_best_index]}}}"

    mean_ranked_tables[:, i] = col

# Convert table to dataframe
df = pd.DataFrame(mean_ranked_tables.astype(str), columns=collabels, index=rowlabels)

print(df)

# Save dataframe to latex file
with open(os.path.join(config["output_path"], f"latex/table_aggregated.tex"), "w") as f:
    f.write(df.to_latex(index=True))
