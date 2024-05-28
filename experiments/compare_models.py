import json
import math
import os
import sys
import numpy as np
import pandas as pd
import scipy
import torch

from utils.utils import round_significant, get_output_dir
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
    #("voR", "$\VoR$"),
    #("VP", "$\VP$"),
    #("CN", "$\CN$"),
]

with_std = True

datasets = config["datasets"]

models = [
    ("irae", "IRAE"),
    ("geomae", "GeomAE"),
    ("confae-log", "ConfAE"),
    ("ae", "VanillaAE"),
]

metrics = [metric + "_" for metric, _ in used_metrics]

compared_tables = np.empty((len(datasets), len(models), len(metrics)), dtype=float)

# metrics_dir = "metrics1.25"

results_encoder = np.load(os.path.join(get_output_dir(raw=True), "encoder", "results.npy"), allow_pickle=True).item()
results_decoder = np.load(os.path.join(get_output_dir(raw=True), "decoder", "results.npy"), allow_pickle=True).item()


for dataset in datasets:
    # Load the dictionary from the file
    # results_dict = torch.load(os.path.join(config["results_path"], f"{dataset}/{metrics_dir}/result.pth"))

    best_models_encoder = get_best_model(dataset, results_encoder)
    best_models_decoder = get_best_model(dataset, results_decoder)
    best_models_encoder = dict(best_models_encoder)
    best_models_decoder = dict(best_models_decoder)

    # Create an empty table
    compared_table = np.empty((len(models), len(metrics)), dtype=float)

    # Iterate through the models and metrics, calculating the mean and standard deviation for each metric for each model
    for i, model in enumerate(models):
        model, _ = model

        for j, metric in enumerate(metrics):
            metric_values_encoder = results_encoder[dataset][model][best_models_encoder[model]][metric]
            metric_values_decoder = results_decoder[dataset][model][best_models_decoder[model]][metric]

            mean_encoder = np.mean(metric_values_encoder)
            mean_decoder = np.mean(metric_values_decoder)
            std_encoder = np.std(metric_values_encoder)
            std_decoder = np.std(metric_values_decoder)

            # compare if mean_encoder or mean_decoder is better and write into the compared_table (1 for encoder, 0 for decoder)
            if larger_is_better[metric]:
                if mean_encoder > mean_decoder:
                    compared_table[i, j] = 1.
                else:
                    compared_table[i, j] = 0.
            else:
                if mean_encoder > mean_decoder:
                    compared_table[i, j] = 0.
                else:
                    compared_table[i, j] = 1.

    compared_tables[datasets.index(dataset)] = compared_table

    collabels = np.array([metric for _, metric in used_metrics])
    rowlabels = np.array([model for _, model in models])

    # Convert table to dataframe
    #df = pd.DataFrame(final_table, columns=collabels, index=rowlabels)

    # Save dataframe to latex file
    #with open(
    #    os.path.join(get_output_dir(), "latex", f"table_{dataset}.tex"), "w"
    #) as f:
    #    f.write(df.to_latex(index=True))

# calculate mean of ranked_tables over datasets
# mean_ranked_tables = np.around(np.mean(ranked_tables, axis=0, dtype=float), decimals=1).astype(object)
print(compared_tables)
mode_compared_tables_raw = scipy.stats.mode(compared_tables, axis=0)

mode_compared_tables = mode_compared_tables_raw.mode[0]
mode_compared_tables_count = mode_compared_tables_raw.count[0]
mode_compared_tables[mode_compared_tables_count == len(datasets) / 2] = 3.0

#mode_local_rank = np.mean(mode_compared_tables[:, 0:3], axis=1, dtype=float)
#mode_global_rank = np.mean(mode_compared_tables[:, 3:6], axis=1, dtype=float)
#mode_compared_tables = np.insert(mode_compared_tables, 3, mode_local_rank, axis=1)
#mode_compared_tables = np.insert(mode_compared_tables, 7, mode_global_rank, axis=1)

#collabels = np.insert(collabels, 3, "Local", axis=0)
#collabels = np.insert(collabels, 7, "Global", axis=0)

# mode_compared_tables = np.around(mode_compared_tables, decimals=1).astype(object)

#for i, col in enumerate(mode_compared_tables.T):
#    ordered_col = np.argsort(col)
#    ranked_col = ordered_col.argsort().astype("float") + 1

#    best_index = ordered_col[0]
#    second_best_index = ordered_col[1]

#    col[best_index] = f"\\underline{{\\textbf{{{col[best_index]}}}}}"
#    col[second_best_index] = f"\\textbf{{{col[second_best_index]}}}"

#    mean_ranked_tables[:, i] = col

mode_compared_tables = mode_compared_tables.astype(str)
# replace numeric values by encoder, decoder and draw
mode_compared_tables[mode_compared_tables == "1.0"] = "Encoder"
mode_compared_tables[mode_compared_tables == "0.0"] = "Decoder"
mode_compared_tables[mode_compared_tables == "3.0"] = "Draw"


# Convert table to dataframe
df = pd.DataFrame(mode_compared_tables.astype(str), columns=collabels, index=rowlabels)

# Save dataframe to latex file
with open(os.path.join(get_output_dir(raw=True), "latex", "table_compared.tex"), "w") as f:
    f.write(df.to_latex(index=True))
