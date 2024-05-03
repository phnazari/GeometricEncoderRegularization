import os
import numpy as np


from experiments.util import get_best_model, save_all_results
from experiments.plots import (
    tradeoff_plot,
    indicatrix_plots,
    visualize_metrics,
    reg_strength_plot,
    determinants_plots,
    latent_plots,
    cn_table
)

from utils.config import Config
from utils.utils import get_output_dir


config = Config()

if __name__ == "__main__":
    results = {}
    for dataset in config["datasets"]:
        results[dataset] = dict()

    #results = {
    #    "mnist": dict(),
    #    #"earth": dict(),
    #    #"celegans": dict(),
    #    #"zilionis": dict(),
    #    #"pbmc": dict(),
    #}
    #  "confae-log-inside", "confae-noapprox"]  # "confae-log-inside", "confae-noapprox"]  # , "confae"] # ae
    load = True
    mode = "cn_table"  # tradeoff, reg, indicatrix, detplot, latents, cn_table

    # load results
    if load:
        results = np.load(
            os.path.join(get_output_dir(), "results.npy"), allow_pickle=True
        ).item()
    else:
        #results = np.load(
        #    os.path.join(get_output_dir(), "results.npy"), allow_pickle=True
        #).item()
        #for dataset in config["datasets"]:
        #    results[dataset] = dict()

        i = 0
        for dataset in config["datasets"]:
            # results[dataset] = dict()
            for model in config["models"]:
                print(
                    f"EVALUATING MODEL {model} ON DATASET {dataset} ({i} of {len(config['models']) * len(config['datasets'])})"
                )
                results[dataset][model] = save_all_results(
                    model, dataset
                )

                i += 1

        # save results to file
        np.save(os.path.join(get_output_dir(), "results.npy"), results)

    # tradeoff plots
    if mode == "tradeoff":
        for dataset in config["datasets"]:
            for metric in config["metrics"]:
                tradeoff_plot(dataset, metric, results)

    elif mode == "reg":
        for dataset in config["datasets"]:
            for metric in config["metrics"]:
                reg_strength_plot(dataset, metric, results)

    # indicatrix plots
    elif mode == "indicatrix":
        for dataset in config["datasets"]:
            indicatrix_plots(dataset, get_best_model(dataset, results))

    elif mode == "detplot":
        for dataset in config["datasets"]:
            # for model in config["models"]:
            determinants_plots(dataset, get_best_model(dataset, results))

    elif mode == "latents":
        for dataset in config["datasets"]:
            # for model in config["models"]:
            latent_plots(dataset, get_best_model(dataset, results))

    elif mode == "cn_table":
        for dataset in config["datasets"]:
            cn_table(dataset, get_best_model(dataset, results))
