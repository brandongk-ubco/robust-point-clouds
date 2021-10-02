import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def visualize_loss(lightning_outdir: str):
    sns.set(rc={'figure.figsize': (11, 2.5)})

    lightning_outdir = os.path.abspath(lightning_outdir)
    infile = os.path.join(lightning_outdir, "evaluation_results.csv")
    results = pd.read_csv(infile)

    results = results.drop("Unnamed: 0", axis=1)

    results = results.T.reset_index()
    results["model"] = results.apply(lambda row: row["index"].split("_")[0],
                                     axis=1)
    results["metric"] = results.apply(lambda row: row["index"].split("_")[-1],
                                      axis=1)

    results = results.drop("index", axis=1)
    results = results.set_index(["model", "metric"])

    combinations = results.index.to_flat_index()

    unrolled = pd.DataFrame(columns=["model", "metric", "value"])
    for combination in combinations:
        unrolled_values = results.T[combination]
        u = pd.DataFrame(columns=["model", "metric", "value"])
        u["value"] = unrolled_values
        u["model"] = combination[0]
        u["metric"] = combination[1]
        unrolled = unrolled.append(u, ignore_index=True)

    def metric_mapper(row):
        metric = row["metric"]
        if metric == "norm":
            return "L2 Norm"
        if metric == "bias":
            return "Bias"
        if metric == "imbalance":
            return "Imbalance"
        if metric == "cls":
            return "Classification"
        if metric == "bbox":
            return "Bounding Box"
        if metric == "dir":
            return "Direction"

    unrolled["metric"] = unrolled.apply(metric_mapper, axis=1)

    losses = unrolled[unrolled["metric"].isin(
        ["Classification", "Bounding Box", "Direction"])]

    losses = losses.rename(columns={
        "metric": "Loss Type",
        "value": "Loss Value"
    })

    plot = sns.boxplot(x="Loss Type", y="Loss Value", data=losses, hue='model')
    fig = plot.get_figure()
    plt.title("")
    outfile = os.path.join(lightning_outdir, "loss.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    perturbations = unrolled[unrolled["metric"].isin(
        ["L2 Norm", "Bias", "Imbalance"])]

    plot = sns.boxplot(x="metric", y="value", data=perturbations)
    fig = plot.get_figure()
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    outfile = os.path.join(lightning_outdir, "perturbations.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
