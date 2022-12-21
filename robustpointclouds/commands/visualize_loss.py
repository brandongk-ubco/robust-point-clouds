import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def visualize_loss(lightning_outdir: str):
    sns.set(rc={'figure.figsize': (11, 5.5)})

    lightning_outdir = os.path.abspath(lightning_outdir)
    infile = os.path.join(lightning_outdir, "evaluation_results.csv")
    results = pd.read_csv(infile)

    results = results.drop("Unnamed: 0", axis=1)

    results = results.T.reset_index()
    results["model"] = results.apply(lambda row: row["index"].split("_")[0],
                                     axis=1)
    results["metric"] = results.apply(
        lambda row: "_".join(row["index"].split("_")[1:]), axis=1)

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
        if metric == "perturbation_norm":
            return "Norm"
        if metric == "perturbation_bias":
            return "Bias"
        if metric == "perturbation_imbalance":
            return "Imbalance"
        if metric == "loss_cls":
            return "Classification"
        if metric == "loss_bbox":
            return "Bounding Box"
        if metric == "loss_dir":
            return "Direction"

        if metric == 'direction_norm_mean':
            return "Mean Direction"
        if metric == 'direction_norm_std':
            return "Std Direction"

        if metric == 'intensity_mean':
            return "Mean Intensity"
        if metric == 'intensity_std':
            return "Std Intensity"

        if metric in ["direction_norm_median", "intensity_median"]:
            return None

        raise ValueError("Unknown Metric {}".format(metric))

    unrolled["metric"] = unrolled.apply(metric_mapper, axis=1)

    unrolled = unrolled.dropna()

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
        ["Mean Direction", "Std Direction", "Mean Intensity", "Std Intensity"])]

    perturbations = perturbations.copy()

    perturbations[['Calculation',
                   'Type']] = perturbations.metric.str.split(expand=True)

    def type_mapper(row):
        type = row["Type"]
        if type == "Intensity":
            return "Intensity (%)"
        if type == "Direction":
            return "Position (m)"
        raise ValueError("Unknown Type {}".format(type))

    perturbations["Type"] = perturbations.apply(type_mapper, axis=1)

    plot = sns.boxplot(x="Type",
                       y="value",
                       data=perturbations,
                       hue='Calculation')
    fig = plot.get_figure()
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    outfile = os.path.join(lightning_outdir, "perturbations.png")
    fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
