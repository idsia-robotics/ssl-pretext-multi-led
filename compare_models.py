"""
Reads multiple inference dumps and makes comparison plots
"""
from matplotlib import pyplot as plt
import pandas as pd
from src.config.argument_parser import parse_args
from pathlib import Path
from src.metrics import leds_auc

from src.viz.plots import sns_histplot
import numpy as np


def main():
    args = parse_args("comparison", "inference")

    dfs = []
    names = []


    for file in args.dump_files:
        fname = file.stem
        df = pd.read_pickle(file)
        dfs.append(df)
        names.append(fname)
    
    
    df = pd.concat(dfs, keys = names)
    df.index.names = ["model", "id"]

    out_dir = args.out_dir
    out_dir.mkdir(parents = True, exist_ok = True)

    fig, ax = plt.subplots(1,1)
    p = sns_histplot(df, 'proj_error', "model", ax = ax)
    p.figure.savefig(out_dir / "proj_error.png")

    
    fig, ax = plt.subplots(1,1)
    p = sns_histplot(df, 'theta_error', "model", ax = ax)
    p.figure.savefig(out_dir / "theta_error.png")

    fig, ax = plt.subplots(1,1)
    p = sns_histplot(df, 'pose_rel_err', "model", ax = ax)
    p.figure.savefig(out_dir / "pose_rel_error.png")

    fig, ax = plt.subplots(1,1)
    p = sns_histplot(df, 'dist_abs_error', "model", ax = ax)
    p.figure.savefig(out_dir / "dist_error.png")


    df["pose_add_30_30"] = df["pose_add_30_30"].astype(np.float32)
    summary = df.groupby("model").describe().loc[
        :,
        (("proj_error", "theta_error", "pose_rel_err", "dist_abs_error", "pose_add_30_30"),
         ("mean","std", "50%"))
        ].rename(columns = {"50%" : "median"})
    
    aucs = df.groupby("model").apply(lambda d: leds_auc(np.stack(d["led_pred"]), np.stack(d["led_true"]))[0])
    aucs.name = "AUC"
    pose_add_10 = pd.DataFrame(summary.loc[:, (("pose_add_30_30"), ("mean"))])

    final = summary.loc[:, (("proj_error", "theta_error", "pose_rel_err", "dist_abs_error"), ("median"))].droplevel(1, 1).join(aucs)
    final = final.join(pose_add_10.droplevel(1,1))

    lambas = list(map(lambda n: float(n.split("_")[3]), final.index.values))
    samples = list(map(lambda n: float(n.split("_")[-1]), final.index.values))
    final["theta_error_deg"] = np.rad2deg(final["theta_error"])
    final["lambda"] = lambas
    final["samples"] = samples

    ordered_columns= [
        "lambda",
        "samples",
        "proj_error",
        "theta_error",
        "theta_error_deg",
        "pose_rel_err",
        "dist_abs_error",
        "pose_add_30_30",
        "AUC",
    ]
    final = final[ordered_columns]

    final = final.rename(columns={
        "proj_error" : "UV Median Error",
        "theta_error" : "Theta Absolute Median Error (rad)",
        "theta_error_deg" : "Theta Absolute Median Error (deg)",
        "pose_rel_err" : "Relative Pose Median Error",
        "dist_abs_error" : "Relative Distance Median Absolute Error",
        "pose_add_30_30" : "Pose ADD (30,30)"
    })


    f = open(out_dir / "summary_stats.csv", "w")
    f.write(summary.round(2).to_csv())
    f.close()

    f = open(out_dir / "summary_stats_small.csv", "w")
    f.write(final.round(2).sort_values(by="lambda").to_csv())
    f.close()


if __name__ == "__main__":
    main()