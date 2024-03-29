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
from src.dataset.leds import LED_TYPES


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
    summary["pose_add_30_30"] *= 100
    
    aucs = df.groupby("model").apply(lambda d: leds_auc(np.stack(d["led_pred"]), np.stack(d["led_true"]))[0])
    
    individual_aucs = df.groupby("model").apply(lambda d: leds_auc(np.stack(d["led_pred"]), np.stack(d["led_true"]), np.stack(d["led_visibility_mask"]))[1])
    raw_aucs = np.stack(individual_aucs.values)
    individual_aucs = pd.DataFrame.from_dict({led_key: raw_aucs[:, i] for i, led_key in enumerate(LED_TYPES)})
    individual_aucs.index = aucs.index

    aucs.name = "AUC"
    individual_aucs.name = "per-led AUC"
    pose_add_10 = pd.DataFrame(summary.loc[:, (("pose_add_30_30"), ("mean"))])

    final = summary.loc[:, (("proj_error", "theta_error", "pose_rel_err", "dist_abs_error"), ("median"))].droplevel(1, 1).join(aucs)
    final = final.join(pose_add_10.droplevel(1,1))

    if "w_led" in dfs[0].attrs:
        lambas = [d.attrs["w_led"] for d in dfs]
    else:
        # lambas = list(map(lambda n: float(n.split("_")[-3]), final.index.values))
        lambas = [0,] * len(dfs)

    if "sample_count" in dfs[0].attrs:
        samples = [d.attrs["sample_count"] for d in dfs]
    else:
        # samples = list(map(lambda n: float(n.split("_")[-1]), final.index.values))
        samples = [1000,] * len(dfs)

    final["theta_error_deg"] = np.rad2deg(final["theta_error"])
    final["lambda"] = lambas
    final["samples"] = samples

    ordered_columns= [
        "lambda",
        "samples",
        "proj_error",
        "theta_error",
        "theta_error_deg",
        "dist_abs_error",
        "pose_rel_err",
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
    f.write(final.round(2).sort_values(by=["model", "samples"], ascending = [False, True]).to_csv())
    f.close()

    aucs_ds = pd.concat([aucs, individual_aucs], axis = 1)
    
    f = open(out_dir / "led_performance.csv", "w")
    f.write((aucs_ds * 100).round(2).sort_values(by=["model"]).to_csv())
    f.close()


if __name__ == "__main__":
    main()