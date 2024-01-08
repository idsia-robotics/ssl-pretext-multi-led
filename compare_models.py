"""
Reads multiple inference dumps and makes comparison plots
"""
from matplotlib import pyplot as plt
import pandas as pd
from src.config.argument_parser import parse_args
from pathlib import Path

from src.viz.plots import sns_histplot
import numpy as np


def main():
    args = parse_args("comparison", "inference")

    dfs = []
    names = []


    for file in args.dump_files:
        fname = file.stem
        df = pd.read_pickle(file, converters={"pose_add_30_30": lambda x: int(x == "True"),})
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

    summary = df.groupby("model").describe().loc[
        :,
        (("proj_error", "theta_error", "pose_rel_err", "dist_abs_error", "pose_add_30_30"),
         ("mean","std", "50%"))
        ].rename(columns = {"50%" : "median"}, index = str.capitalize).round(2).to_csv()
    
    f = open(out_dir / "summary_stats.csv", "w")
    f.write(summary)
    f.close()




if __name__ == "__main__":
    main()