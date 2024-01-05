from matplotlib import pyplot as plt
import numpy as np
from itertools import product
from src.metrics import angle_difference
import seaborn as sns
from itertools import product
import pandas as pd
from matplotlib import colormaps as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def theta_scatter_plot(ds):

    theta_true = ds["theta_true"]
    theta_pred = ds["theta_pred"]
    fig, ax = plt.subplots(1,1)
    theta_to_deg = np.rad2deg(theta_pred)
    true_theta_to_deg = np.rad2deg(theta_true)

    # theta_to_deg[theta_to_deg < 0] = 360 + theta_to_deg[theta_to_deg < 0]
    # true_theta_to_deg[true_theta_to_deg < 0] = 360 + true_theta_to_deg[true_theta_to_deg < 0]

    # ax.scatter(theta_true, true_theta_to_deg, color = 'blue', label='Ideal')
    # ax.scatter(theta_true, theta_to_deg, color = 'red', label="Model")
    for k1, k2, in product([-1, 0, 1], [-1, 0, 1]):
        ax.scatter(theta_true + k1 * 2 * np.pi, theta_pred + k2 * 2 * np.pi, color = 'blue')
    ax.set_xlabel("True [rad]")
    ax.set_ylabel("Predicted [rad]")
    ax.set_xlim([-np.pi * 1.1, np.pi * 1.1])
    ax.set_ylim([-np.pi * 1.1, np.pi * 1.1])
    ax.set_title("True relative orientation vs predicted")
    fig.legend()
    return fig

def proj_scatter_plot(ds):
    fig, ax = plt.subplots(1,2)

    proj_true = ds["proj_true"]
    proj_pred = ds["proj_pred"]

    ax[0].scatter(proj_true[:, 0], proj_pred[:, 0], alpha = .1)
    ax[1].scatter(proj_true[:, 1], proj_pred[:, 1], alpha = .1)


    ax[0].set_title("Proj U")
    ax[1].set_title("Proj V")
    ax[0].set_xlim([0, 640])
    ax[0].set_ylim([0, 640])

    ax[1].set_xlim([0, 360])
    ax[1].set_ylim([0, 360])

    ax[0].axline((0, 0), (1, 1), linewidth=1, color='black', linestyle = '--')
    ax[1].axline((0, 0), (1, 1), linewidth=1, color='black', linestyle = '--')
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].set_xlabel("Ground truth")
    ax[0].set_ylabel("Prediction")


    return fig

def proj_error_distribution(ds):
    proj_true = ds["proj_true"]
    proj_pred = ds["proj_pred"]
    errors = np.linalg.norm(proj_true - proj_pred, axis = 1)
    errors = errors[errors < 700]
    fig, ax = plt.subplots(1,1)


    ax.hist(errors, bins = 300, density = True)
    ax.set_xlim(0, 700)
    ax.set_xlabel("Error [px]")
    ax.set_ylabel("Frequency")
    ax.set_title("Image space prediction error distribution")


    return fig

def orientation_error_distribution(ds):
    theta_true = ds["theta_true"]
    theta_pred = ds["theta_pred"]
    errors = angle_difference(theta_true, theta_pred)
    fig, ax = plt.subplots(1,1)


    ax.hist(errors, bins = 300, density=True)
    ax.set_xlim(0, np.pi)
    ax.set_xlabel("Error [rad]")
    ax.set_ylabel("Frequency")
    ax.set_title("Orientation prediction error distribution")


    return fig

def orientation_error_by_orientation(ds):
    theta_true = ds["theta_true"]
    theta_pred = ds["theta_pred"]
    error = angle_difference(theta_true, theta_pred)
    fig, ax = plt.subplots(1,1)
    ax.scatter(theta_true, error)
    corr = np.corrcoef(theta_true, error)[0, 1]
    ax.set_title(f"Orientation error by orientation. Pearson: {corr}")
    ax.set_aspect('equal')
    ax.set_xlabel("Theta true [rad]")
    ax.set_ylabel("Error [rad]")
    return fig

def distance_error_distribution(ds):
    dist_true = ds["dist_true"]
    dist_pred = ds["dist_pred"]
    errors = np.abs(dist_true - dist_pred)
    fig, ax = plt.subplots(1,1)


    ax.hist(errors, bins = 300, density=True)
    ax.set_xlim(0, np.pi)
    ax.set_xlabel("Error [m]")
    ax.set_ylabel("Frequency")
    ax.set_title("Predicted Distance Aboslute Error Distribution")

    return fig

def custom_scatter(x_key, y_key, title, correlation = False, plot_name = None, **kwargs):

    def scatter_fn(data):
        x_data = data[x_key]
        y_data = data[y_key]
        fig, ax = plt.subplots(1,1)
        ax.scatter(x_data, y_data)
        if correlation:
            corr = np.corrcoef(x_data, y_data)
            ax.set_title(title + f"\nPearson: {corr[0, 1]}")
        else:
            ax.set_title(title)
        ax.set(**kwargs)
        # ax.set_aspect('equal')
        return fig
    if plot_name:
        scatter_fn.__name__ = plot_name
    return scatter_fn

def sns_histplot(df, x_col, group_col):
    return sns.histplot(df, x = x_col, hue=group_col, bins = 100, stat='probability', element='step')

def pose_add_jointplot_with_title(title):

    def wrapper(data):
        res = pose_add_jointplot(data)
        res.axes[0].set_title(title)
    return wrapper

def pose_add_jointplot(data):
    pose_add = data["pose_add_30_30"]
    dist_true = data["dist_true"]
    theta_true = data["theta_true"]
    
    dist_quants = np.quantile(dist_true, [.2, .4, .6, .8, 1.])
    theta_quants = np.quantile(theta_true, [.2, .4, .6, .8, 1.])
    dist_bins = np.digitize(dist_true, dist_quants)
    theta_bins = np.digitize(theta_true, theta_quants)

    dist_bin_ids = np.arange(dist_quants.shape[0])
    theta_bin_ids = np.arange(theta_quants.shape[0])

    m = np.zeros(len(dist_quants) * len(theta_quants))

    for i, (db, tb) in enumerate(product(dist_bin_ids, theta_bin_ids)):
        joint_bin_mask = (dist_bins == db) & (theta_bins == tb)
        m[i] = pose_add[joint_bin_mask].mean() * 100
    
    m = m.reshape(len(dist_quants), len(theta_quants))
    
    fig, ax = plt.subplots(1,1)

    ax.matshow(m)

    ax.set_yticks(np.arange(len(dist_quants)), labels=np.round(dist_quants, 2))
    ax.set_xticks(np.arange(len(theta_quants)), labels=np.round(theta_quants, 2))
    ax.set_xlabel("Orientation [rad]")
    ax.set_ylabel("Distance [m]")

    for i in range(len(theta_quants)):
        for j in range(len(dist_quants)):
            text = ax.text(j, i, np.round(m[i, j], 2),
                        ha="center", va="center", color="w")
    
    ax.set_title(f"ADD per distance and orientation. Total ADD: {pose_add.mean() * 100:.2f}")
    return fig


def plot_multi_add(data):
    dists = np.arange(.10, .30, .01)
    thetas = np.arange(np.deg2rad(1), np.deg2rad(30), np.deg2rad(1.5))
    X,Y = np.meshgrid(thetas, dists)

    pos_cond = data["pose_rel_err"][:, None, None] < Y
    theta_cond = data["theta_error"][:, None, None] < X

    combined = pos_cond & theta_cond
    ADD = combined.mean(axis = 0)

    fig, ax = plt.subplots(1,1)
    im = ax.imshow(ADD, cmap = 'plasma')
    cset = ax.contour(ADD, np.arange(0, 1, .1), linewidths=2, colors = 'red')
    ax.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax = cax, orientation = 'vertical')
    ax.set_yticks(np.arange(len(dists)), np.round(dists, 2))
    ax.set_xticks(np.arange(len(thetas)), np.round(np.rad2deg(thetas), 0), rotation = 'vertical')
    ax.set_xlabel("Theta threshold [deg]")
    ax.set_ylabel("Distance threshold [m]")

    ax.set_title("ADD score by different thresholds")
    fig.tight_layout()
    return fig
