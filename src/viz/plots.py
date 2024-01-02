from matplotlib import pyplot as plt
import numpy as np
from itertools import product
from src.metrics import angle_difference

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
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
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


    return fig

def orientation_error_distribution(ds):
    theta_true = ds["theta_true"]
    theta_pred = ds["theta_pred"]
    errors = angle_difference(theta_true, theta_pred)
    fig, ax = plt.subplots(1,1)


    ax.hist(errors, bins = 300, density=True)
    ax.set_xlim(0, np.pi)
    ax.set_xlabel("Error [rad]")

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
    ax.set_title("Absolute distance error distribution")

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
        ax.set_aspect('equal')
        return fig
    if plot_name:
        scatter_fn.__name__ = plot_name
    return scatter_fn
