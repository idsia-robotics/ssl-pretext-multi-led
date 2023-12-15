from matplotlib import pyplot as plt
import numpy as np

def theta_scatter_plot(ds):

    theta_true = ds["theta_true"]
    theta_pred = ds["theta_pred"]
    fig, ax = plt.subplots(1,1, subplot_kw={'projection' : 'polar'})
    theta_to_deg = np.rad2deg(theta_pred)
    true_theta_to_deg = np.rad2deg(theta_true)

    theta_to_deg[theta_to_deg < 0] = 360 + theta_to_deg[theta_to_deg < 0]
    true_theta_to_deg[true_theta_to_deg < 0] = 360 + true_theta_to_deg[true_theta_to_deg < 0]

    ax.scatter(theta_true, true_theta_to_deg, color = 'blue', label='Ideal')
    ax.scatter(theta_true, theta_to_deg, color = 'red', label="Model")
    fig.legend()
    # ax.scatter(theta_true, theta_to_deg)
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
    fig, ax = plt.subplots(1,1)


    ax.hist(errors, bins = 300)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Error [px]")

    return fig
