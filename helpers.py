import math
import os

from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import similaritymeasures

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cluster_trajectories_kmeans(trajectory, k, standardize=False):
    if standardize:
        scaler = StandardScaler()
        trajectory = scaler.fit_transform(trajectory)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(trajectory)
    return kmeans.labels_


def cluster_trajectories_dbscan(distance_matrix, eps=1000, min_samples=1, standardize=False, algorithm='auto', leaf_size=30):
    """
    :param eps: unit m for Frechet distance, m^2 for Area
    """
    if standardize:
        scaler = StandardScaler()
        distance_matrix = scaler.fit_transform(distance_matrix)
    cl = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', algorithm=algorithm, leaf_size=leaf_size)
    cl.fit(distance_matrix)
    return cl.labels_


def compute_distance_matrix(trajectories, method="Frechet", x_axis='x', y_axis='y'):
    """
    :param method: "Frechet" or "Area"
    """
    n = len(trajectories)
    dist_m = np.zeros((n, n))
    for i in range(n - 1):
        p = trajectories[i][[x_axis, y_axis]].values
        for j in range(i + 1, n):
            q = trajectories[j][[x_axis, y_axis]].values
            if method == "Frechet":
                dist_m[i, j] = similaritymeasures.frechet_dist(p, q)
            else:
                dist_m[i, j] = similaritymeasures.area_between_two_curves(p, q)
            dist_m[j, i] = dist_m[i, j]
    return dist_m


def add_laplace(trajectory, epsilon):
    noise = np.random.laplace(
        scale=1 / epsilon, size=np.vstack(trajectory).shape)
    noisy_trajectory = np.vstack(trajectory) + noise
    return noisy_trajectory


def add_laplace_df(trajectory_df, epsilon):
    noise = np.random.laplace(scale=1 / epsilon, size=trajectory_df.shape)
    noisy_trajectory = trajectory_df + noise
    return noisy_trajectory

# Visualization functions


def plot_noisy_trajectory(traj, output_file_prefix='', output_dir_path=''):
    fig, ax = plt.subplots()  # figsize=(20, 10))
    ax.plot(traj['x'], traj['y'], 'b.-', label='Original')
    ax.plot(traj['noisy_x'], traj['noisy_y'], 'r.-', label='Noisy')
    ax.set_title('Original vs. Noisy Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.savefig(output_dir_path + output_file_prefix + 'noisy_trajectory.pdf' , bbox_inches='tight')
    plt.show()


def plot_clustered_trajectories(traj_list, labels, x_col='x', y_col='y', title=None, output_file_prefix='', output_dir_path=''):
    colors = {label: plt.cm.jet(i / len(set(labels))) for i, label in enumerate(set(labels))}
    fig, ax = plt.subplots()

    for traj, label in zip(traj_list, labels):
        ax.plot(traj[x_col], traj[y_col], color=colors[label], linewidth=2)

    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())

    if title:
        ax.set_title(title)

    # Create a custom legend for the clusters
    legend_elements = [Line2D([0], [0], color=colors[label], lw=2, label=f'Cluster {label}') for label in set(labels)]
    ax.legend(handles=legend_elements, loc='best')
    plt.savefig(output_dir_path + output_file_prefix + 'clustered_trajectories.pdf' , bbox_inches='tight')
    plt.show()
