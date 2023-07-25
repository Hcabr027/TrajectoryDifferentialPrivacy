import math
import os

from fastdtw import fastdtw
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cartopy
import similaritymeasures

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
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


def plot_noisy_trajectory_on_map(traj, output_file_prefix='', output_dir_path=''):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)

    # Plot the original trajectory
    ax.plot(traj['x'], traj['y'], 'b.-', label='Original', transform=ccrs.PlateCarree())

    # Plot the noisy trajectory
    ax.plot(traj['noisy_x'], traj['noisy_y'], 'r.-', label='Noisy', transform=ccrs.PlateCarree())

    # Set map limits and title
    ax.set_extent([min(traj['x']), max(traj['x']), min(traj['y']), max(traj['y'])], crs=ccrs.PlateCarree())
    ax.set_title('Original vs. Noisy Trajectory')

    # Add map features (e.g., coastline, gridlines, etc.)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Add legend
    ax.legend()

    # Save the plot to a file or show it
    plt.savefig(output_dir_path + output_file_prefix + 'noisy_trajectory_map.pdf', bbox_inches='tight')
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

    # Add map features (e.g., coastline, gridlines, etc.)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Create a custom legend for the clusters
    legend_elements = [Line2D([0], [0], color=colors[label], lw=2, label=f'Cluster {label}') for label in set(labels)]
    ax.legend(handles=legend_elements, loc='best')
    plt.savefig(output_dir_path + output_file_prefix + 'clustered_trajectories.pdf' , bbox_inches='tight')
    plt.show()


def plot_clustered_trajectories_on_map(traj_list, labels, x_col='x', y_col='y', title=None, output_file_prefix='', output_dir_path=''):
    colors = {label: plt.cm.jet(i / len(set(labels))) for i, label in enumerate(set(labels))}
    
    # Create a figure and axis with Cartopy projection
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add Cartopy features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
    
    for traj, label in zip(traj_list, labels):
        ax.plot(traj[x_col], traj[y_col], color=colors[label], linewidth=2, transform=ccrs.PlateCarree())

    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())

    if title:
        ax.set_title(title)

    # Add map features (e.g., coastline, gridlines, etc.)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Create a custom legend for the clusters
    legend_elements = [Line2D([0], [0], color=colors[label], lw=2, label=f'Cluster {label}') for label in set(labels)]
    ax.legend(handles=legend_elements, loc='best')
    
    # Find the min and max values of x and y across all trajectories
    min_x = min([traj[x_col].min() for traj in traj_list])
    max_x = max([traj[x_col].max() for traj in traj_list])
    min_y = min([traj[y_col].min() for traj in traj_list])
    max_y = max([traj[y_col].max() for traj in traj_list])

    # Set map extent to include all trajectories
    ax.set_extent([min_x, max_x, min_y, max_y], crs=ccrs.PlateCarree())

    plt.savefig(output_dir_path + output_file_prefix + 'clustered_trajectories.pdf', bbox_inches='tight')
    plt.show()


def plot_clustered_trajectories_on_satellite(traj_list, labels, x_col='x', y_col='y', title=None, output_file_prefix='', output_dir_path='', padding=0.1):
    colors = {label: plt.cm.jet(i / len(set(labels))) for i, label in enumerate(set(labels))}
    
    # Create a figure and axis with Cartopy projection
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Define the tile source using a web map tile service (WMTS)
    request = cimgt.GoogleTiles(style='satellite')
    ax.add_image(request, 20)  # You can adjust the zoom level (e.g., 14) to change the level of detail
    
    for traj, label in zip(traj_list, labels):
        ax.plot(traj[x_col], traj[y_col], color=colors[label], linewidth=2, transform=ccrs.PlateCarree())

    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())

    if title:
        ax.set_title(title)

    # Add map features (e.g., coastline, gridlines, etc.)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Create a custom legend for the clusters
    legend_elements = [Line2D([0], [0], color=colors[label], lw=2, label=f'Cluster {label}') for label in set(labels)]
    ax.legend(handles=legend_elements, loc='best')
    
    # Find the min and max values of x and y across all trajectories
    min_x = min([traj[x_col].min() for traj in traj_list])
    max_x = max([traj[x_col].max() for traj in traj_list])
    min_y = min([traj[y_col].min() for traj in traj_list])
    max_y = max([traj[y_col].max() for traj in traj_list])

    # Calculate padding for x and y
    x_padding = padding * (max_x - min_x)
    y_padding = padding * (max_y - min_y)

    # Set map extent with padding
    ax.set_extent([min_x - x_padding, max_x + x_padding, min_y - y_padding, max_y + y_padding], crs=ccrs.PlateCarree())

    plt.savefig(output_dir_path + output_file_prefix + 'clustered_trajectories.pdf', bbox_inches='tight')
    plt.show()


def create_cluster_representatives_random(trajectories_groups, labels, x_axis='x', y_axis='y', seed=None):
    # Get the unique cluster labels
    unique_labels = np.unique(labels)

    # Initialize a list to store the representative trajectories for each cluster
    cluster_representatives = []
    representative_labels = []

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Iterate through each unique cluster label
    for cluster_label in unique_labels:
        # Find the indices of trajectories that belong to the current cluster
        cluster_indices = np.where(labels == cluster_label)[0]

        # Select a random trajectory from the current cluster
        random_trajectory_idx = np.random.choice(cluster_indices)
        random_trajectory = trajectories_groups[random_trajectory_idx]

        # Append the random trajectory to the list
        cluster_representatives.append(random_trajectory)

        # Add the label of the random trajectory
        representative_labels.append(cluster_label)

    return cluster_representatives, representative_labels


# Find the average trajectory using DTW
def find_representative_trajectory(sequences):
    min_avg_distance = float('inf')
    representative_traj = None

    for i, traj in enumerate(sequences):
        total_distance = 0

        for j, other_traj in enumerate(sequences):
            if i != j:
                # Use DTW to calculate the distance between two trajectories
                distance, _ = fastdtw(traj, other_traj)
                total_distance += distance

        # Calculate the average distance to other trajectories
        avg_distance = total_distance / (len(sequences) - 1)

        # Update the representative trajectory if it has a lower average distance
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            representative_traj = traj

    return representative_traj


def create_cluster_representatives_dtw(trajectories_groups, labels, x_axis='x', y_axis='y', seed=None):
    # Initialize lists to store the representative trajectories and their labels for each cluster
    cluster_representatives = []
    representative_labels = []

    # Create a random seed if not provided
    if seed is not None:
        np.random.seed(seed)

    # Get the unique cluster labels
    unique_labels = np.unique(labels)

    # Iterate through each unique cluster label
    for cluster_label in unique_labels:
        # Find the indices of trajectories that belong to the current cluster
        cluster_indices = np.where(labels == cluster_label)[0]

        # Create sequences with specified columns for each group
        sequences = []
        for i in cluster_indices:
            trajectory_group = trajectories_groups[i]
            sequence = trajectory_group[[x_axis, y_axis, 'Temperature (C)', 'Depth (cm)', 'Turbidity (NTU)']].values
            sequences.append(sequence)

        # Find the representative trajectory using the custom distance function
        representative_traj = find_representative_trajectory(sequences)

        # Convert the representative trajectory back to a DataFrame
        representative_traj_df = pd.DataFrame(representative_traj, columns=[x_axis, y_axis, 'Temperature (C)', 'Depth (cm)', 'Turbidity (NTU)'])

        # Append the representative trajectory and its label to the lists
        cluster_representatives.append(representative_traj_df)
        representative_labels.append(cluster_label)

    return cluster_representatives, representative_labels
