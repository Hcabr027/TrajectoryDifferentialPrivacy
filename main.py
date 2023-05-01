import math
import os

from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import similaritymeasures

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# def cluster_trajectories(trajectory, k):
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(trajectory)
#     return kmeans.labels_
#
#
# def add_noise_and_cluster(trajectory, epsilon, k):
#     noisy_trajectory = add_laplace(trajectory, epsilon)
#     labels = cluster_trajectories(noisy_trajectory, k)
#     label_colors = pd.Series(labels).map({0: 'orange', 1: 'yellow', 2: 'red', 3: 'purple', 4: 'green'})
#
#     noisy_trajectory_df = pd.DataFrame(
#         {'Latitude': noisy_trajectory[:, 0], 'Longitude': noisy_trajectory[:, 1], 'Label': labels,
#          'Color': label_colors})
#     return label_colors, noisy_trajectory_df
#
#
# def plot_trajectories(trajectory, noisy_trajectory_df, marker, label_color):
#     fig, ax = plt.subplots(figsize=(20, 10))
#     ax.plot(np.vstack(trajectory)[:, 0], np.vstack(trajectory)[:, 1], f'b{marker}-', label='Original')
#     for label, group in noisy_trajectory_df.groupby('Label'):
#         ax.scatter(group['Latitude'], group['Longitude'], color=group['Color'], marker=marker, label=f'Cluster {label}')
#     ax.set_title('Original vs. Noisy Trajectory')
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.legend()//
#     plt.show()
# Plot the original and the noisy trajectory
# Utility functions
def frechet_dist(traj1, traj2):
    m, n = len(traj1), len(traj2)
    ca = np.zeros((m, n))
    ca[0, 0] = point_line_distance(traj1[0], traj2[0], traj2[-1])
    for i in range(1, m):
        ca[i, 0] = max(
            ca[i - 1, 0], point_line_distance(traj1[i], traj2[0], traj2[-1]))
    for j in range(1, n):
        ca[0, j] = max(
            ca[0, j - 1], point_line_distance(traj1[0], traj2[j], traj2[-1]))
    for i in range(1, m):
        for j in range(1, n):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                point_line_distance(traj1[i], traj2[j], traj2[-1])
            )
    return ca[m - 1, n - 1]


def point_line_distance(point, start, end):

    x1, y1 = point.astype(float)
    x2, y2 = start.astype(float)
    x3, y3 = end.astype(float)
    px = x3 - x2
    py = y3 - y2
    norm = px * px + py * py
    u = ((x1 - x2) * px + (y1 - y2) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x2 + u * px
    y = y2 + u * py
    dx = x - x1
    dy = y - y1
    distance = math.sqrt(dx * dx + dy * dy)
    return distance


def rdp_with_index(points, indices, epsilon):
    """rdp with returned point indices
    """
    dmax, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            dmax, index = d, i
    if dmax >= epsilon:
        first_points, first_indices = rdp_with_index(points[:index + 1], indices[:index + 1], epsilon)
        second_points, second_indices = rdp_with_index(points[index:], indices[index:], epsilon)
        results = first_points[:-1] + second_points
        results_indices = first_indices[:-1] + second_indices
    else:
        results, results_indices = [points[0], points[-1]], [indices[0], indices[-1]]
    return results, results_indices


def compute_distance_matrix(trajectories, method="Frechet"):
    """
    :param method: "Frechet" or "Area"
    """
    n = len(trajectories)
    dist_m = np.zeros((n, n))
    for i in range(n - 1):
        p = trajectories[i]
        for j in range(i + 1, n):
            q = trajectories[j]
            if method == "Frechet":
                dist_m[i, j] = similaritymeasures.frechet_dist(p, q)
            else:
                dist_m[i, j] = similaritymeasures.area_between_two_curves(p, q)
            dist_m[j, i] = dist_m[i, j]
    return dist_m


def clustering_by_dbscan(distance_matrix, eps=1000, min_samples=1):
    """
    :param eps: unit m for Frechet distance, m^2 for Area
    """
    cl = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cl.fit(distance_matrix)
    return cl.labels_


def add_laplace(trajectory, epsilon):
    noise = np.random.laplace(
        scale=1 / epsilon, size=np.vstack(trajectory).shape)
    noisy_trajectory = np.vstack(trajectory) + noise
    return noisy_trajectory

# Visualization functions


def plot_trajectories(trajectory, noisy_trajectory, output_file_prefix=''):
    fig, ax = plt.subplots()  # figsize=(20, 10))
    ax.plot(np.vstack(trajectory)[:, 0], np.vstack(
        trajectory)[:, 1], 'b.-', label='Original')
    ax.plot(noisy_trajectory[:, 0],
            noisy_trajectory[:, 1], 'r.-', label='Noisy')
    ax.set_title('Original vs. Noisy Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.savefig(output_dir + output_file_prefix + 'noisy_trajectory.pdf' , bbox_inches='tight')
    plt.show()


def plot_clustered_trajectories(traj_list, labels, x_col='x', y_col='y', title=None, output_file_prefix=''):
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
    plt.savefig(output_dir + output_file_prefix + 'clustered_trajectories.pdf' , bbox_inches='tight')
    plt.show()


# Data reading and preprocessing
traj_dir = './in/'  # directory where the trajectories are stored
output_dir = './out/'
simplified_trajs = []
simplified_indices = []
num_trajectories = 200

for traj_num in range(1, num_trajectories + 1):
    traj_file = os.path.join(traj_dir, f"traj_{traj_num}.txt")
    traj = pd.read_table(traj_file, delimiter=';')

    traj_points = traj[['x', 'y']].values
    traj_indices = np.arange(len(traj_points))
    simplified_traj, simplified_index = rdp_with_index(traj_points, traj_indices, epsilon=1.0)

    simplified_trajs.append(simplified_traj)
    simplified_indices.append(simplified_index)

simplified_traj_df = pd.DataFrame(np.concatenate(simplified_trajs), columns=['x', 'y'])
simplified_traj_df['traj_num'] = np.repeat(np.arange(1, num_trajectories + 1), [len(s) for s in simplified_trajs])
simplified_traj_df['index'] = np.concatenate(simplified_indices)

print('simplified trajectories')
print(simplified_traj_df)

header = 'Latitude,Longitude'
traj_file_name = "traj_2"
traj_file = os.path.join(traj_dir, traj_file_name + '.txt')
trajectory = np.loadtxt(traj_file, delimiter=';', usecols=(0, 1), skiprows=2)
noisy_trajectory = add_laplace(trajectory, 300)
output_file = os.path.join(output_dir, 'noisy_' + traj_file_name + '.csv')
np.savetxt(output_file, noisy_trajectory, delimiter=',', header=header, comments='')
# plot_trajectories(trajectory, noisy_trajectory)

traj_1 = simplified_traj_df[simplified_traj_df['traj_num'] == 1]
traj_2 = simplified_traj_df[simplified_traj_df['traj_num'] == 2]

traj_list = [traj_1.copy(), traj_2.copy()]  # Create deep copies to avoid SettingWithCopyWarning

# Add noisy x and y columns to each DataFrame in traj_list
for traj in traj_list:
    noisy_points = add_laplace(traj[['x', 'y']].values, 10)
    traj['noisy_x'] = noisy_points[:, 0]
    traj['noisy_y'] = noisy_points[:, 1]

dist_m = compute_distance_matrix(traj_list)
print('dist_m')
print(dist_m)

labels = clustering_by_dbscan(dist_m)
print(labels)

print(traj_list[0])

# Call the function to plot the original and noisy clustered trajectories
# plot_clustered_trajectories(traj_list, labels, x_col='x', y_col='y', title="Clustered Trajectories")
# plot_clustered_trajectories(traj_list, labels, x_col='noisy_x', y_col='noisy_y', title="Clustered Noisy Trajectories", output_file_prefix='noisy_')

print('all trajectories')

# Group simplified_traj_df by 'traj_num' and create a list of DataFrames
traj_list = [group.copy() for _, group in simplified_traj_df.groupby('traj_num')]

# Add noisy x and y columns to each DataFrame in traj_list
for traj in traj_list:
    noisy_points = add_laplace(traj[['x', 'y']].values, 10)
    traj['noisy_x'] = noisy_points[:, 0]
    traj['noisy_y'] = noisy_points[:, 1]

traj_list = traj_list[:20]
# Compute distance matrix and perform clustering
dist_m = compute_distance_matrix(traj_list)
labels = clustering_by_dbscan(dist_m, 40)
print(labels)

# Call the function to plot the original and noisy clustered trajectories
# plot_clustered_trajectories(traj_list, labels, x_col='x', y_col='y', title="Clustered Trajectories")
plot_clustered_trajectories(traj_list, labels, x_col='noisy_x', y_col='noisy_y', title="Clustered Noisy Trajectories", output_file_prefix='all_noisy_')
