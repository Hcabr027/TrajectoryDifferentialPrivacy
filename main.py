import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import glob
import os


def add_laplace(trajectory, epsilon):
    noise = np.random.laplace(scale=1 / epsilon, size=np.vstack(trajectory).shape)
    noisy_trajectory = np.vstack(trajectory) + noise
    return noisy_trajectory
# 
# 
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
#     ax.legend()
#     plt.show()
# Plot the original and the noisy trajectory

def plot_trajectories(trajectory, noisy_trajectory):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(np.vstack(trajectory)[:, 0], np.vstack(trajectory)[:, 1], 'b.-', label='Original')
    ax.plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], 'r.-', label='Noisy')
    ax.set_title('Original vs. Noisy Trajectory')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.show()

traj_dir = './'  # directory where the trajectories are stored
def read_trajectories():
    for traj_num in range(1, 201):
        traj_file = os.path.join(traj_dir, f"traj_{traj_num}.txt")
        traj = pd.read_table(traj_file, header=None, delimiter=';', skiprows=1)  # skip the first row
        traj['traj_num'] = traj_num  # add a column with the trajectory number
        yield traj

    # Concatenate the trajectory DataFrames together using a generator expression


traj_df = pd.concat(read_trajectories(), ignore_index=True)
print(traj_df)



def rdp_with_index(points, indices, epsilon):
    """rdp with returned point indices
    """
    dmax, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        #print(f"i={i}, d={d}, dmax={dmax}, index={index}")  # add this line
        if d > dmax:
            dmax, index = d, i
    if dmax >= epsilon:
        first_points, first_indices = rdp_with_index(points[:index + 1], indices[:index + 1], epsilon)
        second_points, second_indices = rdp_with_index(points[index:], indices[index:], epsilon)
        results = np.concatenate((first_points[:-1], second_points))
        results_indices = np.concatenate((first_indices[:-1], second_indices))
    else:
        results, results_indices = np.array([points[0], points[-1]]), np.array([indices[0], indices[-1]])
    return results, results_indices


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

def frechet_dist(traj1, traj2):
    m, n = len(traj1), len(traj2)
    ca = np.zeros((m, n))
    ca[0, 0] = point_line_distance(traj1[0], traj2[0], traj2[-1])
    for i in range(1, m):
        ca[i, 0] = max(ca[i-1, 0], point_line_distance(traj1[i], traj2[0], traj2[-1]))
    for j in range(1, n):
        ca[0, j] = max(ca[0, j-1], point_line_distance(traj1[0], traj2[j], traj2[-1]))
    for i in range(1, m):
        for j in range(1, n):
            ca[i, j] = max(
                min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]),
                point_line_distance(traj1[i], traj2[j], traj2[-1])
            )
    return ca[m-1, n-1]


def compute_distance_matrix(traj_df, eps):
    traj_list = []
    for traj_num in traj_df['traj_num'].unique():
        traj_points = traj_df.loc[traj_df['traj_num'] == traj_num, ['x', 'y']].values
        traj_list.append(traj_points)
    dist_matrix = np.zeros((len(traj_list), len(traj_list)))
    for i in range(len(traj_list)):
        for j in range(i + 1, len(traj_list)):
            dist = frechet_dist(traj_list[i], traj_list[j])
            if dist <= eps:
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    return dist_matrix


#traj_dir = './'  # replace with actual path

simplified_trajs = []
simplified_indices = []
for traj in read_trajectories():
    traj_points = traj[[0, 1]].values
    traj_indices = np.arange(len(traj_points))
    simplified_traj, simplified_index = rdp_with_index(traj_points, traj_indices, epsilon=1.0)
    simplified_trajs.append(simplified_traj)
    simplified_indices.append(simplified_index)

# Concatenate the simplified trajectories into a single DataFrame
simplified_traj_df = pd.DataFrame(np.concatenate(simplified_trajs), columns=['x', 'y'])
simplified_traj_df['traj_num'] = np.repeat(np.arange(1, 201), [len(s) for s in simplified_trajs])
simplified_traj_df['index'] = np.concatenate(simplified_indices)

#print(simplified_traj_df)

header = 'lan,lon'
trajectory_33 = np.loadtxt('traj_33.txt', delimiter=';', usecols=(0, 1), skiprows=2)
noisy_trajectory_33 = add_laplace(trajectory_33, 300)

np.savetxt('noisy_trajectory_33.csv', noisy_trajectory_33, delimiter=',', header=header, comments='')
plot_trajectories(trajectory_33, noisy_trajectory_33)
