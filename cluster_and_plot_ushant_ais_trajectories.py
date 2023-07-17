import math
import os

from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import similaritymeasures

import helpers as hlp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data reading and preprocessing
INPUT_DIR = './in/'  # directory where the trajectories are stored
OUTPUT_DIR = './out/'


def main():
    num_trajectories = 200
    simplified_trajs = []
    simplified_indices = []

    # Open first 200 Ushant AIS trajectories
    for traj_num in range(1, num_trajectories + 1):
        traj_file = os.path.join(INPUT_DIR, f"traj_{traj_num}.txt")
        traj = pd.read_table(traj_file, delimiter=';')

        traj_points = traj[['x', 'y']].values
        traj_indices = np.arange(len(traj_points))
        simplified_traj, simplified_index = hlp.rdp_with_index(traj_points, traj_indices, epsilon=1.0)

        simplified_trajs.append(simplified_traj)
        simplified_indices.append(simplified_index)

    simplified_traj_df = pd.DataFrame(np.concatenate(simplified_trajs), columns=['x', 'y'])
    simplified_traj_df['traj_num'] = np.repeat(np.arange(1, num_trajectories + 1), [len(s) for s in simplified_trajs])
    simplified_traj_df['index'] = np.concatenate(simplified_indices)

    print('simplified trajectories')
    print(simplified_traj_df)

    header = 'Latitude,Longitude'
    traj_file_name = "traj_2"
    traj_file = os.path.join(INPUT_DIR, traj_file_name + '.txt')
    trajectory = np.loadtxt(traj_file, delimiter=';', usecols=(0, 1), skiprows=2)
    noisy_trajectory = hlp.add_laplace(trajectory, 300)
    output_file = os.path.join(OUTPUT_DIR, 'noisy_' + traj_file_name + '.csv')
    np.savetxt(output_file, noisy_trajectory, delimiter=',', header=header, comments='')
    # plot_trajectories(trajectory, noisy_trajectory, '', OUTPUT_DIR)

    traj_1 = simplified_traj_df[simplified_traj_df['traj_num'] == 1]
    traj_2 = simplified_traj_df[simplified_traj_df['traj_num'] == 2]

    traj_list = [traj_1.copy(), traj_2.copy()]  # Create deep copies to avoid SettingWithCopyWarning

    # Add noisy x and y columns to each DataFrame in traj_list
    for traj in traj_list:
        noisy_points = hlp.add_laplace(traj[['x', 'y']].values, 10)
        traj['noisy_x'] = noisy_points[:, 0]
        traj['noisy_y'] = noisy_points[:, 1]

    dist_m = hlp.compute_distance_matrix(traj_list)
    print('dist_m')
    print(dist_m)

    labels = hlp.clustering_by_dbscan(dist_m)
    print(labels)

    print(traj_list[0])

    # Call the function to plot the original and noisy clustered trajectories
    # plot_clustered_trajectories(traj_list, labels, x_col='x', y_col='y', title="Clustered Trajectories")
    # plot_clustered_trajectories(traj_list, labels, x_col='noisy_x', y_col='noisy_y', title="Clustered Noisy Trajectories", output_file_prefix='noisy_')

    print('all trajectories')

    # Group simplified_traj_df by 'traj_num' and create a list of DataFrames
    traj_list = [group.copy() for _, group in simplified_traj_df.groupby('traj_num')]
    
    print('before noise')
    print(traj_list[0])

    # Add noisy x and y columns to each DataFrame in traj_list
    for traj in traj_list:
        noisy_points = hlp.add_laplace(traj[['x', 'y']].values, 10)
        traj['noisy_x'] = noisy_points[:, 0]
        traj['noisy_y'] = noisy_points[:, 1]

    print('after noise')
    print(traj_list[0])

    traj_list = traj_list[:20]
    # Compute distance matrix and perform clustering
    dist_m = hlp.compute_distance_matrix(traj_list)
    labels = hlp.clustering_by_dbscan(dist_m, eps=1)
    print('clusters without noise')
    print('labels')
    print(labels)

    # Call the function to plot the original and noisy clustered trajectories
    prefix = 'ushant_ais'

    hlp.plot_clustered_trajectories(traj_list, labels, x_col='x', y_col='y', title="Clustered Trajectories", output_file_prefix=prefix, output_dir_path=OUTPUT_DIR)

    print('clusters with noise')
    dist_m = hlp.compute_distance_matrix(traj_list, x_axis='noisy_x', y_axis='noisy_y')
    labels = hlp.clustering_by_dbscan(dist_m, eps=1)
    print('labels')
    print(labels)
    hlp. plot_clustered_trajectories(traj_list, labels, x_col='noisy_x', y_col='noisy_y', title="Clustered Noisy Trajectories", output_file_prefix=prefix + '_noisy_', output_dir_path=OUTPUT_DIR)


if __name__ == "__main__":
    print('Main Function called')
    main()
