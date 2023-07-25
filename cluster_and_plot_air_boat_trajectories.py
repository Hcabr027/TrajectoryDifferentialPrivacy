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
    file_names = ['20230706_FIU_MMC_LAKE', '20230706_FIU_MMC_POND', '20230712_FIU_MMC_LAKE']
    file_path = os.path.join(INPUT_DIR, file_names[2] + '.csv')
    trajectories_df = pd.read_csv(file_path)

    print('all trajectories')
    print(trajectories_df.to_string())

    # Group simplified_traj_df by 'traj_num' and create a list of DataFrames
    trajectories_group_df = trajectories_df.groupby('Trajectory ID')
    trajectories_groups = [group.copy() for _, group in trajectories_group_df]

    print(trajectories_groups[0].to_string())

    # Add noisy x and y columns to each DataFrame in trajectories_groups
    for traj in trajectories_groups:
        noisy_points = hlp.add_laplace(traj[['Longitude', 'Latitude']].values, 1000000)
        traj['noisy_x'] = noisy_points[:, 0]
        traj['noisy_y'] = noisy_points[:, 1]

    print(trajectories_groups[0].to_string())

    # Compute distance matrix and perform clustering
    dist_m = hlp.compute_distance_matrix(trajectories_groups, x_axis='Longitude', y_axis='Latitude')
    labels = hlp.cluster_trajectories_dbscan(dist_m, 0.0001)
    print('clusters without noise')
    print(labels)

    # Call the function to plot the original and noisy clustered trajectories
    prefix = 'airboat'

    hlp.plot_clustered_trajectories_on_satellite(trajectories_groups, labels, x_col='Longitude', y_col='Latitude', title="Clustered Trajectories", output_file_prefix=prefix + '_', output_dir_path=OUTPUT_DIR)

    print('clusters with noise')
    dist_m = hlp.compute_distance_matrix(trajectories_groups, x_axis='noisy_x', y_axis='noisy_y')
    labels = hlp.cluster_trajectories_dbscan(dist_m, eps=0.0001)
    print('labels')
    print(labels)
    hlp. plot_clustered_trajectories_on_satellite(trajectories_groups, labels, x_col='noisy_x', y_col='noisy_y', title="Clustered Noisy Trajectories", output_file_prefix=prefix + '_noisy_', output_dir_path=OUTPUT_DIR)

    # Create cluster representatives and get their corresponding labels
    representatives, representative_labels = hlp.create_cluster_representatives_random(trajectories_groups, labels, x_axis='noisy_x', y_axis='noisy_y')

    hlp. plot_clustered_trajectories_on_satellite(representatives, representative_labels, x_col='noisy_x', y_col='noisy_y', title="Random Representative Trajectories", output_file_prefix=prefix + '_representative_random_', output_dir_path=OUTPUT_DIR)
    for df in representatives:
        print(df.to_string())
    # Create cluster representatives and get their corresponding labels
    representatives, representative_labels = hlp.create_cluster_representatives_dtw(trajectories_groups, labels, x_axis='noisy_x', y_axis='noisy_y')
    print(representatives)
    hlp. plot_clustered_trajectories_on_satellite(representatives, representative_labels, x_col='noisy_x', y_col='noisy_y', title="DTW Representative Trajectories", output_file_prefix=prefix + '_representative_dtw_', output_dir_path=OUTPUT_DIR)


if __name__ == "__main__":
    print('Main Function called')
    main()
