import math
import os

from matplotlib.lines import Line2D
from shapely.geometry import LineString
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
PREFIX = 'ushant_ais'


def main():
    num_trajectories = 200
    original_trajs = []
    simplified_trajs = []

    # Open first 200 Ushant AIS trajectories
    for traj_num in range(1, num_trajectories + 1):
        traj_file = os.path.join(INPUT_DIR, f"traj_{traj_num}.txt")
        traj = pd.read_table(traj_file, delimiter=';')
        traj['traj_num'] = traj_num  # Assign trajectory number
        original_trajs.append(traj)  # Add the original trajectory to the list

        traj_points = traj[['x', 'y']].values

        # Create LineString and simplify it
        line = LineString(traj_points)
        simplified_line = line.simplify(tolerance=1.0, preserve_topology=False)

        # Convert the simplified LineString back to a DataFrame
        simplified_traj = pd.DataFrame(simplified_line.coords, columns=['x', 'y'])
        simplified_traj['traj_num'] = traj_num  # Assign trajectory number

        simplified_trajs.append(simplified_traj)

    # Concatenate all original and simplified trajectories into a single DataFrame
    original_traj_df = pd.concat(original_trajs, ignore_index=True)
    simplified_traj_df = pd.concat(simplified_trajs, ignore_index=True)
    
    print('original trajectories')
    print(original_traj_df)
    
    print('simplified trajectories')
    print(simplified_traj_df)

    header = 'Latitude,Longitude'
    traj_file_name = "traj_2"
    traj_file = os.path.join(INPUT_DIR, traj_file_name + '.txt')
    trajectory = np.loadtxt(traj_file, delimiter=';', usecols=(0, 1), skiprows=2)
    noisy_trajectory = hlp.add_laplace(trajectory, 300)
    output_file = os.path.join(OUTPUT_DIR, 'noisy_' + traj_file_name + '.csv')
    np.savetxt(output_file, noisy_trajectory, delimiter=',', header=header, comments='')
    
    print('plot original trajectories')
    traj_1 = original_traj_df[original_traj_df['traj_num'] == 1]
    traj_2 = original_traj_df[original_traj_df['traj_num'] == 2]

    traj_list = [traj_1.copy(), traj_2.copy()]  # Create deep copies to avoid SettingWithCopyWarning

    # Add noisy x and y columns to each DataFrame in traj_list
    for traj in traj_list:
        noisy_points = hlp.add_laplace(traj[['x', 'y']].values, 100)
        traj['noisy_x'] = noisy_points[:, 0]
        traj['noisy_y'] = noisy_points[:, 1]

    dist_m = hlp.compute_distance_matrix(traj_list)
    print('dist_m')
    print(dist_m)

    labels = hlp.cluster_trajectories_dbscan(dist_m, eps=1)
    print(labels)

    print('first two trajectories')
    print(traj_list[0].to_string())
    print(traj_list[0].to_string())

    hlp.plot_noisy_trajectory(traj_list[0], PREFIX + '_1_', OUTPUT_DIR)
    hlp.plot_noisy_trajectory(traj_list[1], PREFIX + '_2_', OUTPUT_DIR)

    print('plot simplified trajectories')
    traj_1 = simplified_traj_df[simplified_traj_df['traj_num'] == 1]
    traj_2 = simplified_traj_df[simplified_traj_df['traj_num'] == 2]

    traj_list = [traj_1.copy(), traj_2.copy()]  # Create deep copies to avoid SettingWithCopyWarning

    # Add noisy x and y columns to each DataFrame in traj_list
    for traj in traj_list:
        noisy_points = hlp.add_laplace(traj[['x', 'y']].values, 100)
        traj['noisy_x'] = noisy_points[:, 0]
        traj['noisy_y'] = noisy_points[:, 1]

    dist_m = hlp.compute_distance_matrix(traj_list)
    print('dist_m')
    print(dist_m)

    labels = hlp.cluster_trajectories_dbscan(dist_m, eps=1)
    print(labels)

    print('first two simplified trajectories')
    print(traj_list[0].to_string())
    print(traj_list[0].to_string())

    hlp.plot_noisy_trajectory(traj_list[0], PREFIX + '_simplified_1_', OUTPUT_DIR)
    hlp.plot_noisy_trajectory(traj_list[1], PREFIX + '_simplified_2_', OUTPUT_DIR)

    # Call the function to plot the original and noisy clustered trajectories
    print('all simplified trajectories')

    # Group simplified_traj_df by 'traj_num' and create a list of DataFrames
    traj_list = [group.copy() for _, group in simplified_traj_df.groupby('traj_num')]

    print('before noise')
    print(traj_list[0])

    # Add noisy x and y columns to each DataFrame in traj_list
    for traj in traj_list:
        noisy_points = hlp.add_laplace(traj[['x', 'y']].values, 100)
        traj['noisy_x'] = noisy_points[:, 0]
        traj['noisy_y'] = noisy_points[:, 1]

    print('after noise')
    print(traj_list[0])

    traj_list = traj_list[:20]
    # Compute distance matrix and perform clustering
    dist_m = hlp.compute_distance_matrix(traj_list)
    labels = hlp.cluster_trajectories_dbscan(dist_m, eps=1)
    print('clusters without noise')
    print('labels')
    print(labels)

    # Call the function to plot the original and noisy clustered trajectories
    hlp.plot_clustered_trajectories(traj_list, labels, 'x', 'y', "Clustered Trajectories", PREFIX + '_', OUTPUT_DIR)

    print('clusters with noise')
    dist_m = hlp.compute_distance_matrix(traj_list, 'noisy_x', 'noisy_y')
    labels = hlp.cluster_trajectories_dbscan(dist_m, eps=.005)
    print('labels')
    print(labels)
    hlp. plot_clustered_trajectories(traj_list, labels, 'noisy_x', 'noisy_y', "Clustered Noisy Trajectories", PREFIX + '_noisy_', OUTPUT_DIR)


if __name__ == "__main__":
    print('Main Function called')
    main()
