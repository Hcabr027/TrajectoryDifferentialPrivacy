# Trajectory Privacy Preservation with Differential Privacy

This repository contains Python scripts that implement trajectory privacy preservation techniques using differential privacy. The goal is to protect sensitive trajectory data while enabling useful analysis at a broader level.

## Prerequisites

- Python 3.x installed on your system
- Required Python packages (Install with `pip install -r requirements.txt`): matplotlib,scikit-learn,similaritymeasures,shapely,numpy,pandas,fastdtw,cartopy,GDAL

## Note for Windows users:
Installing GDAL and Cartopy packages on Windows is complicated, for higher chances of success please do the following:

1. Download and install Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   
3. Download precompiled GDAL Wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
   
5. Install GDAL wheel with PIP: `pip install package-name.whl`
   
7. Install Cartopy from github with PIP: `pip install git+https://github.com/SciTools/cartopy`


## Getting Started

1. Clone this repository to your local machine.

2. Install the required Python packages by running `pip install -r requirements.txt`.

3. Run 'cluster_and_plot_air_boat_trajectories.py' to analyze in-situ data collected by our low-cost airboat USVs.

4. Run 'cluster_and_plot_ushant_ais_trajectories.pu' to analyze trajectories downloaded from the public Ushant AIS dataset.

## Usage

Both scripts do the following:

- Reads datasets in the 'in' directory

- Adds Laplacian noise to the GPS coordinates of trajectories.

- DBSCAN clustering based on the noisy GPS coordinates.

- Uses the DTW algorithm to select representative trajectories for each cluster.

- Adjust the privacy parameter (epsilon) in the scripts to control the level of privacy protection.

- Outputs Charts to the 'out' directory

## Contributors

- Cesar A. Rojas(https://github.com/croja022)
- Heidys Cabrera(https://github.com/hcabr027)

## Related Links
- Low-cost airboat USVs(https://github.com/AdmiralCrow/HoverBoat)
- Ushant AIS Public Dataset(https://github.com/rtavenar/ushant_ais)

## License

This project is licensed under the [MIT License](LICENSE).
