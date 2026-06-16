# Targeted marine cloud brightening weakens subsequent El Niño
This repository contains the code for running analyses and producing figures in the associated manuscript (Wan et al., 2026).

## Data access
Note, you will need to populate `_data` with associated data files available from NCAR GDEX. Small data files used in analysis but not in GDEX are provided in `_data`. See `/_data/data_README.txt` for more information.

Wan, J. S., J. T. Fasullo, N. A. Rosenbloom, C. Chen, and K. Ricke. 2026. CESM2 SMYLE MCB. NSF National Center for Atmospheric Research. https://gdex.ucar.edu/datasets/d651084/. Accessed† dd mmm yyyy.

## Running the code
All provided scripts are written in Python 3.8.2. Code has been tested on Linux-64 OS. Runtime may take longer (>1 hour) running on a "normal" desktop computer, so it is recommended to use a system with ample storage and memory. Before running the code, you will need to create a new conda environment with the correct dependencies from `conda_requirements.txt`
```
conda create --name <env> --file conda_requirements.txt
```


