# Targeted marine cloud brightening can dampen El Niño
This repository contains the code for running analyses and producing figures in the associated manuscript (Wan et al., submitted).

## Data access
Note, you will need to populate `_data` with associated data files available from NCAR's campaign storage (accessible via Casper). Some sample data files have been provided to demo the code. See `/_data/data_README.txt` for more information.

## Running the code
All provided scripts are written in Python 3.8.2. Code has been tested on Linux-64 OS. Runtime may take longer (>1 hour) running on a "normal" desktop computer, so it is recommended to use a system with ample storage and memory. Before running the code, you will need to create a new conda environment with the correct dependencies from `conda_requirements.txt`
```
conda create --name <env> --file conda_requirements.txt
```


