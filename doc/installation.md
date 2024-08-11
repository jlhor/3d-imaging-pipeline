## Installation Guide

A Conda virtual environment needs to be set up on the HPC cluster. We strongly recommend starting out by following the instructions as specified in the [Stardist Conda environment setup guide](https://github.com/CSBDeep/CSBDeep/tree/main/extras#conda-environment) *and* the additional steps as laid out in the [Stardist installation guide](https://github.com/stardist/stardist?tab=readme-ov-file#installation) to ensure the proper installation of GPU-enabled Stardist package.

> [!IMPORTANT]
> For GPU support to be enabled for Tensorflow during Stardist installation, the steps above must be performed on a GPU node on the HPC. We recommend launching an interactive job on a GPU node on your HPC cluster and proceed with the installation from there.

### Additional Python dependencies

After Stardist has been successfully installed on the Conda environment (accessible with the `conda activate <env_name>` command), the following Python packages can be installed using the `pip install` command:

```
numpy (>=1.22.1)
scipy (>=1.10.1)
scikit-image (>= 0.21.0)
numba (>=0.58.1)
pandas (>=2.0.3)
tables (>=3.8.0)
h5py (>=2.10.0)
tifffile (>=2023.7.10)
zarr (>=2.13.1)
dask (>=2023.5.0)
dask-jobqueue (>=0.8.2)
distributed (>=2023.5.0)
pyyaml (>=6.0.1)
ruamel-yaml (>=0.18.6)
tqdm (>=4.66.2)
```

### Cloning the `3d-imaging-pipeline` repository

Navigate to the desired directory and execute the command `git clone https://github.com/jlhor/3d-imaging-pipeline.git` to clone the entire repository to the current directory.

If the Conda environment has not been activated, use the command `conda activate <env_name>` to activate the virtual environment containing the installed packages. The Python scripts for `3d-imaging-pipeline` are now ready to be executed.

