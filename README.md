# 3D Image Processing and Analysis Pipeline

https://github.com/user-attachments/assets/8b3f1c21-825f-4c91-abda-081dcaa4e244

This workflow comprises Python-based scripts that leverage High-Performance Computing (HPC) cluster to enable distributed processing of large volumetric 3D imaging datasets, including 3D cell segmentation and single-cell channel intensity extraction. The current pipeline is optimized for SLURM-based scheduling system, with efforts to support its use in cloud-based service (e.g., Amazon AWS) currently ongoing.

## Requirements:

> [!IMPORTANT]
> This workflow is intended to be installed and executed on the HPC cluster for parallel processing of large imaging datasets. It has been configured and optimized for the NIH Biowulf HPC cluster with a Slurm-based batch scheduling system. HPC systems that utilize other types of scheduler should be compatible with slight modifications to the code base. *Support for cloud-based service (e.g., Amazon AWS) is currently under development.*

> [!IMPORTANT]
> This workflow has been optimized for use with the HDF5-based Imaris file format (`.ims`) for parallel reading of image files by multiple compute nodes. A licensed version of Bitplane Imaris is not required. Image dataset can be converted to the `.ims` format using the free [Imaris File Converter](https://imaris.oxinst.com/microscopy-imaging-software-free-trial#file-converter) tool. *Support for OME-TIFF/Zarr formats will be made available in the near future.*

This package requires a Conda virtual environment to be installed on the HPC cluster (see *Installation* guide below).   
The development version of `3d-imaging-pipeline` has been tested on the *Rocky Linux 8.7 (Green Obsidian)* OS (kernel version: Linux 4.18.0-425.19.2) running on the NIH Biowulf HPC cluster.

The [*Stardist*](https://github.com/stardist/stardist) segmentation module additionally requires access to GPU nodes on the HPC to accelerate the prediction step that generates object probabilities and the distances to object boundaries.

## Installation 

Please follow the [Installation Guide](doc/installation.md) for setting up a Conda environment on the HPC as well as the associated Python dependencies.

## Code execution

See the [Workflow](doc/workflow.md) of `3d-imaging-pipeline` and the detailed instructions for the execution of each step.

*For a step-by-step walkthrough of a demo dataset, follow the instructions [here](doc/demo_instructions.md)*

## License
This project is covered under the MIT License.

