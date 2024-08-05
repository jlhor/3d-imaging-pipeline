# Segmentation with StarDist3D

This workflow requires a model to be first trained using [StarDist3D](https://github.com/stardist/stardist), using the original [example](https://github.com/stardist/stardist/tree/main/examples/3D) as a starting point. (*Additional instruction will be added soon for how to train your own custom model)

Once the model has been trained, the directory containing the model can be copied and placed into a central directory, or within the directory of individual project.

## Instructions

The segmentation workflow is executed in two phases:

1) a GPU-intensive prediction step that can be run on a local workstation or on the HPC. The script provided here is tailored for a local workstation (equipped with a dedicated GPU) run of the prediction step, which will yield the `probabilities`, `distances`, and `points` saved into a HDF5-based array.

2) the output HDF5 array can then be uploaded to the HPC for the subsequent CPU-intensive prediction and labeling steps, which benefit from highly parallelized processing with large number of compute nodes to reduce the processing time by at least an order of magnitude. This step outputs a `label.h5` array that contain the individual segmented cell labels for further processing and analysis.

Note that this requires setting up a Conda-based environment in both the local workstation and on the HPC (*additional instruction will be added soon)

### Step 1: GPU prediction step

1. Configure the `local_template.yaml` file. Specify the location of the image file, channel for segmentation (starts from `0`), the directory for the trained model.
2. Run `python local_prediction.py local_template.yaml` to generate the intermediate `prediction_output.h5` array.

### Step 2: CPU prediction and labeling step

1. Copy the `probabilities.h5` file to the `output` directory of the project on the HPC, together with the directory of the trained model to the `models` sub-directory. The image file is not needed from this point onward.
2. Configure the `hpc_prediction_template.yaml` file, by specifying the location of the `probabilities.h5` file and the directory containing the trained model. Set the number of cluster (compute nodes) to request as needed.   
   Typically, editing of the configuration file is not required. However, adjusting the number of nodes to request, the number of CPUs and memory may be required depending on the HPC infrastructure of the institution. For details on the configuration parameters, see the [instructions](doc/configuration_hpc_prediction.md) here.
4. Configure the `jobscript_template.sh` to run `python hpc_prediction.py hpc_prediction_template.yaml` on a SLURM scheduler.
5. The output `prediction.h5` file containing the individual segmented cell masks/labels will be generated and can be used for downstream applications.

Note: the output label file is a 32-bit array that often contains many more cells than the maximum bit-depth of a 16-bit image (65535) and should not be converted into a 16-bit image as all the segmented cell objects above the max bit-depth value will be lost. It is important that the subsequent processing and analysis steps do not inadvertently convert the array to a lower bit-depth (e.g. 8 or 16-bit).
