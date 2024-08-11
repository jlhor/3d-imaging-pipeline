# Workflow Overview

The primary workflow consists of three separate steps:

1. [Setting up a new project](#setup-project)
2. [Segmentation with Stardist3D](#segmentation-with-stardist3d)
3. [Signel cell data extraction](#single-cell-data-extraction)

*For a step-by-step walkthrough of a demo dataset, follow the instructions [here](doc/demo_instructions.md)*

## Setup project

This step will create a new project directory within the `3d-imaging-pipeline` root directory, and is the only step of the entire workflow that requires launching an interactive job on the HPC, where the Conda environment can be activated to execute the `setup_project.py` script.

1. Log into the HPC server via a terminal
2. Use `conda activate <env_name>` to activate the Conda virtual environment
3. Navigate to the `3d-imaging-pipeline` directory where the Python scripts are located
4. Run `python -m setup_project '<new_project_name>'`
   
A new project directory will be generated from the above step, comprising the following elements:

```
|- 3d-imaging-pipeline
   |- Projects
      |- <new_project_name>
         |- input            <- copy image file(s) here
         |- output           <- output will be generated here
         |- models           <- copy custom Stardist model specific for the project here
         |- temp             <- temporary/intermediate files will be created here.
                                (Automatically cleaned up upon completion.)

         |--- segmentation_<new_project_name>.yaml   <- configuration file for segmentation module
         |--- extraction_<new_project_name>.yaml     <- configuration file for data extraction module.

|--- script_segmentation_<new_project_name>.sh       <- bash script for launching Slurm jobs.
|--- script_extraction_<new_project_name>.sh       <- bash script for launching Slurm jobs.

```

5. Exit the interactive job and proceed to configure for the segmentation/data extraction modules.

## Segmentation with Stardist3D

This module requires a custom model to be first trained using [StarDist3D](https://github.com/stardist/stardist). Please follow the [example](https://github.com/stardist/stardist/tree/main/examples/3D) tutorial using `jupyter-lab` as written by the Stardist development team as a starting point.

The training step can be performed on a local workstation equipped with CUDA-compatible GPU and configured with a Conda environment installed with the Stardist package, by following the instructions from the [Stardist Conda environment setup guide](https://github.com/CSBDeep/CSBDeep/tree/main/extras#conda-environment) *and* the additional steps as laid out in the [Stardist installation guide](https://github.com/stardist/stardist?tab=readme-ov-file#installation).

Generally, we find that a few small cropped regions from a representative image (from the same batch of datasets) containing a total of 200-300 cells are sufficient to train a robust segmentation model.

### Segmentation workflow: Overview

The segmentation module is executed in three steps:
1. a GPU-intensive prediction step that runs on a GPU node to predict the object probabilities and distances to object boundaries.   
2. a CPU-intensive non-maximum suppression (NMS) step that culls overlapping candidates
3. a CPU-intensive labeling step that generates the final output containing the segmentation labels of single cells

Typically, a single GPU node is sufficient for processing an entire full-sized image within 30-40 minutes during the prediction step. The subsequent CPU-intensive NMS and labeling steps split the dataset into multiple smaller blocks and leverage a large number of compute nodes to process the data in parallel.

### Segmentation tutorial

1. Once the model has been trained, copy the output model directory to `3d-imaging-pipeline/Projects/<project_name>/models` directory
2. Copy the image file containing the segmentation channel to `3d-imaging-pipeline/Projects/<project_name>/input`
   (Currently, both `.ims` and `.tif` formats are supported)
3. Configure the `segmentation_<project_name>.yaml` file with a text editor.   
   - Specify the `InputImage` name contained within the `input` sub-directory, and the `InputChannel` with an integer corresponding to the channel order within the image (first channel is `0`, and so on). If the image contains only one channel (e.g. `.tif` file), use `0`.
   - Specify the `ModelName` contained within the `models` sub-directory   
   - Configure the HPC resources under the `DASK` section based on the specific cluster and infrastructure available. Note that `gpu_type` and `cpu_type` vary between institutions, and should be modified as such to request for the exact type of nodes available.   
   - For details on the configuration parameters, see the [instructions](./configuration_hpc_prediction.md) here.

4. Navigate to the `3d-imaging-pipeline` root folder containing the `run_segmentation.py` Python script and the `script_segmentation_<project_name>.sh` shell script.

> [!NOTE]
> The `.sh` script contains an additional command `export OMP_NUM_THREADS=14` to specify the number of threads to be used in the OpenMP multi-processing step utilized by the Stardist non-maximum suppression (NMS) step.   
> This is critical for accelerating the NMS processing step within each compute node, and can be configured as needed to suit the type of compute nodes available.

5. Execute the command `sbatch script_segmentation_<project_name>.sh` to start a Slurm job for the segmentation module.
   - The log and error files will be output as `seg_<project_name>.out` and `seg_<project_name>.err` respectively.
6. Upon completion of the script, the output `<project_name>_prediction.h5` file containing the segmented labels of the single cells will be generated in the `output` sub-directory for downstream processing and analysis.

> [!NOTE]
> The output `prediction.h5` file is a 32-bit array that often contains many more cells than the maximum bit-depth of a 16-bit image (65535) and should not be converted into a 16-bit image as all the segmented cell objects above the max bit-depth value will be lost.   
> It is important that the subsequent processing and analysis steps do not inadvertently convert the array to a lower bit-depth (e.g. 8 or 16-bit).

## Single cell data extraction

The data extraction module follows the segmentation module by accessing the segmented labels to extract individual image channel information for single cells. This requires the `<project_name>_prediction.h5` label file generated in the previous segmentation step.

### Data extraction: Overview

1. Individual cell patch (a small 3D block) that extends beyond the boundaries of 3D cell masks will be extracted from individual image channels based on the cell mask positions.
2. This module assumes that the segmented labels are based on nuclear staining of the cells, and thus additional masks are also generated to extend beyond the nuclear mask to generate a whole cell mask as well as a membrane/cytoplasmic mask using morphological dilation and erosion filters. Four layers of cell masks are generated:   
   
   | Mask | Description |
   | -- | -- |
   | `nuclear` | the original masks predicted by the segmentation step |
   | `cell` | dilation of the original `nuclear` masks to encompass the membrane/cytoplasmic region of the cells |
   | `eroded` | slight erosion of the original `nuclear` masks. Can be useful in cases when the predicted nuclear masks overextend into the cytoplasmic region of the cells |
   | `membrane` | membrane/cytoplasmic mask resulting from a subtraction of the  `eroded` mask from the dilated `cell` mask. Provides a more accurate quantification of the membrane protein staining |   
   
3. The output array will comprise the mean intensity value of each channel for the four separate masks for each cell.
   - This is calculated by dividing the sum of all masked voxel intensities by the sum of all mask voxels
4. The coordinates for each cell in `(x, y, z)` are also calculated and exported as the coordinate parameters. Conversion between `image` coordinates and `world` coordinates (in µm) can be specified in the configuration file.
  
### Data extraction tutorial

1. If the input image file is not the same as the input image used for segmentation, copy the image file(s) where the channels will be used for data extraction into the `input` sub-directory.
   - Note that more than one images can be used for data extraction.

> [!NOTE]
> Currently only `.ims` format is supported in this module. Support for OME-TIFF format is  under development. We recommend converting image datasets into `.ims` format using the free [Imaris File Converter](https://imaris.oxinst.com/microscopy-imaging-software-free-trial#file-converter) tool if needed.

2. Configure the `extraction_<project_name>.yaml` file contained within the `Projects/<project_name>` directory.
   - Provide the `InputImage` as a list of image file names.
   - `Channels` specify the channels to be extracted (first channel is `0`).   
     If more than one input images are used, provide the channel numbers in multiple sub-lists.
   - `ChannelNames` can be provided as a list of strings indicating the channel names, or use `auto` to automatically extract the channel names from the `.ims` metadata
   - `PredictionFileName` should point to the `prediction.h5` label file generated from the segmentation step in the `output` sub-directory. No change is needed if following directly from the previous segmentation step.
   - `OutputFilePrefix` will specify the prefix of the **final output array** file name.
   - `CellCoordinates` can be set to `world` if the positions in µm distance is desired.
   - `VoxelDimensions` can be provided as a list in `[x, y, z]` dimensions for conversion into `world` coordinates. Set to `auto` to automatically retrieve from the `.ims` metadata.
   - `OutputCSV` can be set to `True` if export to individual `.csv` dataframes is desired.
   - Configure the HPC resources under the `DASK` section based on the specific cluster and infrastructure available. Note that `gpu_type` and `cpu_type` vary between institutions, and should be modified as such to request for the exact type of nodes available.   
   - For details on the configuration parameters, see the [instructions](/doc/configuration_hpc_data_extraction.md) here.
  
4. Navigate to the `3d-imaging-pipeline` root folder containing the `run_extraction.py` Python script and the `script_extraction_<project_name>.sh` shell script.
5. Execute the command `sbatch script_extraction_<project_name>.sh` to start a Slurm job for the data extraction module.
   - The log and error files will be output as `ext_<project_name>.out` and `ext_<project_name>.err` respectively.
6. Upon completion of the script, the output `output-array.h5` file containing the extracted information of the single cells will be generated in the `output` sub-directory for subsequent analysis.

### Output array

The output array will be exported in a HDF5 container (`.h5`) with multiple 2D `pandas` dataframe that can be retrived with the appropriate keys using the following command: `pandas.read_hdf('output_array.h5', key='nuclear')`

The keys for the datasets are as follows:
| Key | Description |
| -- | -- |
| `positions` |  X, Y, Z coordinates for each cell (Conversion from image coordinates to world coordinates can be set in the configuration file) |
| `nuclear` | Mean intensity extracted from each channel based on the nuclear mask of the cell |
| `membrane` | Mean intensity extracted from each channel based on the membrane/cytoplasmic mask of the cell |
| `cell` | Mean intensity extracted from each channel based on the nuclear + membrane/cytoplasmic mask of the cell |
| `eroded` | Mean intensity extracted from each channel based on the eroded nuclear mask of the cell |

To export the cell coordinates as world coordinates (in µm scale), set `CellCoordinates` to `world` and `VoxelDimensions` can be defined as an `[X, Y, Z]` list, or set to `auto` to automatically retrieve the voxel dimensions from the image file (Imaris only).   

To export the output arrays as `.csv` format, set `OutputCSV` to `True` in the configuration file.


