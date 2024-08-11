## Step-by-step demonstration with a demo dataset

Download the demo dataset from here: [demo_dataset.zip](https://drive.google.com/file/d/1VuBGjNKoN7lw3ZUjE39UChV-XclILlqt/view?usp=sharing)

The zip compressed file contains two elements:
1. `demo_image.ims` is an Imaris image file (~900MB) that contains a small, cropped region of a full-sized image file used for analysis in the manuscript. It contains 5 raw, unprocessed channels (TCF-1, PD-1, XCR1-venus, Ki-67 and OT-I.GFP), with the exception being OT-I.GFP that has been corrected for spillover from the XCR1-venus channel.   
   This cropped region also contains a distinct PD-1+ TCF-1+ OT-I-XCR1 cluster and a couple smaller ones.
2. `demo_model` is a custom-trained Stardist3D model based on the Ki-67 nuclear channel of the original image, and will used for segmentating the nuclear channel contained within `demo_image.ims`.

### Instructions

The following steps assume the successful installation of the Conda environment and the cloning of `3d-imaging-pipeline` repository by following the instructions from the [Installation guide](https://github.com/jlhor/3d-imaging-pipeline-dev/blob/main/doc/installation.md) 

### 1. Setup a new project
1. Log in to HPC and launch an interactive job
2. Activate conda environment using the `conda activate <env_name>` command
3. Navigate to the `3d-imaging-pipeline` directory where the Python scripts are located
4. Execute `python -m setup_project 'mouse_LN_demo'`
5. Exit the interactive job as it is no longer required for the subsequent steps.

### 2. Segmentation
1. Copy `demo_model` from the downloaded dataset to `Projects/mouse_LN_demo/models`
2. Copy `demo_image.ims` to `Projects/mouse_LN_demo/input`
3. With a text editor, configure `segmentation_mouse_LN_demo.yaml` contained in the `mouse_LN_demo` directory   
   
Modify the following parameters:
```
InputImage: "demo_image.ims"    ## name of the input image
InputChannel: 3                 ## channel 3 (4th channel) is the Ki-67 nuclear channel

ModelName: "demo_model"         ## name of the Stardist model
```
   
Also modify the `DASK` section as needed, especially the `gpu_type` and `cpu_type` that are specific to the HPC nodes of the facility.

4. Navigate to the `3d-imaging-pipeline` directory.
5. Execute the command `sbatch script_segmentation_mouse_LN_demo.sh` to start a Slurm job on the HPC.
6. Wait until completion of the script.

### 3. Single cell data extraction
1. Using a text editor, configure `extraction_mouse_LN_demo.yaml` as contained in the `mouse_LN_demo` directory   
   
Modify the following parameters:
```
InputImage: [ "demo_image.ims" ]     ## name of the input image (can be more than one image)
Channels: [ [0, 1, 2, 3, 4] ]        ## specify the channel intensities to be extracted.
                                     ## Note that this is a sub-list within a list.

ChannelNames: 'auto'                 ## set to 'auto' to get channel names from the image metadata

OutputFilePrefix: "LN_demo"          ## output file name prefix

CellCoordinates: 'world'             ## get world coordinates in um.
VoxelDimensions: 'auto'              ## set to 'auto' get the voxel spacings from image metadata
OutputCSV: True                      ## set to True to export data as .csv files in addition to the .h5 output array

```

Also modify the `DASK` section as needed, especially the `gpu_type` and `cpu_type` that are specific to the HPC nodes of the facility.   

4. Navigate to the `3d-imaging-pipeline` directory.
5. Execute the command `sbatch script_extraction_mouse_LN_demo.sh` to start a Slurm job on the HPC.
6. Wait until completion of the script.

The output array `LN_demo.h5` as well as the individual `.csv` layers will be generated in the `output` sub-directory. 



