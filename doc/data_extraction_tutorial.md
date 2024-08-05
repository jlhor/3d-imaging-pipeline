# Data Extraction

This workflow requires predicted segmentation labels (`prediction.h5`) to be first generated from the segmentation steps.

Individual cell patch (a small 3D block) will be extracted from the individual image channels based on the positions of the segmented labels, and additional masks will be generated to yield a nuclear vs membrane/cytoplasmic masks through morphological filters. The latter step is designed to work with Stardist segmentation trained on a nuclear stain, where an dilation from the nuclear mask is required to cover the membrane/cytoplasmic region of the lymphoid-shaped cells. Other segmentation algorithms e.g. Cellpose should be modified as appropriate.

The output array will comprise the mean intensity values of four separate masks: 1. nuclear, 2. cell (nucleus + membrane/cytoplasm), 3. eroded (erosion from nuclear mask), and 4. membrane/cytoplasmic.

## Instructions

1. If proceeding from the segmentation step, the `prediction.h5` file should already be in the `output` directory of the project on HPC.   
  Otherwise, copy the `prediction.h5` file to the `output` sub-directory of the project on the HPC.
2. Copy the input Imaris `.ims` files to the `input` sub-directory if not already done. Multiple Imaris files (of the same image) can be used.
3. Configure the `hpc_data_extraction_template.yaml` file, by specifying the location of the `prediction.h5` file and the directory containing the input Imaris images.   
   For extracting from more than one Imaris files, set the `InputImage` parameter as a list of multiple filenames, and the `Channels` parameter as a list of sub-lists, with each sub-list containing the exact channels to be extracted from each image file. The order of all sub-lists here will define the order of the extracted channels in the final output array.
   Set the number of cluster (compute nodes) to request as needed.   
   The number of CPUs and memory may need to be adjusted depending on the HPC infrastructure of the institution.   
   For details on the configuration parameters, see the [instructions](/doc/configuration_hpc_data_extraction.md) here.
5. Configure the `jobscript_data_extraction.sh` to run `python hpc_data_extraction.py hpc_data_extraction_template.yaml` on a SLURM scheduler.
6. The `output_array.h5` file containing the extracted channels will be generated in `output` sub-directory for subsequent analyses.