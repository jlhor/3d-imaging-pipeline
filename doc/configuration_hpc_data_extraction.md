##  Configuration for HPC data extraction step:
   

| Settings | description |
| -- | -- |
| `ProjectPath` | Root directory for the project on the HPC |   
| `InputDir` | Input sub-directory inside project path | 
| `InputImage` | The input Imaris image files (`.ims`) to extract data from. To extract from more than one Imaris file, use a list e.g., `["Image1a.ims", "Image1b.ims"]`   |
| `Channels` | Input channels to extract data from, in the exact order in the output array. For extracting from more than one Imaris file, define as sub-lists within a list. |
| `ChannelNames` | Input channel names. Define a list that matches the total number of `Channels` for the channel names to be exported in the output array. **Imaris only:** set to `auto` to automatically retrieve channel names from Imaris datasets |
|  |  |
| `OutputDir` | Output sub-directory inside project path   |
| `OutputFilePrefix` | The name for the **output** array containing all the extracted values from each channel (default is mean intensity, user-defined functions can be used to extract other types of information). Do not include the file extension. |
| | |
| `CellCoordinates` | The output cell coordinate type (if set to `'world'`, coordinates will be scaled with the voxel dimensions (µm), otherwise the coordinates will be given in image positions (voxel coordinates) |
| `VoxelDimensions` | Voxel dimensions in X, Y, Z. Provide in list `[x, y, z]` or **(Imaris only)** set to `auto` to automatically retrieve the voxel dimensions from the Imaris image file |
| `OutputCSV` | Set to `True` for output array to be exported as `.csv` format (as multiple 2D data arrays) as well. |
| | |
| `TempDir` | Temporary directory where intermediate files are generated. Cleaned up automatically if the script runs to completion. |
| | |
| `SaveCoordsMode` | Pre-processing steps by partitioning cell coordinates to sub-blocks. Must be `True` on first run |
| `SaveExtentMode` |  Generating the extents/patch boundaries from each cell. Must be `True` on first run |
| | All intermediate files from the Pre-processing and NMS steps are saved in `TempDir`. If the run is interrupted in between steps, one or both of the above steps can be set to `False` to re-use the previously generated intermediate files already stored in `TempDir`. The intermediate files will be cleaned up automatically upon completion of the script. |
| | |
| `BlockShape ` | Block shape in (`z, y, x`) for pre-processing steps. The entire `z` dimension of the image will be used automatically, so set to `0` here. |
| `{task}_BatchSize` |  Batch size for each {task} to be handled by each node |
| | |
| `MaskDilation` |  The morphological dilation footprint to be applied to the generated masks, which is required to cover the membrane/cytoplasmic regions of the cell if the initial segmentation was performed on nuclear stains. Default is `6.0` pixel. |
| `MaskErosion` |  (Optional) The morphological erosion footprint to be applied to the generated masks, where a slight erosion is applied to the initial (nuclear) mask to yield a slightly larger membrane/cytoplasmic mask, which are then additionally filtered with a Gaussian kernel to create lighter gradient near the edges. The membrane/cytoplasmic mask is created by the subtraction of the eroded mask from the dilated mask. Default is `3.0` pixel. |
| `MaskGaussianSigma` |  The sigma value for Gaussian filtering to be applied after mask generation to create a weighted gradient that emphasizes the core of the masks. Default is `1.0` pixel. Set to `0` to skip this step. |
| | |
| DASK | |
| `cluster_mode` | Currently only `SLURM` is supported. |
| TASKS | The SLURM job requests can be further customized for each task |
| `cluster_size` | The number of nodes to request |
| `cores` | CPUs per node |
| `processes` | Number of worker (set to 1) |
| `memory` | Memory per node |
| `walltime` | Wall time for each node |
| `cpu_type` | `nodetype` specified by the `--constraint` argument |
  
