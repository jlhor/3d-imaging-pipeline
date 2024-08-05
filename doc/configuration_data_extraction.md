##  Configuration for HPC data extraction step:
   

| Settings | description |
| -- | -- |
| `ProjectPath` | Root directory for the project on the HPC |   
| `InputDir` | Input sub-directory inside project path | 
| `InputImage` | The input Imaris image files (`.ims`) to extract data from. To extract from more than one Imaris file, use a list e.g., `["Image1a.ims", "Image1b.ims"]`   |
| `Channels` | Input channels to extract data from, in the exact order in the output array. For extracting from more than one Imaris file, define as sub-lists within a list. |
|  |  |
| `OutputDir` | Output sub-directory inside project path   |
| `OutputFile` | The **output** array containing all the extracted values from each channel (default is mean intensity, user-defined functions can be used to extract other types of information) |
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
| `MaskGaussianSigma` |  The sigma value for Gaussian filtering to be applied after mask generation to create a weighted gradient that emphasizes the core of the masks. Default is `1.0` pixel |
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
  
