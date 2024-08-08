##  Configuration for HPC prediction step:

:eight_spoked_asterisk: indicates parameters that may need to be changed between projects. All other parameters can be left as is.


| Settings | | Description |
| -- | -- | -- | 
| `ProjectPath` | :eight_spoked_asterisk: |  Root directory for the project on the HPC |   
| `InputDir` |  | Input sub-directory inside project path | 
| `ProbabilitiesFileName` | :eight_spoked_asterisk: |  The input `probabilities.h5` generated from the GPU step (this file should be placed inside the `InputDir` directory)   |
|  |  |  |
| `OutputDir` | |  Output sub-directory inside project path   |
| `PredictionFileName`  | :eight_spoked_asterisk: | The **output** array containing the predicted cell labels that will be generated in the `OutputDir` directory |
| | | | 
| `TempDir` | |  Temporary directory where intermediate files are generated. Cleaned up automatically if the script runs to completion. |
| | | | 
| `ModelDir` | |  The model sub-directory inside project path |
| `ModelName` | :eight_spoked_asterisk: |  The name of the model placed inside `ModelDir` |
| | | | 
| `Preprocess` |  | Pre-processing steps by partitioning cell coordinates to sub-blocks. Must be `True` on first run |
| `RunNMS` |  |  Non-maximal suppression steps. Must be `True` on first run |
| |  | All intermediate files from the Pre-processing and NMS steps are saved in `TempDir`. If the run is interrupted in between steps, one or both of the above steps can be set to `False` to re-use the previously generated intermediate files already stored in `TempDir`. The intermediate files will be cleaned up automatically upon completion of the script. |
| | | | 
| `{task}_BatchSize` |  |  Batch size for each {task} to be handled by each node |
| `BlockShape ` | |  Block shape in (`z, y, x`) for pre-processing steps. The entire `z` dimension of the image will be used automatically, so set to `0` here. |
| `ZarrChunks` |  | Chunks in (`z, y, x`) for intermediate files. The entire `z` dimension of the image will be used automatically, so set to `0` here. | 
| | | | 
| DASK | | | 
| `cluster_size` | :eight_spoked_asterisk: |  The number of nodes to request |
| `cores` | :eight_spoked_asterisk: | CPUs per node |
| `processes` | |  Number of worker (set to 1) |
| `memory` | :eight_spoked_asterisk: |  Memory per node |
| `walltime` | :eight_spoked_asterisk: | Wall time for each node |
| `cpu_type` | :eight_spoked_asterisk: | `nodetype` specified by the `--constraint` argument |
  
