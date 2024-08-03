# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:44:06 2024

@author: horj2

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import numpy as np

from utils.utils import calculate_grid_pos, read_yaml

from stardist.models import StarDist3D

from stardist.rays3d import rays_from_json

import h5py
import tempfile

from segmentation.preprocess import preprocessing_linear
from segmentation.nms_custom import nms
from segmentation.labeling import labeling



######################################

def run(config_path):
    config = read_yaml(config_path)
    
    model_name = config['ModelName']
    model_basedir = os.path.join(config['ProjectPath'], config['ModelDir'])
    output_prefix = os.path.join(config['ProjectPath'],config['OutputDir'],  config['OutputPrefix'])
    block_shape = config['BlockShape']
    input_prefix = os.path.join(config['ProjectPath'],config['InputDir'],  config['InputPrefix'])
    #temp_dir = os.path.join(config['ProjectPath'],config['TempDir'])
    
    temp_dir = tempfile.TemporaryDirectory(prefix='.', dir=os.path.join(config['ProjectPath'],config['TempDir']))
    
    config['temp_dir'] = temp_dir
    
    PREPROCESS = config['Preprocess']
    RUN_NMS = config['RunNMS']
    
    dask_config = { 'processes'          : config['DASK']['processes'], 
                    'cores'              : config['DASK']['cores'], 
                    'memory'             : config['DASK']['memory'],
                    'walltime'           : config['DASK']['walltime'], 
                    'cpu_type'           : config['DASK']['cpu_type']}
    

    ## Load from h5
    print('Loading data from h5')
    data_path = f'{input_prefix}_data.h5'
    print(data_path)

    with h5py.File(data_path, mode='r') as data:
        #points = data['points']
        #dist = data['dist']
        prob = data['prob']
        shape_inst = data['shape_inst']
        shape_inst = tuple(shape_inst)
        inds_global = np.arange(len(prob))
    
    
    ## block_shape[0] is unused, default set to 0
    batch_shape = [shape_inst[0],block_shape[1], block_shape[2]]
    
    print('Batch shape: ', batch_shape)
    coords = calculate_grid_pos(shape_inst, batch_size=batch_shape)
    
    ########################
    
    cluster_mode = 'SLURM'
    
    #########################
    
    model = StarDist3D(None, name=model_name, basedir=model_basedir)
    rays = rays_from_json(model.config.rays_json)
    nms_thresh  = model.thresholds.nms
    
    
    ################
    
    #client.upload_file(script_path)
    #.run(_prep)
    
    #inds_global = np.arange(len(prob))
    
    print('Stardist Model name: ', model_name)
    print('Run PREPROCESS: ', PREPROCESS)
    print('Run NMS: ', RUN_NMS)
    
    #PREPROCESS_LINEAR = True
    
    ### pre-processing
    
    if PREPROCESS:
        print('Pre-processing: linear mode')
        preprocess_out = preprocessing_linear(data_path=data_path,
                                        inds_global=inds_global,
                                        coords=coords,
                                        config=config,)

    if RUN_NMS:
        print('Run NMS')
        out_inds = nms(data_path=data_path,
                       params=[shape_inst, rays, nms_thresh],
                       coords=coords,
                       config=config,
                       cluster_config=[dask_config, cluster_mode],
                       )
    else:
        print('Loading NMS data')
        out_inds = np.load(f'{output_prefix}_out_inds.npy')
    
    print('Labeling:')
    labeling_out = labeling(data_path=data_path,
                            params=[shape_inst, out_inds, rays],
                            config=config,
                            cluster_config=[dask_config, cluster_mode])
    
    
    ###############################################################
    ###############################################################

    
    print('Removing temporary directory')
    temp_dir.cleanup()

    print('End of script')


if __name__ == '__main__':
    config_path = sys.argv[1]
    run(config_path)
    