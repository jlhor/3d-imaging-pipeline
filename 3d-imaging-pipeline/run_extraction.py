# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:11:18 2024

@author: JL
"""

import sys
import os
import numpy as np
import pandas as pd

import h5py

import tempfile

from utils import read_yaml
import math

import ctypes

from data_extraction.tiling import coords_tiling
from data_extraction.extents import get_extents_setup, get_coordinates
from data_extraction.extract import process_tiles_and_get_values


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def initialization(config):

    output_dir = os.path.join(config['ProjectPath'],config['OutputDir'])
    block_shape = config['BlockShape']
    
    prediction_filename = config['PredictionFileName']
    
    print('Loading data from h5')
    label_path = f'{output_dir}/{prediction_filename}'
    print(label_path)
    

    with h5py.File(label_path, 'r') as label_file:    
        predicted = label_file['data']
        shape_inst = label_file['shape_inst']
        shape_inst = tuple(shape_inst)
        
        batch_shape = [shape_inst[0],block_shape[1], block_shape[2]]
    
        block_size = []
        for n in range(3):
            block_size.append(int(predicted.shape[n] // batch_shape[n]))
    
    tile_id = np.argwhere(np.arange(np.prod(block_size)).reshape(block_size) >= 0)
    
    block_xy = tile_id * batch_shape
    block_xy = block_xy.astype(int)
    
    init_params = {'label_path'  : label_path,
                   'tile_id'     : tile_id,
                   'block_xy'    : block_xy,
                   'batch_shape' : batch_shape,
                   'block_size'  : block_size}
    
    
    return init_params


def get_patch_info(extents_full, max_labels):
    
    max_extents = np.array([np.max(extents_full[:,6], axis=0),
                           np.max(extents_full[:,7], axis=0),
                           np.max(extents_full[:,8], axis=0)])

    block_size = np.ceil(max_extents * 1.4).astype(int)

    offsets = np.floor(block_size * 0.5).astype(int)

    xy_dim = np.ceil(math.sqrt(max_labels)).astype(int)

    tile_id = np.argwhere(np.arange(xy_dim**2).reshape(xy_dim, xy_dim) >= 0)

    block_xy = tile_id * block_size[1:]
    block_xy = np.concatenate((np.zeros((len(block_xy))).reshape(-1,1), block_xy), axis=1)
    block_xy = block_xy.astype(int)
    
    patch_params = {'max_extents'  : max_extents,
                   'block_size'     : block_size,
                   'offsets' : offsets,
                   'xy_dim'    : xy_dim,
                   'tile_id' : tile_id,
                   'block_xy'  : block_xy}
    
    return patch_params
    


def run(config_path):

    config = read_yaml(config_path)
    
    output_dir = os.path.join(config['ProjectPath'],config['OutputDir'])

    temp_dir = tempfile.TemporaryDirectory(prefix='.', dir=os.path.join(config['ProjectPath'],config['TempDir']))
    config['temp_dir'] = temp_dir
    
    
    BATCH_SIZE = config['ExtentBatchSize']
    
    print('Starting script')
    
    #write_mode=False
    save_coords_mode= config['SaveCoordsMode']
    save_extent_mode= config['SaveExtentMode']
    
    print('save_coords_mode: ', save_coords_mode)
    print('save_extent_mode: ', save_extent_mode)
    
    ## Load from h5
    
    init_params = initialization(config)
    
    if save_coords_mode:    
        
        tile_labels = coords_tiling(config,
                                    coords_params=init_params, 
                                    mode='write')
        
        print('Completed writing coords')
    
    else:
        tile_labels = coords_tiling(config,
                                    coords_params=init_params, 
                                    mode='read')
        
        print('Finished reading coords')

    
    if save_extent_mode:
        ## get extents
        extents_full = get_extents_setup(tile_labels, BATCH_SIZE, config)
        
        print('Saving files')
        
        np.save(f'{temp_dir}/extents', extents_full)
            
    else:
        extents_full = np.load(f'{temp_dir}/extents.npy')
    
    
    cell_coords = get_coordinates(extents_full, config)

    ###################################################
    
    max_labels = max([ max(x) for x in tile_labels if len(x) > 0])
    
    patch_params = get_patch_info(extents_full, max_labels)
    
    extracted_array, metadata = process_tiles_and_get_values(tile_labels, max_labels, patch_params, config)
    
    img_dims, vxl_dims, ch_names = metadata
    

    output_prefix = config['OutputFilePrefix']
    return_csv = config['OutputCSV']
    coords_type = config['CellCoordinates']
    vxl_type = config['VoxelDimensions']
    
    
    ## override voxel dimensions with user defined list
    if isinstance(vxl_type, list):
        if len(vxl_type) == 3:
            vxl_dims = np.array(vxl_type)[::-1].astype(float)  # reverse from xyz to zyx
            print(f'Voxel dimensions: user defined {vxl_dims[::-1]}')
    elif vxl_type == 'auto':
        print(f'Voxel dimensions: auto from image file {vxl_dims[::-1]}')
    else:
        vxl_dims = np.array([1.0, 1.0, 1.0]).astype(float)
        print(f'Voxel dimensions data not provided: using default {vxl_dims}')
    
    output_path = f'{output_dir}/{output_prefix}'
    
    
    print('Converting data to h5')
    print(f'Output location: {output_path}')
    
    layers = ['nuclear', 'cell', 'eroded', 'membrane']  
    
    ## set attributes with h5py
    with h5py.File(output_path, mode='w') as f:
        f.create_group('Info')
        f['Info'].attrs['Image Dimensions'] = img_dims
        f['Info'].attrs['Voxel Dimensions'] = vxl_dims
        f['Info'].attrs['Channel Names'] = ch_names
        
    ## export individual layers
    for i, layer in enumerate(layers):
        pd_arr = pd.DataFrame(extracted_array[i], columns=ch_names)
        pd_arr.to_hdf(f'{output_path}.h5', key=layer, mode='a')
        
        if return_csv:
            pd_arr.to_csv(f'{output_path}_{layer}.csv')

    ## export coordinates
    if coords_type == 'world':
        cell_coords *= vxl_dims
    
    coords_arr = pd.DataFrame(cell_coords, columns=['Z', 'Y', 'X'])
    coords_arr.to_hdf(f'{output_path}.h5', key='positions', mode='a')
    if return_csv:
        coords_arr.to_csv(f'{output_path}_positions.csv')
        
    
    print('Removing temporary files')
    temp_dir.cleanup()
    
    print('End of script')
    
if __name__ == '__main__':
    config_path = sys.argv[1]
    run(config_path)
    
