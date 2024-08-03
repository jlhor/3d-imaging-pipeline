# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:42:44 2024

@author: horj2
"""

import numpy as np
import yaml
import h5py

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    
def ceiling_division(n, d):
    return -(n // -d)


def calculate_grid_pos(img_shape, batch_size=[128,]*3, origin=[0,0,0]):
    """
    Calculates the grid positions based on image size and the user-defined level block size

    Input:
        img: input image array
        batch_size: block size of x, y, z
    
    Output:
        List of block information for this level
        [0, 1]: x start and end indices
        [2, 3]: y start and end indices
        [4, 5]: z start and end indices
        [6, 7, 8]: total number of blocks across x, y, z
        
    """
    if not isinstance(img_shape, tuple):
        img_shape = img_shape.shape
    

    x_start = np.arange(origin[0], img_shape[0], batch_size[0])
    x_end = np.concatenate([x_start[1:], [img_shape[0]]])

    y_start = np.arange(origin[1], img_shape[1], batch_size[1])
    y_end = np.concatenate([y_start[1:], [img_shape[1]]])
    
    z_start = np.arange(origin[2], img_shape[2], batch_size[2])
    z_end = np.concatenate([z_start[1:], [img_shape[2]]])

    x_start_final = np.tile(x_start, len(y_start)*len(z_start))
    x_end_final = np.tile(x_end, len(y_end)*len(z_end))
    
    y_start_final = np.tile(np.repeat(y_start, len(x_start)), len(z_start))
    y_end_final = np.tile(np.repeat(y_end, len(x_end)), len(z_end))
    
    z_start_final = np.repeat(z_start, len(x_start)*len(y_start))
    z_end_final = np.repeat(z_end, len(x_end)*len(y_end))
    
    return [x_start_final, x_end_final, y_start_final, y_end_final, z_start_final, z_end_final, len(x_start), len(y_start), len(z_start)]

def get_coord_blocks(i, input_data, img_shape, coord_pos, overlap):

    start_pos = [coord_pos[0][i], coord_pos[2][i], coord_pos[4][i]]
    end_pos = [coord_pos[1][i], coord_pos[3][i], coord_pos[5][i]]

    start = [0, 0, 0]
    end = [0, 0, 0]
    
    for n in range(3):
        if start_pos[n] > 0:
            if start_pos[n]-overlap[n] > 0:
                start[n] = start_pos[n]-overlap[n]
        else:
            start[n] = 0

        if end_pos[n] > img_shape[n]:
            end[n] = img_shape[n]
        else:
            if end_pos[n]+overlap[n] < img_shape[n]:
                end[n] = end_pos[n]+overlap[n]
            else:
                end[n] = img_shape[n]



    filter_id = np.where((input_data[:,0] > start[0]) & (input_data[:,0] <= end[0]) &
                         (input_data[:,1] > start[1]) & (input_data[:,1] <= end[1]) &
                         (input_data[:,2] > start[2]) & (input_data[:,2] <= end[2]))


    return filter_id[0]


def convert_to_h5(output_path, data, chunk_rows=50000):
    
    dist, prob, points, shape_inst = data

    with h5py.File(output_path, mode='w') as f:
        ds_dist = f.create_dataset("dist", shape=dist.shape, chunks=(chunk_rows, dist.shape[1]), dtype=dist.dtype)
        ds_dist[...] = dist[...]
        
        ds_points = f.create_dataset("points", shape=points.shape, chunks=(chunk_rows, points.shape[1]), dtype=points.dtype)
        ds_points[...] = points[...]
        
        ds_prob = f.create_dataset("prob", shape=prob.shape, chunks=(chunk_rows,), dtype=prob.dtype)
        ds_prob[...] = prob[...]
        
        ds_shape = f.create_dataset("shape_inst", shape=(len(shape_inst),), dtype=shape_inst.dtype)
        ds_shape[...]  = shape_inst[...]
        
    return output_path
