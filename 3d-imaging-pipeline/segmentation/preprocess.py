# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:27:20 2024

@author: JL
"""

import numpy as np
import h5py
from tqdm import tqdm
from .utils.utils import get_coord_blocks

def preprocessing_linear(data_path, inds_global, coords, config):
    
    temp_dir = config['temp_dir'].name
    
    with h5py.File(data_path, mode='r') as data:
        points = np.array(data['points'])
        dist = np.array(data['dist'])
        prob = np.array(data['prob'])
        shape_inst = data['shape_inst']
        shape_inst = tuple(shape_inst)
    
        for i in tqdm(range(len(coords[0])), total=len(coords[0]), desc='Pre-processing'):
            
            filtered_id = get_coord_blocks(i, points, shape_inst, coords, overlap=[0,32,32])
            
            probf = prob[filtered_id]
            distf = dist[filtered_id]
            pointsf = points[filtered_id]
            
            inds_original = inds_global[filtered_id]
            _sorted = np.argsort(probf)[::-1]
            probi = probf[_sorted]
            disti = distf[_sorted]
            pointsi = pointsf[_sorted]
            inds_original = inds_original[_sorted]
            
            np.save(f'{temp_dir}/probi_{i}', probi)
            np.save(f'{temp_dir}/disti_{i}', disti)
            np.save(f'{temp_dir}/pointsi_{i}', pointsi)
            np.save(f'{temp_dir}/inds_original_{i}', inds_original)
        
    return True
