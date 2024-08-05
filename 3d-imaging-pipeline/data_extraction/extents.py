# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:45:17 2024

@author: JL
"""


import numpy as np

from tqdm import tqdm

from dask.distributed import Client, as_completed

from .utils.cluster_setup import create_cluster
from tiling import read_coords_tile

def get_extents_setup(tile_labels, batch_size, config):
    
    dask_config = { 'cluster_size'          : config['DASK']['EXTENTS']['cluster_size'],
                    'processes'          : config['DASK']['EXTENTS']['processes'], 
                    'cores'              : config['DASK']['EXTENTS']['cores'], 
                    'memory'             : config['DASK']['EXTENTS']['memory'],
                    'walltime'           : config['DASK']['EXTENTS']['walltime'], 
                    'cpu_type'           : config['DASK']['EXTENTS']['cpu_type']}
    
    cluster_mode = config['DASK']['cluster_mode']
    
    
    max_labels = max([ max(x) for x in tile_labels if len(x) > 0])
        
    print('Max labels: ', max_labels)
    
    ### get extents
    future_batch = np.arange(0, max_labels, batch_size)
    if not future_batch[-1] == max_labels:
        future_batch = np.append(future_batch, max_labels)
    
    print('Initiating cluster')
    print(f'future_batch: {len(future_batch)}')
    
    CLUSTER_SIZE = dask_config['cluster_size']
    cluster=create_cluster(mode=cluster_mode, config=dask_config)
    client = Client(cluster)
    
    print(cluster.job_script())
    cluster.scale(CLUSTER_SIZE)
    
    futures = []
    futures_tasks = list(np.arange(len(future_batch)-1))
    
    extents_full = np.zeros((max_labels, 9))
    
    
    if len(futures_tasks) >= CLUSTER_SIZE:
        init_batch = CLUSTER_SIZE
    else:
        init_batch = len(futures_tasks)
    
    print('Submitting jobs')
    for i in range(init_batch):
        t = futures_tasks.pop(0)
        future = client.submit(get_extents, t, future_batch, tile_labels, config) 
        futures.append(future)
    
    
    futures_seq = as_completed(futures)
    
    for future in tqdm(futures_seq, total=len(future_batch)-1, desc='Processing: extents'):
        f, e = future.result()
        
        print(f, e[:3])
        
        extents_full[future_batch[f]:future_batch[f+1], :] = e[...]
        
        #client.run(trim_memory)
        
        if len(futures_tasks) > 0:
            tid = futures_tasks.pop(0)
            future_new = client.submit(get_extents, tid, future_batch, tile_labels, config)
            futures_seq.add(future_new)
        
    
    client.shutdown()
    
    return extents_full
    
    
    
def get_extents(f, future_batch, tile_labels, config):
    
    temp_dir = config['temp_dir'].name
    coords_file = config['TempCoordsFile']
    
    coords_path = f'{temp_dir}/{coords_file}'
    
    extents = np.zeros((future_batch[f+1]-future_batch[f], 9))
    
    for j, k in enumerate(range(future_batch[f], future_batch[f+1])):
        
        if k > 0:
        ## check for tiles with label       
            tiles_to_search = [ i for i, x in enumerate(tile_labels) if k in x ]
            
            coords = []
            
            if len(tiles_to_search) > 0:      
                for tile in tiles_to_search:
                    
                    tile_coords, pos = read_coords_tile(tile, k, coords_path)
                    
                    if len(tile_coords) > 0:
                    # trim out empty rows
                        coords.append(tile_coords[:pos[1]])
                
                if len(coords) > 0:
                    coords = np.concatenate(coords, axis=0)  
                    
            if len(coords) > 0:
                extents[j] = [np.min(coords[:, 0]), np.max(coords[:, 0]),
                             np.min(coords[:, 1]), np.max(coords[:, 1]),
                             np.min(coords[:, 2]), np.max(coords[:, 2]),
                             np.max(coords[:, 0])-np.min(coords[:, 0]),
                             np.max(coords[:, 1])-np.min(coords[:, 1]),
                             np.max(coords[:, 2])-np.min(coords[:, 2])]
                
    return f, extents

