# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:43:35 2024

@author: JL
"""

import numpy as np

import h5py
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from dask.distributed import Client, as_completed

from .utils.cluster_setup import create_cluster


def read_coords_tile(t, label, coords_path):
    
    zarr_f = zarr.open(coords_path, mode='r')
    open_ds = zarr_f[f'tile_{t}/data']
    open_meta = zarr_f[f'tile_{t}/metadata']
    
    ## find label index
    label_idx = np.where(open_meta[:,0] == label)[0]
    
    if len(label_idx) > 0:
        return open_ds[label_idx[0]], open_meta[label_idx[0]].reshape(2,)
    else:
        return [], []


def save_coords_tile(i, grp, label_path, block_xy, block_shape, block_size):
    
    with h5py.File(label_path, 'r') as label_file:
        
        predicted = label_file['data']
        
        start_coord = block_xy[i]
        end_coord = start_coord + block_shape
        
        ## stretch end coord to max image size
        for n in range(3):
            if end_coord[n] == block_size[n]*block_shape[n]:
                if not end_coord[n] == predicted.shape[n]:
                    end_coord[n] = predicted.shape[n]
    
        temp_predicted = predicted[start_coord[0]:end_coord[0],
                                   start_coord[1]:end_coord[1],
                                   start_coord[2]:end_coord[2]]
        
        coords = np.where(temp_predicted > 0)
        coords = np.array(coords).T
        
        # return empty list if nothing is found
        unique_labels = []
        
        if len(coords) > 0:
            mask_values = temp_predicted[coords[:,0], coords[:,1], coords[:,2]]
            
            unique_labels = np.unique(mask_values)
            
            coords += start_coord
            
            data = []
            grp_meta = np.zeros((len(unique_labels), 2), dtype=int)


            for idx, j in enumerate(unique_labels):
                unique_mask = np.where(mask_values == j)[0]
                filtered_coords=coords[unique_mask]
                
                data.append(filtered_coords)
                grp_meta[idx] = [j, len(filtered_coords)]
            
            max_arr_length = max([ len(x) for x in data])
            
            #print(f'Writing tile {i}', flush=True)
            
            ## append each array to the maximum length
            data = [ np.concatenate((x, np.zeros(((max_arr_length-len(x)), 3)))) for x in data ]
            data = np.array(data).astype(int)
                    
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
            
            ## potential BUG: dask tries to write to grp that has already been written
            ## check if dataset already exists, if not - skip
            if not 'data' in grp.keys():
                ds = grp.create_dataset('data', shape=data.shape, chunks=(200,1000,3), dtype=int, compressor=compressor)
                ds_meta = grp.create_dataset('metadata', shape=grp_meta.shape, chunks=(1000,2), dtype=int, compressor=compressor)
                ds[...] = data[...]
                ds_meta[...] = grp_meta[...]
            else:
                print(f'grp {i} already exists. Skipping.')

    return i, unique_labels



def coords_tiling(config, coords_params, mode='read'):

    
    dask_config = { 'processes'          : config['DASK']['TILING']['processes'], 
                    'cores'              : config['DASK']['TILING']['cores'], 
                    'memory'             : config['DASK']['TILING']['memory'],
                    'walltime'           : config['DASK']['TILING']['walltime'], 
                    'cpu_type'           : config['DASK']['TILING']['cpu_type']}
    
    cluster_mode = config['DASK']['cluster_mode']
    
    
    label_path = coords_params['label_path']
    tile_id = coords_params['tile_id']
    block_xy = coords_params['block_xy'] 
    batch_shape = coords_params['batch_shape'] 
    block_size = coords_params['block_size'] 
    
    coords_file = config['TempCoordsFile']
    
    temp_dir = config['temp_dir'].name
    
    if mode == 'write':
        zarr_grp = zarr.open_group(f'{temp_dir}/{coords_file}', mode='w')
        grps = []
        for i in range(len(tile_id)):
            grps.append(zarr_grp.create_group(f'tile_{i}'))
        
        print('Initiating cluster')
        
        CLUSTER_SIZE = dask_config['cluster_size_tiling']
        cluster=create_cluster(mode=cluster_mode, config=dask_config)
        client = Client(cluster)
        
        print(cluster.job_script())
        cluster.scale(CLUSTER_SIZE)
        
        futures = []
        futures_tasks = list(np.arange(len(tile_id)))
        
        tile_labels = [ [] for _ in range(len(tile_id)) ]
        
        if len(futures_tasks) >= CLUSTER_SIZE:
            init_batch = CLUSTER_SIZE
        else:
            init_batch = len(futures_tasks)
        
        print('Submitting jobs')
        for i in range(init_batch):
            t = futures_tasks.pop(0)
            print(f'Submitting {t}')
            future = client.submit(save_coords_tile, t, grps[t], label_path, block_xy, batch_shape, block_size) #, z_coords) #, coords, max_labels, mask_values)
            futures.append(future)
        
        futures_seq = as_completed(futures)
        
        for future in tqdm(futures_seq, total=len(tile_id), desc='Processing: writing coords'):
            f, tl = future.result()
            
            tile_labels[f] = tl
            
            future.release()
            
            #client.run(trim_memory)
            
            if len(futures_tasks) > 0:
                tid = futures_tasks.pop(0)
                print(f'Submitting {tid}')
                future_new = client.submit(save_coords_tile, tid, grps[tid], label_path, block_xy, batch_shape, block_size) #, z_coords) #, coords, max_labels, mask_values)
                #futures.append(f)
                futures_seq.add(future_new)
            
        #max_labels = max([ max(x) for x in tile_labels if len(x) > 0])
        
        client.shutdown()
    
    elif mode == 'read':
        
        zarr_grp = zarr.open_group(f'{temp_dir}/{coords_file}', mode='r')
        
        #unsorted_grps = list(zarr_grp.group_keys())
        
        tile_labels = [ [] for _ in range(len(tile_id)) ] 
        
        for i in range(len(tile_id)):
            try:
                grp = zarr_grp[f'tile_{i}']
                if len(grp['metadata'][:,0]) > 0:
                    temp_labels = grp['metadata'][:,0]
                    temp_labels = temp_labels[temp_labels > 0]
                    
                    print(f'tile {i}, unique: {len(np.unique(temp_labels))}, labels: {len(temp_labels)}')
                    
                    tile_labels[i] = temp_labels
            except Exception:
                print(f'Error reading tile {i}')
           
    return tile_labels
