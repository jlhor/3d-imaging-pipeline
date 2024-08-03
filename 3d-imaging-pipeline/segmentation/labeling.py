# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:28:09 2024

@author: JL
"""

import os
import numpy as np

import h5py
import zarr
from numcodecs import Blosc

from dask.distributed import Client, as_completed
from .utils.cluster_setup import create_cluster
from draw_lib import dask_polyhedron_to_label

from tqdm import tqdm

import ctypes


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def labeling(data_path, params, config, cluster_config):
    
    shape_inst, out_inds, rays = params
    output_prefix = os.path.join(config['ProjectPath'],config['OutputDir'],  config['OutputPrefix'])
    zarr_chunks = config['ZarrChunks']
    dask_config, cluster_mode = cluster_config
    BATCH_SIZE = config['LabelingBatchSize']
    
    with h5py.File(data_path, mode='r') as data:
        print('Loading dataset')
        pointsc = np.array(data['points'])
        distc = np.array(data['dist'])
        probc = np.array(data['prob'])
        
        pointsc = pointsc[out_inds]
        distc = distc[out_inds]
        probc = probc[out_inds]
        print('Dataset loaded')

    faces = rays.faces
    verts = rays.vertices
    
    label = np.arange(1, len(pointsc) + 1)
    ind = np.argsort(probc)[::-1]
    pointsc = pointsc[ind]
    distc = distc[ind]
    label = label[ind]

    z_order = np.arange(0, len(pointsc))
    
    ###############################################

    print('Initializing result')

    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

    result = zarr.open(f'{output_prefix}_result_full.zarr', 'w', shape=shape_inst, chunks=[shape_inst[0],zarr_chunks[1],zarr_chunks[2]], dtype=np.int32, compressor=compressor)
    result[...] = 0
    
    batch_idx = np.arange(0, len(pointsc), BATCH_SIZE)
    if not batch_idx[-1] == len(pointsc):
        batch_idx = np.append(batch_idx, len(pointsc))
    
    CLUSTER_SIZE = config['DASK']['cluster_size_labeling']
    
    if CLUSTER_SIZE > len(batch_idx):
        CLUSTER_SIZE = len(batch_idx)
        
    cluster=create_cluster(mode=cluster_mode, config=dask_config)
    if cluster_mode == 'SLURM':
        print(cluster.job_script())
        cluster.scale(CLUSTER_SIZE)

    client = Client(cluster)
        

    futures = []
    futures_tasks = list(np.arange(len(batch_idx)-1))
    
    if len(futures_tasks) >= CLUSTER_SIZE:
        init_batch = CLUSTER_SIZE
    else:
        init_batch = len(futures_tasks)
    
    for i in range(init_batch):
        t = futures_tasks.pop(0)
        
        dc = distc[batch_idx[t]:batch_idx[t+1]]
        ptc = pointsc[batch_idx[t]:batch_idx[t+1]]

        f = client.submit(dask_polyhedron_to_label, t, batch_idx, dc, ptc, verts, faces, shape_inst, verbose=True)
        futures.append(f)
        

    label_new = np.append(label, 0)

    sorter = np.argsort(label_new)
    
    futures_seq = as_completed(futures)

    for future in tqdm(futures_seq, total=len(batch_idx)-1, desc='Processing jobs: Labeling'):
        f, o = future.result()
        batch_label = label[batch_idx[f]:batch_idx[f+1]]
        
        for fi, pc in enumerate(o):
            
            ## check if pixels is already labeled
            #fill_idx = np.where(result[pc[:,0], pc[:,1], pc[:,2]] == 0)[0]  
            
            ## check z order
            existing_labels = result[pc[:,0], pc[:,1], pc[:,2]]

            z_order = sorter[np.searchsorted(label_new, existing_labels, sorter=sorter)]
            to_draw = (batch_idx[f] + fi) < z_order
            
            
            result[pc[to_draw][:,0], pc[to_draw][:,1], pc[to_draw][:,2]] = batch_label[fi]
        
        future.release()
        #client.run(trim_memory)
        
        if len(futures_tasks) > 0:
            tid = futures_tasks.pop(0)
            
            dc = distc[batch_idx[tid]:batch_idx[tid+1]]
            ptc = pointsc[batch_idx[tid]:batch_idx[tid+1]]
            
            future_new = client.submit(dask_polyhedron_to_label, tid, batch_idx, dc, ptc, verts, faces, shape_inst, verbose=True)
            
            #futures.append(f)
            futures_seq.add(future_new)
        

    ###
    print('Converting array to numpy')
    result_arr = np.array(result, dtype=result.dtype)
    
    print('Saving array to H5')
    
    with h5py.File(f'{output_prefix}_result.h5', 'w') as h5_result:
        h5_result.create_dataset('data', data=result_arr, chunks=True, compression="gzip")   
    

    return True

