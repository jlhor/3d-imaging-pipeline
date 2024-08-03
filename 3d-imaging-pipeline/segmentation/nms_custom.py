# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:23:43 2024

@author: JL
"""

import os
import numpy as np
import h5py
from tqdm import tqdm

from dask.distributed import Client, as_completed

from .utils.cluster_setup import create_cluster
from .utils.utils import get_coord_blocks

from stardist.nms import non_maximum_suppression_3d_inds

import ctypes

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def nms_dask_batch(b, idx, coords, _shape_inst, rays, nms_thresh, overlap, overlap_post, config):
    
    temp_dir = config['temp_dir'].name
    
    inds_batch = []
    #inds_original_batch = []
    
    for i in idx:
    
        probi = np.load(f'{temp_dir}/probi_{i}.npy')
        disti = np.load(f'{temp_dir}/disti_{i}.npy')
        pointsi = np.load(f'{temp_dir}/pointsi_{i}.npy')
        #inds_original = np.load('./temp/inds_original_{i}.npy')
        
        inds = non_maximum_suppression_3d_inds(disti, pointsi, rays=rays, scores=probi, thresh=nms_thresh, use_kdtree = True, verbose=True)
        
        inds_batch.append(inds)

    return b, inds_batch


def nms(data_path, params, coords, config, cluster_config):
    
    shape_inst, rays, nms_thresh = params
    dask_config, cluster_mode = cluster_config
    temp_dir = config['temp_dir'].name
    output_prefix = os.path.join(config['ProjectPath'],config['OutputDir'],  config['OutputPrefix'])
    BATCH_SIZE = config['NMSBatchSize']
    
    with h5py.File(data_path, mode='r') as data:
        points = np.array(data['points'])
    

    print('NMS tiling:')
    nms_out = [ [] for x in range(len(coords[0])) ]
    
    cluster=create_cluster(mode=cluster_mode, config=dask_config)
    client = Client(cluster)

    batch_idx = np.arange(0, len(coords[0]), BATCH_SIZE)
    if not batch_idx[-1] == len(coords[0]):
        batch_idx = np.append(batch_idx, len(coords[0]))
    
    CLUSTER_SIZE = dask_config['cluster_size_NMS']
    
    if CLUSTER_SIZE > len(batch_idx):
        CLUSTER_SIZE = len(batch_idx)
    
    
    if cluster_mode == 'SLURM':
        print(cluster.job_script())
        cluster.scale(CLUSTER_SIZE)
    
    futures = []
    futures_tasks = list(np.arange(len(batch_idx)-1))
    
    if len(futures_tasks) >= CLUSTER_SIZE:
        init_batch = CLUSTER_SIZE
    else:
        init_batch = len(futures_tasks)
    
    #for i in tqdm(range(100), total=100, desc='Submitting jobs'):
    for i in range(init_batch):
        t = futures_tasks.pop(0)
        
        #print(f'Batch futures: {batch_idx[t]} to {batch_idx[t+1]}')
    
        f = client.submit(nms_dask_batch, t, np.arange(batch_idx[t], batch_idx[t+1]), coords, shape_inst, rays, nms_thresh,
                          overlap=[0,32,32], overlap_post=[0,16,16], config=config)
        futures.append(f)
        
    futures_seq = as_completed(futures)
        
    for future in tqdm(futures_seq, total=len(batch_idx)-1, desc='Processing jobs: NMS'):
        fi, inds_n = future.result()
        
        f_batch = np.arange(batch_idx[fi], batch_idx[fi+1])
        
        for f, fb in enumerate(f_batch):
            i_o = np.load(f'{temp_dir}/inds_original_{fb}.npy')
            i_n = inds_n[f]
            points_inds = points[i_o[i_n]]
            #print('points_inds:')
            #print(points_inds[:5])
            trimmed_id = get_coord_blocks(fb, points_inds, shape_inst, coords, overlap=[0,16,16])
            
            #print(fb, len(i_n))
            nms_out[fb] = i_o[i_n][trimmed_id]
        
        future.release()
        #client.run(trim_memory)
        
        if len(futures_tasks) > 0:
            tid = futures_tasks.pop(0)
            future_new = client.submit(nms_dask_batch, tid, np.arange(batch_idx[tid], batch_idx[tid+1]), coords, shape_inst, rays, nms_thresh,
                              overlap=[0,32,32], overlap_post=[0,16,16], config=config)
            #futures.append(f)
            futures_seq.add(future_new)
    
    client.shutdown()
    
    nms_out_combined = [ x for y in nms_out for x in y]
    nms_out_combined = np.array(nms_out_combined)
    out_inds = np.unique(nms_out_combined)
    
    np.save(f'{output_prefix}_out_inds', out_inds)
    
    return out_inds


