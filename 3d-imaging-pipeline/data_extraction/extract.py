# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:45:57 2024

@author: JL
"""

import os
import numpy as np

import h5py

from tqdm import tqdm

from dask.distributed import Client, as_completed

from skimage.filters import gaussian

from data_extraction.mask_func import func_multiply, mask_functions
from data_extraction.tiling import read_coords_tile
from utils.cluster_setup import create_cluster

def get_channel_names(img_path, channels):
    cch = [ x for y in channels for x in y] ## channels from each image
    cimg = [ [i] * len(x) for i, x in enumerate(channels)]
    cimg = [ x for y in cimg for x in y ]  ## image_id from each image
    
    ch_names = [None] * len(cimg)
    
    for img_idx in range(len(channels)):
        with h5py.File(img_path[img_idx], 'r') as img_file:
            for c, ch in enumerate(cch):
                if cimg[c] == img_idx: 
                    ch_name = img_file[f'DataSetInfo/Channel {ch}'].attrs['Name']
                    ch_name = [ str(s, encoding='UTF-8') for s in ch_name ]
                    ch_names[c] = ''.join(ch_name)
                    
    return ch_names


def rename_duplicate_channels(channels):
    for i, ch in enumerate(channels):
        if channels.count(ch) > 1:
            dup_idx = [ i for i,x in enumerate(channels) if x == ch]
            for i, d in enumerate(dup_idx):
                if i > 0:
                    channels[d] = f'{ch}_{i}'
                    
    return channels

def get_dimensions(img_path):
    
    dims = ['Z', 'Y', 'X']
    
    img_dimension = np.zeros((3), dtype=int)
    vxl_dimension = np.zeros((3), dtype=float)
    with h5py.File(img_path[0], 'r') as img_file:
        for n, dim in enumerate(dims):
            cur_dim = img_file[f'DataSetInfo/Image'].attrs[f'{dim}']
            cur_dim = ''.join(str(s, encoding='UTF-8') for s in cur_dim)
            img_dimension[n] = int(cur_dim)
        
            cur_vxl_max = img_file[f'DataSetInfo/Image'].attrs[f'ExtMax{2-n}']
            cur_vxl_max = ''.join(str(s, encoding='UTF-8') for s in cur_vxl_max)
            
            cur_vxl_min = img_file[f'DataSetInfo/Image'].attrs[f'ExtMin{2-n}']
            cur_vxl_min = ''.join(str(s, encoding='UTF-8') for s in cur_vxl_min)
            
            vxl_dimension[n] = (float(cur_vxl_max) - float(cur_vxl_min)) / float(cur_dim)
        
    return img_dimension, vxl_dimension


def extract_values(ch, mask, batch_size, block_size):

    patch_id = np.arange(batch_size)
    
    val = []
    
    for i, j in enumerate(patch_id):
        a = np.sum(ch[:,:,block_size[2]*i:block_size[2]*(i+1)]) \
            / np.sum(mask[:,:,block_size[2]*i:block_size[2]*(i+1)])
        
        val.append(a)
        
    return val


def extract_values_by_channels(f, batch_size, block_size, im1_block, im2_block, channels=None, radial_params=None):
    
    mask_channels, context_channels = channels
    
    
    val_c = np.zeros((len(mask_channels), batch_size, len(context_channels)), dtype=float)
    
    for mc in tqdm(mask_channels, total=len(mask_channels), desc=f'Row {f}'):
        for cc in context_channels:
            f, im1, out = func_multiply(f, block_size, im1_block, im2_block, im1_ch = mc, im2_ch = cc)

            val = extract_values(out, im1, batch_size, block_size)
            
            val_c[mc, :, cc] = np.array(val)[...]
            

    return val_c




def extract_contexts_filtered(f, future_rows, patch_params, filter_id, tile_labels, image_path, channels, batch_size, config):

    temp_dir = config['temp_dir'].name
    coords_path = f'{temp_dir}/coords.zarr'
    
    mask_dilation = config['MaskDilation']
    mask_erosion = config['MaskErosion']
    mask_sigma = config['MaskGaussianSigma']
    
    block_size = patch_params['block_size']
    offsets = patch_params['offsets']
    
    
    ## get channel numbers from each image
    block_ch = list(np.arange(len([ x for y in channels for x in y])))  ## channel id for contexts_block
    cch = [ x for y in channels for x in y] ## channels from each image
    cimg = [ [i] * len(x) for i, x in enumerate(channels)]
    cimg = [ x for y in cimg for x in y ]  ## image_id from each image

    radial_params = None
    
    contexts_block = np.zeros((len(block_ch), block_size[0], block_size[1], block_size[2]*batch_size), dtype=np.int64)
    masks_block = np.zeros((block_size[0], block_size[1], block_size[2]*batch_size), dtype=np.int64)                          


    for i, fi in enumerate(range(future_rows[f], future_rows[f+1])):
        
        gid = filter_id[fi]
        
        if gid == 0:
            continue
        
        tiles_to_search = [ j for j, x in enumerate(tile_labels) if gid in x ]
        
        coords = []
        
        if len(tiles_to_search) > 0:      
            for tile in tiles_to_search:
                
                tile_coords, pos = read_coords_tile(tile, gid, coords_path)
                
                if len(tile_coords) > 0:
                # trim out empty rows
                    coords.append(tile_coords[:pos[1]])
            
            if len(coords) > 0:
                coords = np.concatenate(coords, axis=0)
            else:
                continue  ### not sure if needed?
        
            coords_center = []
            for ndim in range(3):
                coords_center.append(np.median(coords[:,ndim]))
            coords_center = np.array(coords_center).astype(np.int64)
            
            x_offset = np.array([0, 0, block_size[2]*i])
            
            coords_arr2 = (coords - coords_center + offsets + x_offset).astype(np.int64)

            masks_block[coords_arr2[:,0],
                        coords_arr2[:,1],
                        coords_arr2[:,2]] = 1
                             
        
            coords_origin = (coords_center - offsets).astype(np.int64)
            coords_end = (coords_origin + block_size).astype(np.int64)
        
            ## check if coords_origin is < 0, which will fail to extract the context
            origin_pads = np.array([0, 0, 0], dtype=np.int64)
            
            if np.sum(coords_origin < 0) > 0:
                origin_pads[coords_origin < 0] = -coords_origin[coords_origin < 0]  ## origin pad
                coords_origin[coords_origin < 0] = 0  ## set negative origin to 0
            
            for img_idx in range(len(channels)):
                with h5py.File(image_path[img_idx], 'r') as img_file:
                    for c, ch in enumerate(cch):
                        if cimg[c] == img_idx:
                            img = img_file[f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {ch}/Data']
                            
                            temp_image = img[coords_origin[0]:coords_end[0],
                                             coords_origin[1]:coords_end[1],
                                             coords_origin[2]:coords_end[2]]
                        
                            temp_pads = []
                            for ndim in range(3):
                                # determine padding (origin_pad, difference between block size and temp_image shape minus origin_pad)
                                temp_pads.append(tuple((origin_pads[ndim], block_size[ndim]-temp_image.shape[ndim]-origin_pads[ndim])))
                            temp_image = np.pad(temp_image, temp_pads)
                        
                            contexts_block[block_ch[c], :, :, block_size[2]*i:block_size[2]*(i+1)] = temp_image[...]
                            
                            
                        
    ### generate masks
    masks_block = mask_functions(f, block_size, masks_block, footprints=[mask_dilation, mask_erosion])
    
    masks_block_gaus = np.zeros_like(masks_block, dtype=float)
    ## gaussian filtering masks
    if mask_sigma > 0:
        for b in range(masks_block.shape[0]):
            for z in range(masks_block.shape[1]):
                masks_block_gaus[b, z, ...] = gaussian(masks_block[b][z], sigma=mask_sigma)
    else:
        masks_block_gaus = masks_block
    
    
    mask_channels = np.arange(masks_block_gaus.shape[0])
    context_channels = np.arange(contexts_block.shape[0])
    
    ## extract values
    val_c = extract_values_by_channels(f, batch_size, block_size, 
                                       im1_block=masks_block_gaus,
                                       im2_block=contexts_block,
                                       channels=[mask_channels, context_channels],
                                       radial_params = radial_params 
                                       )
    
 
    return f, val_c

def process_tiles_and_get_values(tile_labels, max_labels, patch_params, config):
    
    dask_config = { 'cluster_size'       : config['DASK']['EXTRACTION']['cluster_size'], 
                    'processes'          : config['DASK']['EXTRACTION']['processes'], 
                    'cores'              : config['DASK']['EXTRACTION']['cores'], 
                    'memory'             : config['DASK']['EXTRACTION']['memory'],
                    'walltime'           : config['DASK']['EXTRACTION']['walltime'], 
                    'cpu_type'           : config['DASK']['EXTRACTION']['cpu_type']}
    
    cluster_mode = config['DASK']['cluster_mode']
    
    ims_path = config['InputImage']
    
    if not isinstance(ims_path, list):
        ims_path = [ ims_path ]
    
    input_dir = os.path.join(config['ProjectPath'],config['InputDir'])
    img_path = [ f'{input_dir}/{x}' for x in ims_path ]
    
    
    

    cell_id = np.arange(max_labels)    
    context_channels = config['Channels']
    
    ch_name_type = config['ChannelNames']
     
    block_ch = list(np.arange(len([ x for y in context_channels for x in y])))
    
    #### get image dimensions and voxel dimensions
    img_dimensions, vxl_dimensions = get_dimensions(img_path)
    
    ### get channel names
    if ch_name_type == 'auto':
        ch_names = get_channel_names(img_path, context_channels)
    else:
        if isinstance(ch_name_type, list):
            ch_names = ch_name_type
    
    ## check for duplicate channel names and rename
    ch_names = rename_duplicate_channels(ch_names)
    
    print(f'Image path: {img_path}')
    print(f'Channels to process: {context_channels}')
    
    EXTRACT_BATCH_SIZE = config['ExtractionBatchSize']
    print(f'Batch size: {EXTRACT_BATCH_SIZE}')
    
    extracted_arr = np.zeros((4, len(cell_id), len(block_ch)), dtype=float)
    
        
    futures = []
    
    future_rows = np.arange(0, max_labels, EXTRACT_BATCH_SIZE)
    if future_rows[-1] < max_labels:
        future_rows = np.append(future_rows, max_labels)
    
    
    CLUSTER_SIZE = dask_config['cluster_size']
    cluster=create_cluster(mode=cluster_mode, config=dask_config)
    client = Client(cluster)
    
    print(cluster.job_script())
    if len(future_rows) < CLUSTER_SIZE:
        CLUSTER_SIZE = len(future_rows)
        
    cluster.scale(CLUSTER_SIZE)
    
    client.wait_for_workers(n_workers=round(CLUSTER_SIZE*0.75))
    
    
    futures = []
    futures_tasks = list(np.arange(len(future_rows)-1))
    
    if len(futures_tasks) >= CLUSTER_SIZE:
        init_batch = CLUSTER_SIZE
    else:
        init_batch = len(futures_tasks)
    

    for i in range(init_batch):
        t = futures_tasks.pop(0)
        future = client.submit(extract_contexts_filtered, t, future_rows, patch_params, cell_id, tile_labels,
                               image_path=img_path,
                               channels=context_channels,
                               batch_size=EXTRACT_BATCH_SIZE,
                               config=config)
                               
        futures.append(future)
    
    futures_seq = as_completed(futures)
    
    for future in tqdm(futures_seq, total=len(future_rows)-1, desc='Processing - extract intensity from each channel'):
        f, c = future.result()    
        
        ## trim out extra rows
        c_len = future_rows[f+1] - future_rows[f]
        if c.shape[1] > c_len:
            c = c[:,:c_len,:]
        
        extracted_arr[:, future_rows[f]:future_rows[f+1], :] = c[...]
        
        future.release()
        
        if len(futures_tasks) > 0:
            tid = futures_tasks.pop(0)
            future_new = client.submit(extract_contexts_filtered, tid, future_rows, patch_params, cell_id, tile_labels,
                                       image_path=img_path,
                                       channels=context_channels,
                                       batch_size=EXTRACT_BATCH_SIZE,
                                       config=config)
            futures_seq.add(future_new)
        

    client.shutdown()
    
    return extracted_arr, [img_dimensions, vxl_dimensions, ch_names]
    
