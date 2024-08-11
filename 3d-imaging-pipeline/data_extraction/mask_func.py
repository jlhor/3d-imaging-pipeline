# -*- coding: utf-8 -*-

import numpy as np

from skimage.morphology import dilation, erosion, disk

def func_dilation(img, footprint=6):
    out = np.zeros_like(img)
    for z in range(img.shape[0]):
        out[z] = dilation(img[z], footprint=disk(footprint))        
    return out



def func_erosion(img, footprint=3):
    out = np.zeros_like(img)
    for z in range(img.shape[0]):
        out[z] = erosion(img[z], footprint=disk(footprint))
        
    return out

                  
                                                                 
def func_subtraction(img1, img2):
    return img1-img2


def mask_functions(f, block_size, read_block, footprints):

    d_fp, e_fp = footprints
    
    out_blocks = np.zeros((4, read_block.shape[0], read_block.shape[1], read_block.shape[2]), dtype=read_block.dtype)
    
    out_blocks[0,:,:,:] = read_block[:,:,:]
    
    ## dilation -> write
    out_blocks[1,:,:,:] = func_dilation(read_block, footprint=d_fp)
    
    ## erosion -> write
    out_blocks[2,:,:,:] = func_erosion(read_block, footprint=e_fp)
    
    ## subtraction -> write
    out_blocks[3,:,:,:] = func_subtraction(out_blocks[1,:,:,:], out_blocks[2,:,:,:])
    
    return out_blocks



def func_multiply_single(im1_block, im2_block, im1_ch=None, im2_ch=None):
    
    if im1_ch is not None:
        im1_arr = im1_block[im1_ch,:,:,:].astype(float)
    else:
        im1_arr = im1_block.astype(float)
        
    if im2_ch is not None:  
        im2_arr = im2_block[im2_ch,:,:,:].astype(float)
    else:
        im2_arr = im2_block.astype(float)
    
    out = im1_arr * im2_arr
    
    return im1_arr, out


def func_multiply(f, block_size, im1_block, im2_block, im1_ch=None, im2_ch=None):
    

    im1_arr = im1_block[im1_ch,:,:,:].astype(float)
    im2_arr = im2_block[im2_ch,:,:,:].astype(float)
    
    out = im1_arr * im2_arr
    
    return f, im1_arr, out