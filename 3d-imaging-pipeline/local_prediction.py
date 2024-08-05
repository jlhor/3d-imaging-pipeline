# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:32:28 2024

@author: horj2
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import numpy as np


from tifffile import imread
import h5py


from csbdeep.utils import normalize

from stardist import random_label_cmap

from segmentation.stardist_custom import StarDist3D_custom
from utils.utils import read_yaml, ceiling_division, convert_to_h5

## seeding
np.random.seed(6)
lbl_cmap = random_label_cmap()


def run(config_path):
    
    config = read_yaml(config_path)
    
    project_name = config['ProjectName']
    image_name = config['InputImage']
    image_path = os.path.join(config['ProjectPath'], config['InputDir'], image_name)
    input_channel = config['InputChannel']
    model_name = config['ModelName']
    model_basedir = os.path.join(config['ProjectPath'], config['ModelDir'])
    output_prefix = os.path.join(config['ProjectPath'],config['OutputDir'],  config['OutputPrefix'])
    block_shape = config['BlockShape']
    output_filename = config['OutputFileName']
    

    print(f'StarDist3D prediction for {project_name}')
    
    ## check image extension
    img_ext = os.path.splitext(image_name)[-1]
    if img_ext == '.tif':
        print('Loading {image_name}')
        X = [ imread(image_path) ]
    elif img_ext == '.ims':
        with h5py.File(image_path, 'r') as f:
            print(f'Loading {image_name}')
            X = [ np.array(f[f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {input_channel}/Data']) ]
            ds_name = f[f'DataSetInfo/Channel {input_channel}'].attrs['Name'].astype(str)
            ds_name = ''.join(ds_name)
            print(f'Channel {input_channel}: {ds_name}')
    else:
        print("Image extension not recognized. Currently supports '.tif' and '.ims' format only")
        
    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0,1,2)   # normalize channels independently
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        
    print(f'Loading model: {model_name}')
    model = StarDist3D_custom(None, name=model_name, basedir=model_basedir)

    print('Normalizing image')
    img = normalize(X[0], 1, 99.8, axis=axis_norm)
    
    ############################################
    
    axes=None
    _axes = model._normalize_axes(img, axes)
    _axes_net     = model.config.axes
    _permute_axes = model._make_permute_axes(_axes, _axes_net)
    _shape_inst   = tuple(s for s,a in zip(_permute_axes(img).shape, _axes_net) if a != 'C')
    
    ##############################################
    
    
    ## calculate number of tiles needed
    tile_shape = tuple( [ ceiling_division(x, block_shape[i]) for i, x in enumerate(list(img.shape)) ])
    print('Image shape: ', img.shape)
    print('Tile shape: ', tile_shape)
    
    print('Running prediction')
    prob, dist, points = model.predict_instances(img, axes='ZYX', n_tiles=tile_shape)
    
    
    print('Converting data to h5')
    data_path = convert_to_h5(f'{output_prefix}_{output_filename}.h5', 
                  data=[dist, prob, points, np.array(_shape_inst)],
                  chunk_rows=5000)
    
    
    print('End of script')
    

        
if __name__ == '__main__':
    config_path = sys.argv[1]
    run(config_path)
    

