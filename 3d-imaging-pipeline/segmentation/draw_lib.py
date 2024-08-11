# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import ConvexHull

from numba import jit


def py_polyhedron_bbox(curr_dist, curr_center, verts, shape_inst):
    

    out = curr_center + np.repeat(curr_dist.reshape(-1,1), 3, axis=1)*verts


    bbox = [ np.floor(max(0, min(out[:,0]))), np.ceil(min(shape_inst[0], max(out[:,0]))),
             np.floor(max(0, min(out[:,1]))), np.ceil(min(shape_inst[1], max(out[:,1]))),
             np.floor(max(0, min(out[:,2]))), np.ceil(min(shape_inst[2], max(out[:,2])))]
    
    ### check out of bounds
    
    bbox = np.array(bbox, dtype=np.int32)

    return bbox, out


def py_halfspace_convex(pv):
    hull = ConvexHull(pv)
    
    return hull.equations

@jit(nopython=True)
def py_halfspace_kernel(pv, faces):
    
    halfspace = []
    
    for i, face in enumerate(faces):
        
        hs = np.zeros((3,3), dtype=np.float64)
        for n in range(3):
            hs[n] = pv[face[n]]
        
        P = hs[1] - hs[0]
        Q = hs[2] - hs[0]
        
        N = np.array([ -(P[1]*Q[2]-P[2]*Q[1]),
              -(P[2]*Q[0]-P[0]*Q[2]),
              -(P[0]*Q[1]-P[1]*Q[0])])
        
        N = np.append(N, -(hs[0,0]*N[0]+hs[0,1]*N[1]+hs[0,2]*N[2]))
        
        halfspace.append(N)
        
    return halfspace
        
@jit
def py_point_in_halfspaces(c, hs):
    
    z,y,x = c
    
    test = np.sum(hs*np.array([z,y,x,1], dtype=np.float64), axis=1)
    
    if np.max(test) > 0:
        return 0
    else:
        return 1
    

@jit(nopython=True)
def py_inside_polyhedron(cur_pos, curr_center, pv, faces):
    
    curr_center = curr_center.astype(np.float64)
    
    for i, face in enumerate(faces):
        A = pv[face[0]]
        B = pv[face[1]]
        C = pv[face[2]]
    
        if py_inside_tetrahedron(cur_pos, curr_center, A, B, C):
            return 1
    return 0
    
    

@jit(nopython=True)
def py_inside_tetrahedron(cur_pos, R, A, B, C):
    
    if (py_inside_halfspace(cur_pos, A, B, C) and \
            py_inside_halfspace(cur_pos, R, B, A) and \
            py_inside_halfspace(cur_pos, R, C, B) and \
            py_inside_halfspace(cur_pos, R, A, C)):
        return 1
    else:
        return 0

@jit(nopython=True)
def py_inside_halfspace(cur_pos, A, B, C):

    mat = np.zeros((3,3), dtype=np.float64)
    mat[0] = B - A
    mat[1] = C - A
    mat[2] = cur_pos - A
    
    det = np.linalg.det(mat)
    
    if det >= 0:
        return 1
    else:
        return 0

@jit(nopython=True)
def check_pixels(bbox, hs_kernel, hs_convex, curr_center, pv, faces):
    
    inside_coords = []
    
    for z in range(bbox[0], bbox[1]):
        for y in range(bbox[2], bbox[3]):
            for x in range(bbox[4], bbox[5]):
                cur_pos = np.array([z,y,x], dtype=np.float64)
                if (py_point_in_halfspaces(cur_pos,hs_kernel) or \
                  (py_point_in_halfspaces(cur_pos,hs_convex) and \
                   py_inside_polyhedron(cur_pos, curr_center, pv, faces))):
                    inside_coords.append(cur_pos)  
                    
    return inside_coords



def dask_polyhedron_to_label(b, batch, dist, points, verts, faces, 
                           shape_inst, verbose):

    n_polys=len(points)

    output_polys = []
    
    for i in range(n_polys):
        
        inside_coords = []
        
        curr_dist = dist[i]
        curr_center = points[i]
        
        bbox, pv = py_polyhedron_bbox(curr_dist, curr_center, verts, shape_inst)
        
        hs_convex = py_halfspace_convex(pv)
        hs_kernel = np.array(py_halfspace_kernel(pv, faces), dtype=np.float64)
        
        inside_coords = check_pixels(bbox, hs_kernel, hs_convex, curr_center, pv, faces)

        poly_coords = np.array(inside_coords, dtype=np.int32)
        poly_coords = np.unique(poly_coords, axis=0)
        

        output_polys.append(poly_coords)

    return b, output_polys
