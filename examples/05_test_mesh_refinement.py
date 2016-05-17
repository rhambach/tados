# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 10:37:13 2016

@author: Hambach
"""

from __future__ import division
import logging
import numpy as np
import matplotlib.pylab as plt

import _set_pkgdir
from PyOptics.illumination.point_in_triangle import point_in_triangle
from PyOptics.illumination.adaptive_mesh import *
from PyOptics.illumination.transmission import *
from PyOptics.zemax import sampling, dde_link


def analyze_transmission(hDDE):  
  # set up ray-trace parameters
  image_surface = 4;
  wavenum  = 1;
  def raytrace(params, pupil_points):      # local function for raytrace
    x,y   = params;      
    px,py = pupil_points.T;                # shape (nPoints,)
    ret   = hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
    error = ret[:,0];
    vigcode= ret[:,[1]];     
    xy    = ret[:,[2,3]];    
    # return (x,y) coordinates in image space    
    xy   += image_size*(vigcode<>0);       # include vignetting by shifting ray outside image
    xy[error<>0]=np.nan;                   # rays that could not be traced
    return xy;                             

  # set up image detector
  image_size = np.asarray((1.1,0.3));
  img = RectImageDetector(extent=image_size,pixels=(600,200));
  dbg = CheckTriangulationDetector();
  detectors=[img,]#dbg]
 
  # field sampling
  xx,yy=sampling.cartesian_sampling(3,3,rmax=.1);  # single point
  for i in xrange(len(xx)):
    x=xx[i]; y=yy[i];
    print("Field point: x=%5.3f, y=%5.3f"%(x,y))
    
    # pupil sampling (circular, adaptive mesh)
    px,py=sampling.fibonacci_sampling_with_circular_boundary(300);
    pupil_sampling = np.vstack((px,py)).T;                 # size (nPoints,2)  
    mapping = lambda(mesh_points): raytrace((x,y),mesh_points);
    Mesh=AdaptiveMesh(pupil_sampling, mapping);

    # iterative refinement of skinny triangles
    rthresh = [1.7,1.5,1.7,2,3,3,4,4];
    Athresh = 0.00001;
    ref_steps =5;
    is_skinny= lambda(simplices): Mesh.find_skinny_triangles(simplices=simplices,rthresh=rthresh[0]);
    Mesh.plot_triangulation(skip_triangle=is_skinny);
    for it in range(ref_steps): 
      Mesh.refine_skinny_triangles(rthresh=rthresh[it],Athresh=Athresh,bPlot=True);
    Mesh.plot_triangulation();
 
    # subdivision of invalid triangles (raytrace failed for some vertices)
    Mesh.refine_invalid_triangles(nDivide=100,bPlot=False);    

    # segmentation of triangles along cutting line
    lthresh = 0.5*image_size[1];
    is_broken = lambda(simplices): Mesh.find_broken_triangles(simplices=simplices,lthresh=lthresh);  
    Mesh.refine_broken_triangles(is_broken,nDivide=100,bPlot=False);

    # update detectors
    broken = Mesh.find_broken_triangles(lthresh=lthresh);
    for d in detectors:
      d.add(Mesh,bSkip=broken);# update detectors

  # plotting
  fig=img.show();
  fig.suptitle('Geometric Image Analysis (%d refinement steps, %d rays)'% (ref_steps,Mesh.domain.shape[0]));
  
  return 1



if __name__ == '__main__':
  import os as os
  logging.basicConfig(level=logging.DEBUG);
   
  with dde_link.DDElinkHandler() as hDDE:
  
    ln = hDDE.link;
    # load example file
    #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
    #                        'Cooke 40 degree field.zmx')
    #filename= os.path.realpath('../tests/zemax/fraunhofer_logo.ZMX');
    filename = os.path.realpath('X:/projekte/1504_surface_reconstruction/zemax/01_fraunhofer_logo.ZMX');    
    hDDE.load(filename);
    analyze_transmission(hDDE);
    