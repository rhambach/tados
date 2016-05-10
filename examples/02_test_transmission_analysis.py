# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:23:26 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt
import logging

from PyOptics.illumination.point_in_triangle import point_in_triangle
from PyOptics.illumination.transmission import *
from PyOptics.zemax.dde_link import *


def __test_intensity_footprint(hDDE):  
  
  # raytrace parameters
  image_surface = 20;                      # 20: without spider, 21: with spider aperture
  wavenum  = 4;
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


  # field sampling (octagonal fiber)
  # sampling should be rational approx. of tan(pi/8), using continued fractions:
  # approx tan(pi/2) = [0;2,2,2,2,....] ~ 1/2, 2/5, 5/12, 12/29, 29/70, 70/169
  # results in samplings: (alwoys denominator-1): 4,11,28,69,168
  xx,yy=cartesian_sampling(4,4,rmax=2);   # low: 4x4, high: 11x11
  ind = (np.abs(xx)<=1) & (np.abs(yy)<=1) & \
              (np.abs(xx+yy)<=np.sqrt(2)) & (np.abs(xx-yy)<=np.sqrt(2));
  field_sampling = np.vstack((xx[ind],yy[ind])).T;       # size (nFieldPoints,2)
  plt.figure(); plt.title("field sampling (normalized coordinates)");
  plt.plot(xx[ind].flat,yy[ind].flat,'.')
  plt.xlabel('x'); plt.ylabel('y');
  
  # pupil sampling (circular, adaptive mesh)
  px,py=fibonacci_sampling_with_circular_boundary(50,20) # low: (50,20), high: (200,50)
  pupil_sampling = np.vstack((px,py)).T;                 # size (nPoints,2)
  
  # set up image detector
  image_size=(0.2,0.05);  # [mm]
  img = RectImageDetector(extent=image_size,pixels=(201,401));
  dbg = CheckTriangulationDetector();
  
  # run Transmission calculation
  T = Transmission(field_sampling,pupil_sampling,raytrace,[dbg,img]);
  lthresh = 0.5*image_size[1];  
  T.total_transmission(lthresh)
  
  # plotting
  img.show();

  # analyze img detector in detail (left and right sight separately)
  img.show(fMask = lambda x,y: np.logical_or(2*x+y<0, x>0.07))  
  img.show(fMask = lambda x,y: 2*x+y>=0)
  

def __test_angular_distribution(hDDE):
  # raytrace parameters
  image_surface = 20;                      # 20: without spider, 21: with spider aperture
  wavenum  = 4;
  def raytrace(params, field_points):      # local function for raytrace
    px,py = params;      
    x,y   = field_points.T;                # shape (nPoints,)
    ret   = hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
    error = ret[:,0];
    vigcode=ret[:,[1]];     
    kxky  = ret[:,[5,6]];                  # return (kx,ky) direction cosine in image space    
    kxky[error<>0]=np.nan;                 # rays that could not be traced
    return kxky;                             

  # field sampling (octagonal fiber, adaptive mesh)
  # sampling should be rational approx. of tan(pi/8), using continued fractions:
  # approx tan(pi/2) = [0;2,2,2,2,....] ~ 1/2, 2/5, 5/12, 12/29, 29/70, 70/169
  # results in samplings: (alwoys denominator-1): 4,11,28,69,168
  xx,yy=cartesian_sampling(28,28,rmax=2);  # low: 11x11, high: 69x69
  ind = (np.abs(xx)<=1) & (np.abs(yy)<=1) & \
              (np.abs(xx+yy)<=np.sqrt(2)) & (np.abs(xx-yy)<=np.sqrt(2));
  field_sampling = np.vstack((xx[ind],yy[ind])).T;       # size (nFieldPoints,2)
  
  # pupil sampling (cartesian grid with circular boundary)
  px,py=cartesian_sampling(3,3,rmax=1)     # low: 7x7, high: 11x11
  pupil_sampling = np.vstack((px,py)).T;                 # size (nPoints,2)
  plt.figure(); plt.title("pupil sampling (normalized coordinates)");
  plt.plot(px.flat,py.flat,'.')
  plt.xlabel('x'); plt.ylabel('y');
  
  # set up image detector (in angular space)
  NAmax = 0.3;
  img = PolarImageDetector(rmax=NAmax,nrings=100);
  dbg = CheckTriangulationDetector(ref_area=8*np.tan(np.pi/8)); # area of octagon with inner radius 1
  
  # run Transmission calculation
  T = Transmission(pupil_sampling,field_sampling,raytrace,[dbg,img]);
  lthresh = 0.5*NAmax;  
  T.total_transmission(lthresh)
  
  # plotting
  img.show();


if __name__ == '__main__':
  import os as os
  import sys as sys
  logging.basicConfig(level=logging.INFO);
  
  with DDElinkHandler() as hDDE:
  
    ln = hDDE.link;
    # load example file
    #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
    #                        'Cooke 40 degree field.zmx')
    filename= os.path.realpath('../tests/zemax/pupil_slicer.ZMX');
    hDDE.load(filename);
    __test_intensity_footprint(hDDE);
    __test_angular_distribution(hDDE);
    