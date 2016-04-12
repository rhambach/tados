# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:23:26 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt
import logging
from transmission import *
from zemax_dde_link import *

def __test_intensity_footprint(hDDE):  
  
  # raytrace parameters
  image_surface = 22;
  wavenum  = 4;
  def raytrace(params, pupil_points):      # local function for raytrace
    x,y   = params;      
    px,py = pupil_points.T;                # shape (nPoints,)
    ret = hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
    vigcode = ret[:,[6]]<>0;               # include vignetting by shifting ray outside image
    return ret[:,[0,1]]+image_size*vigcode;# return (x,y) coordinates in image space

  # field sampling (octagonal fiber)
  xx,yy=cartesian_sampling(3,3,rmax=2);   # low: 7x7, high: 11x11
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


def __test_angular_distribution(hDDE):
  # raytrace parameters
  image_surface = 22;
  wavenum  = 4;
  def raytrace(params, field_points):      # local function for raytrace
    px,py= params;      
    x,y = field_points.T;                  # shape (nPoints,)
    ret = hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
    vigcode = ret[:,[6]]<>0;               # include vignetting by shifting ray outside image
    return ret[:,[3,4]]+image_size*vigcode;# return (kx,ky) direction cosine in image space

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
  image_size=(0.5,0.5);  # NA_max
  img = PolarImageDetector(rmax=0.3,nrings=100);
  dbg = CheckTriangulationDetector(ref_area=8*np.tan(np.pi/8)); # area of octagon with inner radius 1
  
  # run Transmission calculation
  T = Transmission(pupil_sampling,field_sampling,raytrace,[dbg,img]);
  lthresh = 0.5*image_size[1];  
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
    filename= os.path.realpath('./tests/pupil_slicer.ZMX');
    hDDE.load(filename);
    __test_intensity_footprint(hDDE);
    __test_angular_distribution(hDDE);
    