# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:55:26 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt
import logging
import os
import sys as sys
from tolerancing import *
from transmission import RectImageDetector
from zemax_dde_link import *
import  pyzdde.zdde as zdde
import re as _re

def GeometricImageAnalysis(hDDE, testFileName=None):
  """
  perform Geometric Image Analysis in Zemax and return Detector information
    textFileName ... (opt) name of the textfile used for extracting data from Zemax
    
  Returns: (img,params)
    img   ... RectImageDetector object containing detector data
    params... dictionary with parameters of the image analysis
  """  
  data,params = hDDE.zGeometricImageAnalysis(testFileName);
  imgSize = float(params['Image Width'].split()[0]);
  
  # save results in RectImageDetector class
  img = RectImageDetector(extent=(imgSize,imgSize), pixels=data.shape);
  img.intensity = data.copy();
  return img,params
  
def find_quantiles(x,F, thresh):
  thresh = np.atleast_1d(thresh); 
  dx=x[1]-x[0];  
  # offset by half bin width necessary, to have same results if x and F are reversed  
  return np.interp([thresh, F[-1]-thresh],F,x+dx/2); 
 
def compensator_rotz(tol,angle):  
  " apply rotation of slicer about surface nomal by given angle [deg]"
  ln = tol.ln; 
  tol.tilt_decenter_elements(6,8,ztilt=angle,cbComment1="compensator",cbComment2="~compensator");
  surf = tol.get_orig_surface(20);
  assert ln.zGetComment(surf)=="rotate image plane";
  # correct rotation of image plane
  angle_img = 90+np.rad2deg(np.arctan(np.sqrt(2)*np.tan(np.deg2rad(-22.20765+angle))));
  ln.zSetSurfaceParameter(surf,5,angle_img);   #  5: TILT ABOUT Z 
  ln.zPushLens(); 
  
  
logging.basicConfig(level=logging.INFO);

with DDElinkHandler() as hDDE:
  ln = hDDE.link;
  # load example file
  #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
  #                        'Cooke 40 degree field.zmx')
  filename= os.path.realpath('../13_catalog_optics_1mm_pupil_inf-inf-relay_point_source_with_slicer_tolerancing.ZMX');
  tol=ToleranceSystem(hDDE,filename)

  # raytrace parameters for image intensity before aperture
  image_surface = 22;
  wavenum  = 3;
  
  # disturb system (tolerancing)
  tol.change_thickness(4,11,value=2);     # shift of pupil slicer
  
  # compensator: rotate slicer around surface normal
  compensator_rotz(tol,3)

  #tol.tilt_decenter_elements(1,3,ydec=0.02);  # [mm]
  #tol.TETX(1,3,2.001) # [deg]
  tol.print_current_geometric_changes();
  
  # geometric image analysis
  img,params = GeometricImageAnalysis(hDDE);
  fig = img.show();

  # analyze img detector in detail (left and right sight separately)
  right= lambda x,y: np.logical_or(2*x+y<0, x>0.07); # right detector side
  left = lambda x,y: 2*x+y>=0;                       # left detector side
  x,intx_right = img.x_projection(fMask=right);
  y,inty_right = img.y_projection(fMask=right);
  x,intx_left  = img.x_projection(fMask=left);
  y,inty_left  = img.y_projection(fMask=left);
  dx=x[1]-x[0]; dy=y[1]-y[0];
    
  ax1,ax2 = fig.axes;  
  ax2.plot(x,inty_right); 
  ax2.plot(x,inty_left);  
  ax2.plot(x,inty_right+inty_left,':')
  
  # total intensity in image
  int_tot = np.sum(inty_right)*dy + np.sum(inty_left)*dy; # [W]
  
  # cumulative sum for getting percentage of loss
  cumx_right = np.cumsum(inty_right)*dy;
  cumx_left  = np.cumsum(inty_left )*dy;
  
  # where is loss 1mW ?
  thresh = [0.0001, 0.0005, 0.001, 0.002];  # [W]
  print find_quantiles(x, cumx_right, thresh); 
  print find_quantiles(x[::-1], np.cumsum(inty_right[::-1])*dy, thresh); 
  
  print find_quantiles(x, cumx_left, thresh); 
  
  plt.figure();
  Itot = np.sum(inty_right)+np.sum(inty_left);
  plt.plot(x,np.cumsum(inty_right)*dx);
  plt.plot(x,np.cumsum(inty_left)*dx)
  
  print