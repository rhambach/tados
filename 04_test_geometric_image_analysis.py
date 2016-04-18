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
  data,params = hDDE.zGeometricImageAnalysis(testFileName,timeout=1000);
  imgSize = float(params['Image Width'].split()[0]);
  
  # save results in RectImageDetector class
  img = RectImageDetector(extent=(imgSize,imgSize), pixels=data.shape);
  img.intensity = data.copy();
  return img,params
  
def find_quantiles(x,F,thresh):
  thresh = np.atleast_1d(thresh); 
  dx=x[1]-x[0];  
  # offset by half bin width necessary, to have same results if x and F are reversed  
  return np.interp([thresh, F[-1]-thresh],F,x); 

def intensity_in_box(x,F,min_width=None,max_width=None):
  """
  calculate maximum intensity in a box of different width
    x: x-axis (pixel centers)
    F: cumulative distribution function (intensity)
    min_width: minimal width of box
    max_width: maximal width of box
  returns:
    w: widht of box
    I: intensity inside box
  """  
  dx=x[1]-x[0];
  if min_width is None: min_width=dx*1.5;          # nBox>0
  if max_width is None: max_width=dx*(x.size-0.5); # nBox<x.size
  nBox = np.arange(np.floor(min_width/dx),np.ceil(max_width/dx));
  w= dx*(nBox+1);           # width of the box with nBox pixels
  I=np.asarray([np.max(F[n:]-F[:-n]) for n in nBox]);
  return (w,I)
 
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

  
  for rotz in (0,1,2,3,4):

    # disturb system (tolerancing)
    tol.reset();
    tol.change_thickness(4,11,value=2);     # shift of pupil slicer
    #tol.tilt_decenter_elements(1,3,ydec=0.02);  # [mm]
    #tol.TEX(1,3,.001) # [deg]
    if rotz==0: tol.print_current_geometric_changes();
    
    # compensator: rotate slicer around surface normal
    if rotz<>0: compensator_rotz(tol,rotz);
  
    # geometric image analysis
    img,params = GeometricImageAnalysis(hDDE);
    if rotz==0: fig = img.show();
    
  
    # analyze img detector in detail (left and right sight separately)
    right= lambda x,y: np.logical_or(2*x+y<0, x>0.07); # right detector side
    left = lambda x,y: 2*x+y>=0;                       # left detector side
    x,intx_right = img.x_projection(fMask=right);
    y,inty_right = img.y_projection(fMask=right);
    x,intx_left  = img.x_projection(fMask=left);
    y,inty_left  = img.y_projection(fMask=left);
    dx=x[1]-x[0]; dy=y[1]-y[0];
      
    if rotz==0:
      ax1,ax2 = fig.axes;  
      ax2.plot(x,inty_right); 
      ax2.plot(x,inty_left);  
      ax2.plot(x,inty_right+inty_left,':')
    
    # total intensity in image
    int_tot = np.sum(inty_right)*dy + np.sum(inty_left)*dy; # [W]
    
    # cumulative sum for getting percentage of loss
    cumy_right = np.cumsum(inty_right)*dy;
    cumy_left  = np.cumsum(inty_left )*dy;
    cumy       = cumy_right+cumy_left;
    
    # find intensity in box of given width along y
    y_boxy,I_boxy = intensity_in_box(y,cumy,min_width=0.03,max_width=0.04);
  
    # same for x-direction
    cumx = np.cumsum(intx_left+intx_right)*dx;
    x_boxx,I_boxx = intensity_in_box(x,cumx,min_width=0.120);
  
    # where is loss 1mW ?
    thresh = [0.001, 0.01];  # [W]
    print rotz,int_tot,thresh;
    print find_quantiles(y_boxy, I_boxy, thresh)[1]; 
    print find_quantiles(x_boxx, I_boxx, thresh)[1];
    
    if rotz==0:    
      plt.figure();
      Itot = np.sum(inty_right)+np.sum(inty_left);
      plt.plot(y,cumy_right);
      plt.plot(y,cumy_left);
      plt.plot(y_boxy,I_boxy);
    