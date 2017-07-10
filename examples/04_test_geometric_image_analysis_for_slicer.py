# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:55:26 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt
import logging
import os

import _set_pkgdir
from PyOptics.illumination.transmission import RectImageDetector
from PyOptics.tolerancing import tolerancing
from PyOptics.zemax import dde_link


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
  assert min_width>dx, 'min_widht too small';
  assert max_width<dx*(x.size-1), 'max_width too large';
  nBox = np.arange(np.floor(min_width/dx),np.ceil(max_width/dx),dtype=int);
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
  
 
logging.basicConfig(level=logging.WARNING);

with dde_link.DDElinkHandler() as hDDE:
  ln = hDDE.link;
  # load example file
  #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
  #                        'Cooke 40 degree field.zmx')
  filename= os.path.realpath('../tests/zemax/pupil_slicer.ZMX');
  tol=tolerancing.ToleranceSystem(hDDE,filename)

  # allow for compensators, here rotation about surface normal of slicer
  fig1,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True);   
  for rotz in (0,0.5,1):
  
    # shift of pupil slicer
    tol.reset();
    tol.change_thickness(4,11,value=2);
    tol.ln.zDeleteSurface(21);           # remove spider aperture
    tol.ln.zPushLens(1);    
    if rotz==0: tol.print_current_geometric_changes();
  
    # compensator: rotate slicer around surface normal
    if rotz!=0: compensator_rotz(tol,rotz);

    # geometric image analysis
    img,params = GeometricImageAnalysis(hDDE);
    if rotz==0: 
      fig2 = img.show();
      fig2.axes[0].set_ylim(-0.025,0.025);
      fig2.axes[1].set_ylim(0,40);
    
    # analyze img detector (enboxed energy along x and y):
    x,intx = img.x_projection(fMask=lambda x,y: x>0.07);
    y,inty = img.y_projection(fMask=lambda x,y: x>0.07);
    dx=x[1]-x[0]; dy=y[1]-y[0];
    
    # total intensity in image
    Itot = np.sum(inty)*dy; # [W]
    
    # cumulative sum for getting percentage of loss
    cumx = np.cumsum(intx)*dx;
    cumy = np.cumsum(inty)*dy;
    
    # find intensity in box of given width along y    
    x_boxx,I_boxx = intensity_in_box(x,cumx,min_width=0.115,max_width=0.13)
    y_boxy,I_boxy = intensity_in_box(y,cumy,min_width=0.03,max_width=0.04);
  
    # update figure 
    ax1.plot(x_boxx*1000,I_boxx,label='rotz=%3.1f'%rotz);
    ax2.plot(y_boxy*1000,I_boxy,label='rotz=%3.1f'%rotz);
  
    
  # end: for rotz      
  
  # draw results
  fig1.suptitle("system %s, Itot=%5.3fW" \
                  % (os.path.split(filename)[-1],Itot));
  ax1.set_xlabel('x-width [um]'); ax1.set_ylabel('Intensity [W]');
  ax2.set_xlabel('y-width [um]'); ax2.set_ylabel('Intensity [W]');
  ax1.set_ylim(0.95,1);  
  ax1.set_xlim(120,135);
  ax1.set_xticks(np.arange(120,136,5))
  ax2.set_xlim(32,36);
  ax2.set_xticks(np.arange(32,36.5,1))
  ax1.legend(loc=0);
  ax2.legend(loc=0);
  ax1.axvline(132,color='r',linestyle='--');
  ax2.axvline(33,color='r',linestyle='--');