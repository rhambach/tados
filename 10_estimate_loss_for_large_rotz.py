# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:55:26 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt
import logging
import os
from tolerancing import *
from transmission import RectImageDetector
from zemax_dde_link import *
import cPickle as pickle
import gzip

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
  

logging.basicConfig(level=logging.WARNING);

with DDElinkHandler() as hDDE:
  ln = hDDE.link;
  # load example file
  #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
  #                        'Cooke 40 degree field.zmx')
  #filename= os.path.realpath('../13_catalog_optics_1mm_pupil_inf-inf-relay_point_source_with_slicer_tolerancing.ZMX');
  filename= os.path.realpath('../12_catalog_optics_1mm_pupil_point_source_with_slicer_tolerancing.ZMX');  
  tol=ToleranceSystem(hDDE,filename)
  print('system: %s\n'%filename)
  
  # iterate over different fnumbers
  for fnum in (3.5,3):
    NA = np.sin(np.arctan(0.5/fnum));
    print " Calculation for NA=%8.3f of beam at F1"%NA;
      
    # change rotation about surface normal of slicer
    # fig1,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True);   
    for rotz in (0,0.5,1,2,3,4,5,6):
  
      # restore undisturbed system
      tol.reset();
      tol.change_thickness(4,11,value=2);     # shift of pupil slicer
      if rotz<>0: compensator_rotz(tol,rotz); # compensator: rotate slicer around surface normal
      
      # set NA in object space
      (aType,stopSurf,aperVal) = tol.ln.zGetSystemAper();    
      tol.ln.zSetSystemAper(aType,stopSurf,NA);
      
      # update changes
      tol.ln.zPushLens(1);    
      if rotz==0: tol.print_current_geometric_changes();
    
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
      x_boxx,I_boxx = intensity_in_box(x,cumx,min_width=0.11,max_width=0.13)
      y_boxy,I_boxy = intensity_in_box(y,cumy,min_width=0.022,max_width=0.035);
    
      # write results
      print('  %8.5f  %8.5f'%(rotz,Itot));
          
    
      # plot enboxed energy:
      if rotz==0: #open new figure 
        fig1,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True);   
        fig1.suptitle("Input beam f/#: %3.1f, Itot=%5.3fW" \
                    % (fnum,Itot));  
      ax1.plot(x_boxx*1000,I_boxx,label='rotz=%3.1f'%rotz);
      ax2.plot(y_boxy*1000,I_boxy,label='rotz=%3.1f'%rotz);
      ax1.set_xlabel('x-width [um]'); ax1.set_ylabel('Intensity [W]');
      ax2.set_xlabel('y-width [um]'); ax2.set_ylabel('Intensity [W]');
      ax1.set_ylim(0.80,1);  
      ax1.set_xlim(1000*x_boxx[0],1000*x_boxx[-1]);
      ax2.set_xlim(1000*y_boxy[0],1000*y_boxy[-1]);
      ax1.legend(loc=0);  
      #ax2.legend(loc=0);
  
  
      # analyze img detector (enboxed energy along x and y):
      x,intx = img.x_projection(fMask=lambda x,y: x>0.07);
      y,inty = img.y_projection(fMask=lambda x,y: x>0.07);
      dx=x[1]-x[0]; dy=y[1]-y[0];
  
    # end: for rotz      
  
