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
  
  
  
logging.basicConfig(level=logging.INFO);

with DDElinkHandler() as hDDE:

  ln = hDDE.link;
  # load example file
  #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
  #                        'Cooke 40 degree field.zmx')
  filename= os.path.realpath('./tests/pupil_slicer.ZMX');
  tol=ToleranceSystem(hDDE,filename)

  # raytrace parameters for image intensity before aperture
  image_surface = 22;
  wavenum  = 3;
  
  # disturb system (tolerancing)
  tol.change_thickness(5,12,value=2); # shift of pupil slicer
  tol.tilt_decenter_elements(1,3,ydec=0.02);  # [mm]
  tol.TETX(1,3,2.001) # [deg]
  tol.print_current_geometric_changes();
  
  # geometric image analysis
  img,params = GeometricImageAnalysis(hDDE);
  
   # analyze img detector in detail (left and right sight separately)
  img.show(fMask = lambda x,y: np.logical_or(2*x+y<0, x>0.07))  
  img.show(fMask = lambda x,y: 2*x+y>=0)
  