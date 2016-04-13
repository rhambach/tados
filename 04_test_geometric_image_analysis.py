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



def zGeometricImageAnalysis(textFileName=None):
  # set up temporary name for analysis data
  if textFileName is None:
    fdir = os.path.dirname(ln.zGetFile());
    textFileName = os.path.join(fdir,'__pyzdde_geometricImageSimulationAnalysisFile.txt');
    
  # perform Geometric Image Analysis (with current settings)
  ret = ln.zGetTextFile(textFileName,'Ima');
  assert ret == 0, 'zGetTextFile() returned error code {}'.format(ret) 
  lines = zdde._readLinesFromFile(zdde._openFile(textFileName))
  assert(lines[0]=="Image analysis histogram listing");  # expect output of Image analysis
  
  # scan header
  last_line_header = zdde._getFirstLineOfInterest(lines,'Units');
  params = [];  
  for line in lines[1:last_line_header+1]:
    pos = line.find(':');    
    if pos>0: params.append((line[0:pos].strip(), line[pos+1:].strip()));
  params=dict(params);  
  # ... might be improved
  
  # extract image size (e.g. '0.14 Millimeters')
  imgSize = float(params['Image Width'].split()[0]);
  Nrays   = int(params['Total Rays Launched']);
  Nx,Ny   = map(int,params['Number of pixels'].split('x'));
  totFlux = float(params['Total flux in watts']);
  
  # scan data (values in textfile are ordered like in Zemax Window, i.e.
  #   with increasing column index, x increases from -imgSize/2 to imgSize/2
  #   with increasing line   index, y decreases from imgSize/2 to -imgSize/2
  first_line_data  = last_line_header+2;
  # load text reads 2d matrix as index [line,column], corresponds here to [-y,x]
  data = np.loadtxt(lines[first_line_data:]);
  data = data[::-1].T;                           # reorder data as [x,y]
  totFlux_data = np.sum(data)*imgSize**2/Nx/Ny;
  assert (data.shape==(Nx,Ny));                  # correct number of pixels read from file
  assert (abs(1-totFlux_data/totFlux) < 0.001);  # check that total flux is correct within 0.1%
  plt.figure()  
  plt.imshow(data.T,origin='lower',aspect='auto',interpolation='hanning',
             cmap='gray',extent=np.array([-1,1,-1,1])*imgSize/2);
  
  # Plot using RectImageDetector class
  img = RectImageDetector(extent=(imgSize,imgSize), pixels=(Nx,Ny));
  img.intensity = data.copy();
  img.show();
  
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
  zGeometricImageAnalysis();
  
  
  