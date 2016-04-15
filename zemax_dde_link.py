# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 19:45:55 2016

@author: Hambach
"""
import pyzdde.arraytrace as at  # Module for array ray tracing
import pyzdde.zdde as pyz
import numpy as np
import matplotlib.pylab as plt
import logging
import os as _os

class DDElinkHandler(object):
  """
  ensure that DDE link is always closed, see discussion in 
  http://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
  """
  def __init__(self):
    self.link=None;
  
  def __enter__(self):
    " initialize DDE connection to Zemax "
    self.link = pyz.createLink()
    if self.link is None:
      raise RuntimeError("Zemax DDE link could not be established.");
    return self;
    
  def __exit__(self, exc_type, exc_value, traceback):
    " close DDE link"
    self.link.close();

  def load(self,zmxfile):
    " load ZMX file with name 'zmxfile' into Zemax "
    ln = self.link;   

    # load file to DDE server
    ret = ln.zLoadFile(zmxfile);
    if ret<>0:
        raise IOError("Could not load Zemax file '%s'. Error code %d" % (zmxfile,ret));
    logging.info("Successfully loaded zemax file: %s"%ln.zGetFile())
    
    # try to push lens data to Zemax LDE
    ln.zGetUpdate() 
    if not ln.zPushLensPermission():
        raise RuntimeError("Extensions not allowed to push lenses. Please enable in Zemax.")
    ln.zPushLens(1)
    
  def trace_rays(self,x,y, px,py, waveNum, mode=0, surf=-1):
    """ 
    array trace of rays
      x,y   ... list of reduced field coordinates for each ray (length nRays)
      px,py ... list of reduced pupil coordinates for each ray (length nRays)
      waveNum.. wavelength number
      mode  ... (opt) 0= real (Default), 1 = paraxial
      surf  ... surface to trace the ray to. Usually, the ray data is only needed at
                the image surface (``surf = -1``, default)

    Returns
      results... numpy array with ... columns containing
       results[0]:   error value
                      0 = ray traced successfully;
                      +ve number = the ray missed the surface;
                      -ve number = the ray total internal reflected (TIR) at surface
                                   given by the absolute value of the ``error``
       results[1]:   vigcode
                      The first surface where the ray was vignetted. Raytrace is continued.       
       results[2:5]: x,y,z coordinates of ray on requested surface
       results[5:8]: l,m,n direction cosines after requested surface
       
    """
    # enlarge all vector arguments to same size    
    nRays = max(map(np.size,(x,y,px,py,waveNum)));
    if np.isscalar(x): x = np.zeros(nRays)+x;
    if np.isscalar(y): y = np.zeros(nRays)+y;
    if np.isscalar(px): px = np.zeros(nRays)+px;
    if np.isscalar(py): py = np.zeros(nRays)+py;    
    if np.isscalar(waveNum): waveNum=np.zeros(nRays,np.int)+waveNum;
    assert(all(args.size == nRays for args in [x,y,px,py,waveNum]))
    #print("number of rays: %d"%nRays);
    #t = time.time();    
        
    # fill in ray data array (following Zemax notation!)
    rays = at.getRayDataArray(nRays, tType=0, mode=mode, endSurf=surf)
    for k in xrange(nRays):
      rays[k+1].x = x[k]      
      rays[k+1].y = y[k]
      rays[k+1].z = px[k]
      rays[k+1].l = py[k]
      rays[k+1].wave = waveNum[k];

    #print("set pupil values: %ds"%(time.time()-t))

    # Trace the rays
    ret = at.zArrayTrace(rays, timeout=100000)
    #print("zArrayTrace: %ds"%(time.time()-t))
#
    # collect results
    results = np.asarray( [(r.error,r.vigcode,r.x,r.y,r.z,r.l,r.m,r.n) for r in rays[1:]] );
    #print("retrive data: %ds"%(time.time()-t))    
    return results;


  def zGeometricImageAnalysis(self,textFileName=None):
    """
    perform Geometric Image Analysis in Zemax and return Detector information
      textFileName ... (opt) name of the textfile used for extracting data from Zemax
      
    Returns: (data,params)
      data  ... 2D array containing detector intensity for each pixel [W/mm2], shape (Nx,Ny) 
      params... dictionary with parameters of the image analysis
    """  
    
    # set up temporary name for analysis data
    if textFileName is None:
      fdir = _os.path.dirname(self.link.zGetFile());
      textFileName = _os.path.join(fdir,'__pyzdde_geometricImageSimulationAnalysisFile.txt');
      
    # perform Geometric Image Analysis (with current settings)
    ret = self.link.zGetTextFile(textFileName,'Ima');
    assert ret == 0, 'zGetTextFile() returned error code {}'.format(ret) 
    lines = pyz._readLinesFromFile(pyz._openFile(textFileName))
    assert(lines[0]=="Image analysis histogram listing");  # expect output of Image analysis
    
    # scan header
    last_line_header = pyz._getFirstLineOfInterest(lines,'Units');
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
    #plt.figure()  
    #plt.imshow(data.T,origin='lower',aspect='auto',interpolation='hanning',
    #           cmap='gray',extent=np.array([-1,1,-1,1])*imgSize/2);  
    
    return data,params


def cartesian_sampling(nx,ny,rmax=1.):
  """
  cartesian sampling in reduced coordinates (between -1 and 1)
   nx,ny ... number of points along x and y
   rmax  ... (opt) radius of circular aperture, default 1
  
  RETURNS
   x,y   ... 1d-vectors of x and y coordinates for each point
  """
  x = np.linspace(-1,1,nx);
  y = np.linspace(-1,1,ny);
  x,y=np.meshgrid(x,y);   
  ind = x**2 + y**2 <= rmax;
  return x[ind],y[ind]

def hexapolar_sampling(Nr,rmax=1.,ind=False):
  """
  hexapolar sampling with roughly equi-area sampling point distribution
   Nr   ... number of rings, last ring has index (Nr-1)
   rmax ... normalization radius
   ind  ... (opt) return number of points and weights of each ring
  """
  r = np.arange(1,Nr,dtype=np.double)/Nr*rmax;               # ring centers
  Ntet = 6*np.arange(1,Nr);                  # number of points on each ring
  # construct grid points in each ring  
  x=[0]; y=[0];                              # first ring
  for i in xrange(Nr-1):
    tet = np.linspace(0,2*np.pi,Ntet[i],endpoint=False);
    x.extend(r[i]*np.cos(tet));
    y.extend(r[i]*np.sin(tet));
  x=np.asarray(x).flatten();
  y=np.asarray(y).flatten();
  # calculate number of points and weight of each ring, if desired
  if ind==False: 
    return x,y;
  elif Nr==1:
    return x,y,[1],[1];
  else:
    # calculate area of each ring Nr>1
    dr = r[1]-r[0];    
    area = ( (r+dr/2.)**2 - (r-dr/2.)**2 );
    # add first ring    
    area = np.insert(area,0, (dr/2.)**2);
    Ntet = np.insert(Ntet,0, 1);
    return x,y,Ntet,area/rmax**2;
    
      
def fibonacci_sampling(N,rmax=1.):
  """
  Fibonacci sampling in reduced coordinates (normalized to 1)
   N     ... total number of points (must be >32)
   rmax  ... (opt) radius of circular aperture, default 1
  
  RETURNS
   x,y   ... 1d-vectors of x and y coordinates for each point
  """
  k = np.arange(N)+0.5;
  theta = 4*np.pi*k/(1+np.sqrt(5));
  r = rmax*np.sqrt(k/N)
  x = r * np.sin(theta);
  y = r * np.cos(theta);
  return x,y
  
def fibonacci_sampling_with_circular_boundary(N,Nboundary=32,rmax=1.):
  assert(N>Nboundary);  
  x,y = fibonacci_sampling(N-Nboundary,rmax=0.97*rmax);
  theta = 2*np.pi*np.arange(Nboundary)/Nboundary;
  xp = rmax * np.sin(theta);
  yp = rmax * np.cos(theta);
  return np.hstack((x, xp)), np.hstack((y, yp));
  

if __name__ == '__main__':
  x,y,Ntet,w = hexapolar_sampling(20,ind=True);
  plt.figure();
  plt.scatter(x,y);