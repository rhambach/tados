# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 19:45:55 2016

@author: Hambach
"""
import numpy as np
import logging
import os as _os

import pyzdde.arraytrace as at  # Module for array ray tracing
import pyzdde.zdde as pyz

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
    if ret!=0:
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
    
    Parameters
    ----------
      x,y : vectors of length nRays
        reduced field coordinates for each ray (normalized between -1 and 1)
      px,py : vectors of length nRays
        reduced pupil coordinates for each ray (normalized between -1 and 1)
      waveNum : integer
        wavelength number
      mode : integer, optional
        0= real (default), 1 = paraxial
      surf : integer, optional
        surface to trace the ray to. Usually, the ray data is only needed at
        the image surface (``surf = -1``, default)

    Returns
    --------
      numpy array of shape (nRays,8) containing following parameters for each ray
      
      err : error flag
        * 0 = ray traced successfully;
        * +ve number = the ray missed the surface;
        * -ve number = the ray total internal reflected (TIR) at surface given 
          by the absolute value of the ``error``
      vigcode : integer
        The first surface where the ray was vignetted. Raytrace is continued.       
      x,y,z : float 
        cartesian coordinates of ray on requested surface (local coordinates)
      l,m,n : float 
        direction cosines after requested surface (local coordinates)
      l2,m2,n2 : float
        direction cosines of surface normal at point of incidence (local coordinates)
       
    """
    # enlarge all vector arguments to same size    
    nRays = max(map(np.size,(x,y,px,py,waveNum)));
    if np.isscalar(x): x = np.zeros(nRays)+x;
    if np.isscalar(y): y = np.zeros(nRays)+y;
    if np.isscalar(px): px = np.zeros(nRays)+px;
    if np.isscalar(py): py = np.zeros(nRays)+py;    
    if np.isscalar(waveNum): waveNum=np.zeros(nRays,np.int)+waveNum;
    assert(all(args.size == nRays for args in [x,y,px,py,waveNum]))
    print("number of rays: %d"%nRays);
    import time;  t = time.time();    
        
    # fill in ray data array (following Zemax notation!)
    rays = at.getRayDataArray(nRays, tType=0, mode=mode, endSurf=surf)
    for k in range(nRays):
      rays[k+1].x = x[k]      
      rays[k+1].y = y[k]
      rays[k+1].z = px[k]
      rays[k+1].l = py[k]
      rays[k+1].wave = waveNum[k];

    print("set pupil values: %ds"%(time.time()-t))

    # Trace the rays
    ret = at.zArrayTrace(rays, timeout=100000);
    print(("zArrayTrace: %ds"%(time.time()-t)))

    # collect results
    results = np.asarray( [(r.error,r.vigcode,r.x,r.y,r.z,r.l,r.m,r.n,r.Exr,r.Eyr,r.Ezr) for r in rays[1:]] );
    print(("retrive data: %ds"%(time.time()-t)))    
    return results;


  def zGeometricImageAnalysis(self,textFileName=None,timeout=None):
    """
    perform Geometric Image Analysis in Zemax and return Detector information
    
    Parameters
    ----------
      textFileName : string, optional
        name of the textfile used for extracting data from Zemax
      timeout :      integer, optional
        timeout in seconds
      
    Returns
    ------
      data :         2D array of shape (Nx,Ny)
        detector intensity for each pixel [W/mm2] 
      params :       dictionary
        dictionary with parameters of the image analysis
    """  
    
    # set up temporary name for analysis data
    if textFileName is None:
      fdir = _os.path.dirname(self.link.zGetFile());
      textFileName = _os.path.join(fdir,'__pyzdde_geometricImageSimulationAnalysisFile.txt');
      
    # perform Geometric Image Analysis (with current settings)
    ret = self.link.zGetTextFile(textFileName,'Ima',timeout=timeout);
    assert ret == 0, 'zGetTextFile() returned error code {}'.format(ret) 
    lines = pyz._readLinesFromFile(pyz._openFile(textFileName))
    assert lines[0]=='Image analysis histogram listing', "Output of Image analysis not found";

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
    print(totFlux)
    assert (data.shape==(Nx,Ny));                  # correct number of pixels read from file
    assert (abs(1-totFlux_data/totFlux) < 0.001);  # check that total flux is correct within 0.1%
    #plt.figure()  
    #plt.imshow(data.T,origin='lower',aspect='auto',interpolation='hanning',
    #           cmap='gray',extent=np.array([-1,1,-1,1])*imgSize/2);  
    
    return data,params




  def zInterferogram(self,textFileName=None,timeout=None):
    """
    perform calculation of Interferogram in Zemax and return data as numpy array
    
    Parameters
    ----------
      textFileName : string, optional
        name of the textfile used for extracting data from Zemax
      timeout :      integer, optional
        timeout in seconds
      
    Returns
    ------
      data :         2D array of shape (Nx,Ny)
        interferogram intensity for each pixel [W/mm2] 
      params :       dictionary
        dictionary with parameters of the interferogram
    """  
    
    # set up temporary name for analysis data
    if textFileName is None:
      fdir = _os.path.dirname(self.link.zGetFile());
      textFileName = _os.path.join(fdir,'__pyzdde_InterferogramFile.txt');
      
    # perform Geometric Image Analysis (with current settings)
    ret = self.link.zGetTextFile(textFileName,'Int',timeout=timeout);
    assert ret == 0, 'zGetTextFile() returned error code {}'.format(ret) 
    lines = pyz._readLinesFromFile(pyz._openFile(textFileName))
    assert lines[0]=='Listing of Interferogram Data', "Output of Image analysis not found";
    
    # scan header
    last_line_header = pyz._getFirstLineOfInterest(lines,'Xtilt');
    params = [];  
    for line in lines[1:last_line_header+1]:
      pos = line.find(':');    
      if pos>0: params.append((line[0:pos].strip(), line[pos+1:].strip()));
    params=dict(params);  
    # ... might be improved
    
    
    # scan data (values in textfile are ordered like in Zemax Window, i.e.
    #   with increasing column index, x increases from -imgSize/2 to imgSize/2
    #   with increasing line   index, y decreases from imgSize/2 to -imgSize/2
    first_line_data  = last_line_header+2;
    # load text reads 2d matrix as index [line,column], corresponds here to [-y,x]
    data = np.loadtxt(lines[first_line_data:]);
    data = data[::-1].T;                           # reorder data as [x,y]
    assert (data.shape[0]==data.shape[1]);         # Nx=Ny
    #plt.figure()  
    #plt.imshow(data.T,origin='lower',aspect='auto',interpolation='hanning',
    #           cmap='gray',extent=np.array([-1,1,-1,1])*imgSize/2);  
    
    return data,params