# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 10:37:13 2016

@author: Hambach
"""

from __future__ import division
import pyzdde.arraytrace as at  # Module for array ray tracing
import pyzdde.zdde as pyz
import logging
import numpy as np
import matplotlib.pylab as plt

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
       results[0:3]: x,y,z coordinates of ray on requested surface
       results[3:6]: l,m,n direction cosines after requested surface
       results[7]:   error value
                      0 = ray traced successfully;
                      +ve number = the ray missed the surface;
                      -ve number = the ray total internal reflected (TIR) at surface
                                   given by the absolute value of the ``error``
    """
    nRays = x.size;
    if np.isscalar(waveNum): waveNum=np.zeros(nRays,np.int)+waveNum;
    assert(all(args.size == nRays for args in [x,y,px,py,waveNum]))
    print("number of rays: %d"%nRays);
        
    # fill in ray data array (following Zemax notation!)
    t = time.time();    
    rays = at.getRayDataArray(nRays, tType=0, mode=mode, endSurf=surf)
    for k in xrange(nRays):
      rays[k+1].x = x[k]      
      rays[k+1].y = y[k]
      rays[k+1].z = px[k]
      rays[k+1].l = py[k]
      rays[k+1].wave = waveNum[k];

    print("set pupil values: %ds"%(time.time()-t))

    # Trace the rays
    ret = at.zArrayTrace(rays, timeout=100000)
    print("zArrayTrace: %ds"%(time.time()-t))
#
    # collect results
    results = np.asarray( [(r.x,r.y,r.z,r.l,r.m,r.n,r.error) for r in rays[1:]] );
    print("retrive data: %ds"%(time.time()-t))    
    return results;

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

def fibonacci_sampling(N,rmax=1.):
  """
  Fibonacci sampling in reduced coordinates (normalized to 1)
   N     ... total number of points (must be >32)
   rmax  ... (opt) radius of circular aperture, default 1
  
  RETURNS
   x,y   ... 1d-vectors of x and y coordinates for each point
  """
  k = np.arange(N)+0.5;
  theta = 4*np.pi*k/(1+sqrt(5));
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
  return np.hstack([x, xp]), np.hstack([y, yp]);
  

class AnalyzeTransmission(object):

  def __init__(self, hDDE):
    self.hDDE=hDDE;

  def test(self):  
    from scipy.spatial import Delaunay

    # pupil sampling
    #px,py=cartesian_sampling(21,21);   
    px,py=fibonacci_sampling_with_circular_boundary(500,2*sqrt(500))  
    x=np.zeros(px.size); y=x+1;
    results = self.hDDE.trace_rays(x,y,px,py,1);

    # triangulation
    points = np.vstack([px,py]).T;
    tri = Delaunay(points);
    x,y = points[tri.simplices].T;  # [3,nTriangles]
    # area of triangle (see http://geomalgorithms.com/a01-_area.html#2D%20Polygons)    
    pupil_intensity = 0.5 * ( (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]) );
    assert(all(pupil_intensity>0));  # all triangles should be ccw oriented
    
    plt.figure();
    plt.triplot(tri.points[:,0], tri.points[:,1], tri.simplices.copy());
    plt.plot(px,py,'.')
    
    
    # triangulation in image space
    plt.figure();
    plt.triplot(results[:,0],results[:,1], tri.simplices.copy());
    plt.plot(results[:,0],results[:,1],'.')
    x = results[tri.simplices,0].T; # [3,nTriangles]
    y = results[tri.simplices,1].T;
    # area of triangles in image plane
    image_intensity = 0.5 * ( (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]) );

    plt.figure();
    plt.plot(pupil_intensity)
    plt.plot(image_intensity) 
    plt.plot(abs(image_intensity)/pupil_intensity)

    if any(image_intensity<0) and any(image_intensity>0):
      logging.warning('scambling of rays, triangulartion may not be working')

    return results



if __name__ == '__main__':
  import os as os
  import sys as sys
  logging.basicConfig(level=logging.INFO);
  
  with DDElinkHandler() as hDDE:
  
    ln = hDDE.link;
    # load example file
    #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
    #                        'Cooke 40 degree field.zmx')
    filename = 'X:/projekte/1507_image_slicer/zemax/10_complete_system';
    filename+= '/12_catalog_optics_1mm_pupil_point_source_with_slicer_tolerancing.ZMX';
    
    hDDE.load(filename);
    
    AT = AnalyzeTransmission(hDDE);
    resultsre = AT.test();  