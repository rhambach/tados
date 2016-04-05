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
    # enlarge all vector arguments to same size    
    nRays = max(map(np.size,(x,y,px,py,waveNum)));
    if np.isscalar(x): x = np.zeros(nRays)+x;
    if np.isscalar(y): y = np.zeros(nRays)+y;
    if np.isscalar(px): px = np.zeros(nRays)+px;
    if np.isscalar(py): py = np.zeros(nRays)+py;    
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
  return np.hstack((x, xp)), np.hstack((y, yp));
  
def get_area(triangles):
  """
  calculate signed area of each triangle in given list
    triangles ... list of size (nTriangles,3,2) x and y coordinates 
                  for each vertex in each triangle  
  Returns:
    1d vector of size nTriangles containing the signed area of each triangle
    (positive: ccw orientation, negative: cw orientation of vertices)
  """
  x,y = triangles.T;
  # See http://geomalgorithms.com/a01-_area.html#2D%20Polygons
  return 0.5 * ( (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]) );

def get_broken_triangles(triangles,lthresh=None):
  """
  try to identify triangles that are cut or vignetted
    triangles ... list of size (nTriangles,3,2) x and y coordinates 
                  for each vertex in each triangle  
    lthresh   ... threshold for longest side of broken triangle 
  Returns:
    1d vector of size nTriangles indicating if triangle is broken
  """
  # calculate maximum of (squared) length of two sides of each triangle 
  # (X[0]-X[1])**2 + (Y[0]-Y[1])**2; (X[1]-X[2])**2 + (Y[1]-Y[2])**2 
  max_lensq = np.max(np.sum(np.diff(triangles,axis=1)**2,axis=2),axis=1);
  # mark triangle as broken, if max side is 10 times larger than median value
  if lthresh is None: lthresh = 3*np.sqrt(np.median(max_lensq));
  return max_lensq > lthresh**2;


class AnalyzeTransmission(object):

  def __init__(self, hDDE):
    self.hDDE=hDDE;


  def test(self):  
    from scipy.spatial import Delaunay

    # pupil sampling
    #px,py=cartesian_sampling(21,21);   
    image_size = np.asarray((0.14,0.14));
    px,py=fibonacci_sampling_with_circular_boundary(500,2*sqrt(500))  
    x=1; y=0;
    # triangulation
    pupil_points = np.vstack((px,py)).T; # size (nPoints,2)
    tri = Delaunay(pupil_points,incremental=True);

    # raytrace to image plane and refinement
    results = self.hDDE.trace_rays(x,y,px,py,1);
    image_points = results[:,[0,1]];
    lthresh = 0.1*np.sqrt(np.sum(image_size**2));
    for it in range(3): 
      # refine triangulation, if triangles have very large sides (cut triangles)
      ind = get_broken_triangles(image_points[tri.simplices],lthresh=lthresh);
      if np.sum(ind)==0: break;
      # add center of gravity for critical triangles
      new_pupil_points = np.sum(pupil_points[tri.simplices[ind]],axis=1)/3;
      tri.add_points(new_pupil_points);
      logging.info("refining pupil sampling (iteration %d): adding %d points"%(it,new_pupil_points.shape[0]))
      # raytrace for new points and update of data
      new_results = self.hDDE.trace_rays(x,y,new_pupil_points[:,0],new_pupil_points[:,1],1)
      results = np.vstack((results,new_results));
      pupil_points = np.vstack((pupil_points,new_pupil_points));
      image_points = results[:,[0,1]]
    tri.close(); # finish refinement (free resources)

    # DEBUG plotting of footprint + triangulation in pupil and image
    plt.figure();
    plt.title("Pupil Sampling + Triangulation");
    plt.triplot(pupil_points[:,0], pupil_points[:,1], tri.simplices.copy());
    plt.plot(px,py,'.')
    
    plt.figure();
    plt.title("Spot in Image space + Triangulation from Pupil")
    broken = get_broken_triangles(image_points[tri.simplices],lthresh=lthresh)
    plt.triplot(results[:,0],results[:,1], tri.simplices[~broken].copy());
    plt.plot(results[:,0],results[:,1],'.')
    
    # analysis of beam intensity in each triangle (conservation of energy!) 
    pupil_area = get_area(pupil_points[tri.simplices]); 
    assert(all(pupil_area>0));  # all triangles should be ccw oriented in pupil
    err_circ = 1-np.sum(pupil_area)/np.pi;    
    err_broken = np.sum(pupil_area[broken])/np.sum(pupil_area);
    logging.info('error of triangulation: \n' +
     '  %5.3f%% due to approx. of circular pupil boundary \n'%(err_circ*100) +
     '  %5.3f%% due to broken triangles' %(err_broken*100));
    image_area = get_area(image_points[tri.simplices]);
    if any(image_area<0) and any(image_area>0):
      logging.warning('scambling of rays, triangulation may not be working')

    plt.figure();
    plt.title("Intensity in each triangle of the Pupil Triangulation");
    plt.plot(pupil_area,'b',label="$A_{pupil}$")
    plt.plot(image_area,'g',label="$A_{image}$") 
    #ratio = abs(image_area/pupil_area);
    #plt.plot(ratio,'k',label="$A_{image}/A_{pupil}$")
    #ratio[~get_broken_triangles(image_points[tri.simplices])]=np.NaN;    
    #plt.plot(ratio,'r.',label="broken triangles")
    plt.legend(loc='best');
 
 
    # footprint in image plane
    img_shape=(1001,1001);
    img_pixels = np.transpose(cartesian_sampling(*img_shape,rmax=2)); # shape: (nPixels,2)
    img_pixels*= image_size/2;
    tri.points = image_points;
    ind = tri.find_simplex(img_pixels);
    density = abs(pupil_area / image_area);
    density = np.hstack((density, [0,]));
    intensity = density[ind].reshape(img_shape);
    plt.figure();
    plt.imshow(intensity,origin='lower');
    plt.figure();
    plt.plot(np.sum(intensity,axis=0));
    plt.plot(np.sum(intensity,axis=1))
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
    results = AT.test();  