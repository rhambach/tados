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
from point_in_triangle import point_in_triangle
from adaptive_mesh import AdaptiveMesh,get_area,get_broken_triangles

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
    results = np.asarray( [(r.x,r.y,r.z,r.l,r.m,r.n,r.vigcode,r.error) for r in rays[1:]] );
    #print("retrive data: %ds"%(time.time()-t))    
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
  
class AnalyzeTransmission(object):

  def __init__(self, hDDE):
    self.hDDE=hDDE;


  def test(self):  
    # set up ray-trace parameters and image detector
    image_surface = 22;
    wavenum  = 1;
    image_size = np.asarray((0.2,0.05));
    image_size = np.asarray((0.2,0.05));
    image_shape = np.asarray((201,501));
    img_pixels = cartesian_sampling(*image_shape,rmax=2); # shape: (2,nPixels)
    img_pixels*= image_size[:,np.newaxis]/2;
    image_intensity = np.zeros(np.prod(image_shape)); # 1d array

    # field sampling
    xx,yy=cartesian_sampling(3,3,rmax=.1)  
    for i in xrange(len(xx)):
      x=xx[i]; y=yy[i];
      print("Field point: x=%5.3f, y=%5.3f"%(x,y))
      
      # init adaptive mesh for pupil sampling☺
      px,py=fibonacci_sampling_with_circular_boundary(200,2*np.sqrt(500))  
      initial_sampling = np.vstack((px,py)).T;         # size (nPoints,2)
      def raytrace(pupil_points):        # local function for raytrace
        px,py = pupil_points.T;
        ret = self.hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
        vigcode = ret[:,[6]]<>0;        # include vignetting by shifting ray outside image
        return ret[:,[0,1]]+image_size*vigcode;
      Mesh=AdaptiveMesh(initial_sampling, raytrace);

      # mesh refinement  
      if False:  
        # iterative refinement of broken triangles
        lthresh = 0.5*image_size[1];
        is_large= lambda(triangles): get_broken_triangles(triangles,lthresh);    
        for it in range(4): 
          Mesh.refine_large_triangles(is_large);
          if i==0: Mesh.plot_triangulation(skip_triangle=is_large);
      else:
        # segmentation of triangles along cutting line
        lthresh = 0.5*image_size[1];
        is_broken = lambda(triangles): get_broken_triangles(triangles,lthresh);  
        Mesh.refine_broken_triangles(is_broken,nDivide=100,bPlot=False);
        if i==0: Mesh.plot_triangulation(skip_triangle=is_broken);      
      pupil_points, image_points, simplices = Mesh.get_mesh();
  
      # analysis of beam intensity in each triangle (conservation of energy!) 
      broken = get_broken_triangles(image_points[simplices],lthresh=lthresh)  
      pupil_area = get_area(pupil_points[simplices]); 
      assert(all(pupil_area>0));  # triangles should be oriented ccw in pupil
      err_circ = 1-np.sum(pupil_area)/np.pi;    
      err_broken = np.sum(pupil_area[broken])/np.sum(pupil_area);
      logging.info('error of triangulation: \n' +
       '  %5.3f%% due to approx. of circular pupil boundary \n'%(err_circ*100) +
       '  %5.3f%% due to broken triangles' %(err_broken*100));
      image_area = get_area(image_points[simplices]);
      if any(image_area<0) and any(image_area>0):
        logging.warning('scambling of rays, triangulation may not be working')
      
      # footprint in image plane
      density = abs(pupil_area / image_area);
      for s in np.where(~broken)[0]:
        triangle = image_points[simplices[s]];
        mask = point_in_triangle(img_pixels,triangle);
        image_intensity += density[s]*mask;
      
      if i==0:
        plt.figure();
        plt.title("Intensity in each triangle of the Pupil Triangulation");
        plt.plot(pupil_area,'b',label="$A_{pupil}$")
        plt.plot(image_area,'g',label="$A_{image}$") 
        plt.legend(loc='best');
 
    # plotting of footprint in image plane
    img_pixels_2d = img_pixels.reshape(2,image_shape[1],image_shape[0]);
    image_intensity = image_intensity.reshape((image_shape[1],image_shape[0]))   
    xaxis = img_pixels_2d[1,:,0];
    yaxis = img_pixels_2d[0,0,:];
    
    fig,(ax1,ax2)= plt.subplots(2);
    ax1.set_title("footprint in image plane (surface: %d)"%image_surface);
    ax1.imshow(image_intensity,origin='lower',aspect='auto',interpolation='hanning',
               extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]]);
    ax2.set_title("integrated intensity in image plane");    
    ax2.plot(xaxis,np.sum(image_intensity,axis=1),label="along y");
    ax2.plot(yaxis,np.sum(image_intensity,axis=0),label="along x");
    ax2.legend(loc=0)
    
    return 1



if __name__ == '__main__':
  import os as os
  import sys as sys
  logging.basicConfig(level=logging.INFO);
  
  with DDElinkHandler() as hDDE:
  
    ln = hDDE.link;
    # load example file
    #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
    #                        'Cooke 40 degree field.zmx')
    filename= os.path.realpath('./tests/pupil_slicer.ZMX');
    
    hDDE.load(filename);
    
    AT = AnalyzeTransmission(hDDE);
    results = AT.test();  