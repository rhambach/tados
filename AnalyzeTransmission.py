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
import time
import matplotlib.pylab as plt
from matplotlib.path import Path

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
    results = np.asarray( [(r.x,r.y,r.z,r.l,r.m,r.n,r.error) for r in rays[1:]] );
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



class AdaptiveMesh(object):
  
  def __init__(self,initial_domain,mapping):
    """
      initial_domain ... 2d array of shape (nPixels,2)
      mapping        ... function image=mapping(domain) that accepts a list of  
                           domain points and returns corresponding image points
    """
    from scipy.spatial import Delaunay
     
    assert( initial_domain.ndim==2 and initial_domain.shape[1] == 2)
    self.initial_domain = initial_domain;
    self.mapping = mapping;
    # triangulation of initial domain
    self.tri = Delaunay(initial_domain,incremental=True);
    # calculate distorted grid
    self.initial_image = self.mapping(self.initial_domain);
    assert( self.initial_image.ndim==2)
    assert( self.initial_image.shape==(self.initial_domain.shape[0],2))
    # current domain and image during refinement and for plotting
    self.domain = self.initial_domain;    
    self.image  = self.initial_image;   
    
  
  def plot_triangulation(self,skip_triangle=None):
    """
    plot current triangulation of adaptive mesh in domain and image space
      skip_triangle... (opt) function mask=skip_triangle(triangles) that accepts a list of 
                     triangle vertices of shape (nTriangles, 3, 2) and returns 
                     a flag for each triangle indicating that it should not be drawn
    returns figure handle;
    """ 
    simplices = self.tri.simplices.copy();
    if skip_triangle is not None:
      skip = skip_triangle(self.image[simplices]);
      skipped_simplices=simplices[skip];
      simplices=simplices[~skip];
          
    fig,(ax1,ax2)= plt.subplots(2);
    ax1.set_title("Sampling + Triangulation in Domain");
    if skip_triangle is not None:
      ax1.triplot(self.domain[:,0], self.domain[:,1], skipped_simplices,'k:');
    ax1.triplot(self.domain[:,0], self.domain[:,1], simplices,'b-');    
    ax1.plot(self.initial_domain[:,0],self.initial_domain[:,1],'r.')
    
    ax2.set_title("Sampling + Triangulation in Image")
    ax2.triplot(self.image[:,0], self.image[:,1], simplices,'b-');
    ax2.plot(self.initial_image[:,0],self.initial_image[:,1],'r.')

    return fig;

  def refine_broken_triangles(self,is_broken,nDivide=1000,bPlot=False):
    """
    subdivide triangles which contain discontinuities in the image mesh
      is_broken ... function mask=is_broken(triangles) that accepts a list of 
                     triangle vertices of shape (nTriangles, 3, 2) and returns 
                     a flag for each triangle indicating if it should be subdivided
      nDivide   ... (opt) number of subdivisions of each side of broken triangle
      bPlot     ... (opt) plot sampling and selected points for debugging 
    """
    
    ind = is_broken(self.image[self.tri.simplices]);
    nTriangles = np.sum(ind)
    if nTriangles==0: return; # noting to do
    
    # for each broken triangle, get vertex points in domain
    triangles = self.domain[self.tri.simplices[ind]]; # shape (nTriangles,3,2)
    
    # - identify the two edges that are cut (correspond to longest edges in image space) 
    #sides = np.diff(np.concatenate((triangles,triangles[:,[0],:]),axis=1));
    #print np.argmin(np.sum(sides**2,axis=2),axis=1);

    # create dense sampling along edges of broken triangles
    # follow regular sampling on A->B->C->A in domain
    x = np.linspace(0,1,nDivide,endpoint=False);
    A,B,C = np.rollaxis(triangles,1);  # vertex points of shape (nTriangles,2)
    AB = np.outer(1-x,A)+np.outer(x,B); # subdivision of shape (Ndivide,nTriangles*2)
    BC = np.outer(1-x,B)+np.outer(x,C);
    CA = np.outer(1-x,C)+np.outer(x,A);
    domain_points = np.concatenate((AB,BC,CA,A.reshape(1,2*nTriangles)),axis=0);
    nSamplePoints = 3*nDivide+1;        # shape (nSamplePoints,nTriangles*2) 
    
    # calculate position of intermeiate points in image space
    image_points = self.mapping(domain_points.reshape(-1,2));
    sides = np.diff(image_points.reshape(nSamplePoints,nTriangles,2),axis=0);
    sides = np.sum(sides**2,axis=-1);  # shape (nSamplePoints-1,nTriangle)
    
    # there should be exactly two very long segments per triangle (crossing the discontinuity)
    largest_segments = np.argpartition(sides,-2,axis=0)[-2:]; 
                      # indices of two largest segments, shape (2,nTriangle)
    # we add the two end points of each of the largest segments to the mesh
    # (for each broken triangle we thus add 4 points)
    ind_first_point = largest_segments.T.flatten(); # index of first end point of segments
    ind_second_point= ind_first_point+1;            # index of second end point of segment
    ind_end_points = np.vstack((ind_first_point,ind_second_point)).T.flatten();
      # shape 4*nTriangles, order 4x(first end point, second end point), next triangle
    ind_triangle = np.arange(nTriangles).repeat(4);
      # corresponding triangle index, shape 4*nTriangles
    new_domain_points = domain_points.reshape((nSamplePoints,nTriangles,2))[ind_end_points,ind_triangle];
    new_image_points = image_points.reshape((nSamplePoints,nTriangles,2))[ind_end_points,ind_triangle];
 
    if bPlot:   
      fig = self.plot_triangulation(skip_triangle=is_broken);
      ax1,ax2 = fig.axes;
      ax1.plot(domain_points[:,0::2].flat,domain_points[:,1::2].flat,'k.',label='test points');
      ax1.plot(new_domain_points[...,0].flat,new_domain_points[...,1].flat,'r.',label='selected points');
      ax1.legend(loc=0);      
      ax2.plot(image_points[:,0],image_points[:,1],'k.')
      ax2.plot(new_image_points[...,0].flat,new_image_points[...,1].flat,'r.',label='selected points');

    # update mesh
    logging.info("refining_broken_triangles(): adding %d points"%(nSamplePoints));
    self.tri.add_points(new_domain_points)
    self.image = np.vstack((self.image,new_image_points));
    self.domain= np.vstack((self.domain,new_domain_points));      
    
  
  def refine_large_triangles(self,is_large):
    """
    subdivide large triangles in the image mesh
      is_large ... function mask=is_large(triangles) that accepts a list of 
                     triangle vertices of shape (nTriangles, 3, 2) and returns 
                     a flag for each triangle indicating if it should be subdivided
    """
    ind = is_large(self.image[self.tri.simplices]);
    if np.sum(ind)==0: return; # nothing to do
    
    # add center of gravity for critical triangles
    new_domain_points = np.sum(self.domain[self.tri.simplices[ind]],axis=1)/3;
    self.tri.add_points(new_domain_points);
    logging.info("refining_large_triangles(): adding %d points"%(new_domain_points.shape[0]))
    
    # calculate image points and update data
    new_image_points = self.mapping(new_domain_points);
    self.image = np.vstack((self.image,new_image_points));
    self.domain= np.vstack((self.domain,new_domain_points));
        
  def get_mesh(self):
    return self.domain,self.image,self.tri;



class AnalyzeTransmission(object):

  def __init__(self, hDDE):
    self.hDDE=hDDE;


  def test(self):  
    # set up ray-trace parameters and image detector
    image_surface = 22;
    wavenum  = 3;
    image_size = np.asarray((0.2,0.05));
    image_size = np.asarray((0.2,0.05));
    image_shape = np.asarray((101,201));
    img_pixels = np.transpose(cartesian_sampling(*image_shape,rmax=2)); # shape: (nPixels,2)
    img_pixels*= image_size/2;
    image_intensity = np.zeros(np.prod(image_shape)); # 1d array

    # field sampling
    xx,yy=cartesian_sampling(3,3,rmax=.1)  
    for i in xrange(len(xx)):
      x=xx[i]; y=yy[i];
      print("Field point: x=%5.3f, y=%5.3f"%(x,y))
      
      # init adaptive mesh for pupil sampling
      px,py=fibonacci_sampling_with_circular_boundary(100,2*np.sqrt(500))  
      initial_sampling = np.vstack((px,py)).T;         # size (nPoints,2)
      def raytrace(pupil_points):        # local function for raytrace
        px,py = pupil_points.T;
        ret = self.hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
        return ret[:,[0,1]];
      Mesh=AdaptiveMesh(initial_sampling, raytrace);

      # iteratively perform refinement
      #lthresh = 0.5*image_size[1];
      #is_large= lambda(triangles): get_broken_triangles(triangles,lthresh);    
      #for it in range(3): 
      #  Mesh.refine_large_triangles(is_large);
      #  if i==0: Mesh.plot_triangulation(skip_triangle=is_large);
      lthresh = 0.5*image_size[1];
      is_broken = lambda(triangles): get_broken_triangles(triangles,lthresh);  
      Mesh.refine_broken_triangles(is_broken,bPlot=True);
      if i==0: Mesh.plot_triangulation(skip_triangle=is_broken);      
      
      pupil_points, image_points, tri = Mesh.get_mesh();
      
        # analysis of beam intensity in each triangle (conservation of energy!) 
      broken = get_broken_triangles(image_points[tri.simplices],lthresh=lthresh)  
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
      
      # footprint in image plane
      density = abs(pupil_area / image_area);
      for s,vertices in enumerate(tri.simplices[~broken]):
        path = Path( image_points[vertices] );
        mask = path.contains_points(img_pixels);
        image_intensity += density[s]*mask;
      
      if i==0:
        plt.figure();
        plt.title("Intensity in each triangle of the Pupil Triangulation");
        plt.plot(pupil_area,'b',label="$A_{pupil}$")
        plt.plot(image_area,'g',label="$A_{image}$") 
        plt.legend(loc='best');
 
    # plotting of footprint in image plane
    img_pixels_2d = img_pixels.reshape(image_shape[1],image_shape[0],2);
    image_intensity = image_intensity.reshape((image_shape[1],image_shape[0]))   
    xaxis = img_pixels_2d[:,0,1];
    yaxis = img_pixels_2d[0,:,0];
    
    fig,(ax1,ax2)= plt.subplots(2);
    ax1.set_title("footprint in image plane (surface: %d)"%image_surface);
    ax1.imshow(image_intensity,origin='lower',aspect='auto',
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
    filename = 'X:/projekte/1507_image_slicer/zemax/10_complete_system';
    filename+= '/12_catalog_optics_1mm_pupil_point_source_with_slicer_tolerancing.ZMX';
    
    hDDE.load(filename);
    
    AT = AnalyzeTransmission(hDDE);
    results = AT.test();  