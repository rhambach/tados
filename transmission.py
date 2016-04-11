# -*- coding: utf-8 -*-
"""

Provides several classes for calculation of transmission through an optical system

Transmission: perform the transmission calculation (iterate over discrete set of
  parameters (e.g. field points) and evaluate the transmission on an adaptive mesh
  (e.g. pupil coordinates) using a given raytrace function
Detectors: analyze the raytrace results

@author: Hambach
"""

from __future__ import division
import abc
import logging
import numpy as np
import matplotlib.pylab as plt
from point_in_triangle import point_in_triangle
from adaptive_mesh import *
from zemax_dde_link import *

class Detector(object):
  __metaclass__ = abc.ABCMeta
  @abc.abstractmethod
  def add(self,mesh,skip=None,weight=1): return;
  @abc.abstractmethod  
  def show(self): return;


class CheckTriangulationDetector(Detector):
  " Detector class for testing completeness of triangulation in domain"

  def __init__(self, ref_area=np.pi):
    """
    ref_area ... (opt) theoretical area of domain space, default: area of unit circle
    """
    self.ref_domain_area=ref_area;
  
  def add(self,mesh,skip=None,weight=1):
    """
    calculate total domain area of mesh and print logging info 
      mesh ... instance of AdaptiveMesh 
      skip ... indices which simplices should be skipped
      weight.. ignored
    """
    triangle_area  = mesh.get_area_in_domain(); 
    assert(all(triangle_area>0));  # triangles should be oriented ccw in mesh    
    mesh_domain_area= np.sum(np.abs(triangle_area));
    err_boundary= 1-mesh_domain_area/self.ref_domain_area;
    out = 'error of triangulation of mesh: \n' + \
     '  %5.3f%% due to approx. of mesh boundary \n'%(err_boundary*100);
    if skip is not None:
      err_skip  = np.sum(triangle_area[skip])/mesh_domain_area;
      out += '  %5.3f%% due to skipped triangles' %(err_skip*100);
    logging.info(out);
    #image_area = Mesh.get_area_in_image();
    #if any(image_area<0) and any(image_area>0):
    #  logging.warning('scambling of rays, triangulation may not be working')
       
  def show(self): 
    raise NotImplemented();


     
class RectImageDetector(Detector):    
  " 2D Image Detector with cartesian coordinates "

  def __init__(self, extent=(1,1), pixels=(100,100)):
    """
     extent ... size of detector in image space (xwidth, ywidth)
     pixels ... number of pixels in x and y (xnum,ynum)
    """
    self.extent = np.asarray(extent);
    self.pixels = np.asarray(pixels);
    self.points = cartesian_sampling(*pixels,rmax=2); # shape: (2,nPixels)
    self.points *= self.extent[:,np.newaxis]/2;
    self.intensity = np.zeros(np.prod(self.pixels));  # 1d array

  def add(self,mesh,skip=None,weight=1):
    """
    calculate footprint in image plane
      mesh ... instance of AdaptiveMesh 
      skip ... indices which simplices should be skipped
      weight.. weight of contribution (intensity in Watt)
    """
    domain_area = mesh.get_area_in_domain(); 
    domain_area /= np.sum(np.abs(domain_area));   # normalized weight in domain
    image_area  = mesh.get_area_in_image();       # size of triangle in image
    density = weight * abs( domain_area / image_area);
    for s in np.where(~skip)[0]:
      triangle = mesh.image[mesh.simplices[s]];
      mask = point_in_triangle(self.points,triangle);
      self.intensity += density[s]*mask;

  def show(self):
    " plotting 2D footprint in image plane, returns figure handle"
    Nx,Ny = self.pixels;
    img_pixels_2d = self.points.reshape(2,Ny,Nx);
    image_intensity = self.intensity.reshape(Ny,Nx);
    xaxis = img_pixels_2d[1,:,0]; dx=xaxis[1]-xaxis[0];
    yaxis = img_pixels_2d[0,0,:]; dy=yaxis[1]-yaxis[0];
  
    fig,(ax1,ax2)= plt.subplots(2);
    ax1.set_title("footprint in image plane");
    ax1.imshow(image_intensity,origin='lower',aspect='auto',interpolation='hanning',
             extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]]);
    ax2.set_title("integrated intensity in image plane");    
    ax2.plot(xaxis,np.sum(image_intensity,axis=1)*dy,label="along y");
    ax2.plot(yaxis,np.sum(image_intensity,axis=0)*dx,label="along x");
    ax2.legend(loc=0)
  
    logging.info('total intensity: %5.3f W'%(np.sum(image_intensity)*dx*dy)); 
    return fig
    
      

class Transmission(object):
  def __init__(self, parameters, mesh_points, raytrace, detectors, weights=None):
    """
    Transmission for rays defined by a set of discrete parameters and a mesh, 
    which can be refined iteratively. The results are recorded by a set of
    given detectors, which are called for each parameter sequentially.
    
      parameters ... list of Np discrete parameters for each raytrace, shape (nParams,Np)
      mesh_points... list of initial points for the adaptive mesh, shape (nMeshPoints,2)
      raytrace   ... function mask=raytrace(para,mesh_points) that performs a raytrace
                       with the given Np parameters for a list of points of shape (nPoints,2)
                       returns list of points in image space, shape (nPoints,2)
      detectors  ... list of instances of Detector class for analyzing raytrace results
      weights    ... (opt) weights of contribution for each parameter set (default: constant)
    """
    self.parameters  = parameters;
    self.mesh_points = mesh_points;
    self.raytrace = raytrace;
    self.detectors = detectors;    
    nParams,Np = self.parameters.shape;
    if weights is None: weights = np.ones(nParams)/nParams;
    self.weights = weights;   
   
    
  def total_transmission(self, lthresh, Athresh=np.pi/1000):
    
    def is_broken(simplices):
        " local help function for defining which simplices should be subdivided"
        broken = Mesh.get_broken_triangles(simplices=simplices,lthresh=lthresh);
        area_broken = Mesh.get_area_in_domain(simplices=simplices[broken]);
        broken[broken] = (area_broken>Athresh);  # only consider triangles > Athresh as broken
        return broken;
        
    # incoherent sum on detector over all raytrace parameters
    for ip,p in enumerate(self.parameters):
      logging.info("Transmission for parameter: "+str(p));      
      
      # initialize adaptive grid for 
      mapping = lambda(mesh_points): self.raytrace(p,mesh_points);
      Mesh=AdaptiveMesh(self.mesh_points, mapping);  
      
      # iterative mesh refinement (subdivision of broken triangles)
      while True:  
        nNew = Mesh.refine_broken_triangles(is_broken,nDivide=100,bPlot=(ip==0));
        if nNew==0: break # converged, no additional subdivisions occured
        if ip==0: 
          skip = lambda(simplices): Mesh.get_broken_triangles(simplices=simplices,lthresh=lthresh)        
          Mesh.plot_triangulation(skip_triangle=skip);
          
      # update detectors
      broken = Mesh.get_broken_triangles(lthresh=lthresh);
      for d in self.detectors:
        d.add(Mesh,skip=broken,weight=self.weights[ip]);


def __test_intensity_footprint(hDDE):  
  
  # raytrace parameters
  image_surface = 22;
  wavenum  = 4;
  def raytrace(params, pupil_points):      # local function for raytrace
    x,y   = params;      
    px,py = pupil_points.T;                # shape (nPoints,)
    ret = hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
    vigcode = ret[:,[6]]<>0;               # include vignetting by shifting ray outside image
    return ret[:,[0,1]]+image_size*vigcode;# return (x,y) coordinates in image space

  # field sampling (octagonal fiber)
  xx,yy=cartesian_sampling(7,7,rmax=2);   # low: 7x7, high: 11x11
  ind = (np.abs(xx)<=1) & (np.abs(yy)<=1) & \
              (np.abs(xx+yy)<=np.sqrt(2)) & (np.abs(xx-yy)<=np.sqrt(2));
  field_sampling = np.vstack((xx[ind],yy[ind])).T;       # size (nFieldPoints,2)
  plt.figure(); plt.title("field sampling (normalized coordinates)");
  plt.plot(xx[ind].flat,yy[ind].flat,'.')
  plt.xlabel('x'); plt.ylabel('y');
  
  # pupil sampling (circular, adaptive mesh)
  px,py=fibonacci_sampling_with_circular_boundary(50,20) # low: (50,20), high: (200,50)
  pupil_sampling = np.vstack((px,py)).T;                 # size (nPoints,2)
  
  # set up image detector
  image_size=(0.2,0.05);  # [mm]
  img = RectImageDetector(extent=image_size,pixels=(201,401));
  dbg = CheckTriangulationDetector();
  
  # run Transmission calculation
  T = Transmission(field_sampling,pupil_sampling,raytrace,[dbg,img]);
  lthresh = 0.5*image_size[1];  
  T.total_transmission(lthresh)
  
  # plotting
  img.show();


def __test_angular_distribution(hDDE):
  # raytrace parameters
  image_surface = 22;
  wavenum  = 4;
  def raytrace(params, field_points):      # local function for raytrace
    px,py= params;      
    x,y = field_points.T;                  # shape (nPoints,)
    ret = hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
    vigcode = ret[:,[6]]<>0;               # include vignetting by shifting ray outside image
    return ret[:,[3,4]]+image_size*vigcode;# return (kx,ky) direction cosine in image space

  # field sampling (octagonal fiber, adaptive mesh)
  xx,yy=cartesian_sampling(21,21,rmax=2);  # low: 11x11, high: 7x7
  ind = (np.abs(xx)<=1) & (np.abs(yy)<=1) & \
              (np.abs(xx+yy)<=np.sqrt(2)) & (np.abs(xx-yy)<=np.sqrt(2));
  field_sampling = np.vstack((xx[ind],yy[ind])).T;       # size (nFieldPoints,2)
  
  # pupil sampling (cartesian grid with circular boundary)
  px,py=cartesian_sampling(7,7,rmax=1)     # low: 7x7, high: 11x11
  pupil_sampling = np.vstack((px,py)).T;                 # size (nPoints,2)
  plt.figure(); plt.title("pupil sampling (normalized coordinates)");
  plt.plot(px.flat,py.flat,'.')
  plt.xlabel('x'); plt.ylabel('y');
  
  # set up image detector (in angular space)
  image_size=(0.5,0.5);  # NA_max
  img = RectImageDetector(extent=image_size,pixels=(201,201));
  dbg = CheckTriangulationDetector();
  
  # run Transmission calculation
  T = Transmission(pupil_sampling,field_sampling,raytrace,[dbg,img]);
  lthresh = 0.5*image_size[1];  
  T.total_transmission(lthresh)
  
  # plotting
  img.show();


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
    #__test_intensity_footprint(hDDE);
    __test_angular_distribution(hDDE);
    