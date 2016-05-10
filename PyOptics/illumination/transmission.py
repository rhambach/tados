# -*- coding: utf-8 -*-
"""

Provides several classes for calculation of transmission through an optical system

Transmission: perform the transmission calculation (iterate over discrete set of
  parameters (e.g. field points) and evaluate the transmission on an adaptive mesh
  (e.g. pupil coordinates) using a given raytrace function
Detectors: analyze the raytrace results

ToDo: add unit tests

@author: Hambach
"""

from __future__ import division
import abc
import logging
import numpy as np
import matplotlib.pylab as plt

from PyOptics.illumination.point_in_triangle import point_in_triangle
from PyOptics.illumination.adaptive_mesh import *
from PyOptics.zemax.dde_link import *

class Detector(object):
  __metaclass__ = abc.ABCMeta
  @abc.abstractmethod
  def add(self,mesh,bSkip=[],weight=1): return;
  @abc.abstractmethod  
  def show(self): return;


class CheckTriangulationDetector(Detector):
  " Detector class for testing completeness of triangulation in domain"

  def __init__(self, ref_area=np.pi):
    """
    ref_area ... (opt) theoretical area of domain space, default: area of unit circle
    """
    self.ref_domain_area=ref_area;
  
  def add(self,mesh,bSkip=[],weight=1):
    """
    calculate total domain area of mesh and print logging info 
      mesh ... instance of AdaptiveMesh 
      bSkip... logical array indicating simplices that should be skipped
      weight.. ignored
    """
    triangle_area  = mesh.get_area_in_domain(); # domain area in current mesh
    assert(all(triangle_area>0));  # triangles should be oriented ccw in mesh 
    
    # check error of initial sampling of domain boundary    
    err_boundary= 1-mesh.initial_domain_area/self.ref_domain_area;
    out = 'estimated loss of power due to triangulation of mesh: \n' + \
     '  %5.3f%% due to initial approximation of mesh boundary \n'%(err_boundary*100);   
    # check error due to invalid triangles (raytrace errors for one of its vertices)
    tot_triangle_area= np.sum(np.abs(triangle_area));
    err_invalid = 1-tot_triangle_area / mesh.initial_domain_area;
    if np.abs(err_invalid)>1e-8:
      out += '  %5.3f%% due to invalid triangles (raytrace errors)\n' %(err_invalid*100);
    if np.any(bSkip):
      err_skip  = np.sum(triangle_area[bSkip]) / mesh.initial_domain_area;
      out += '  %5.3f%% due to broken triangles (contain discontinuity) \n' %(err_skip*100);
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
     
     Note, 'ij' indexing is used,  i.e,. self.points[:,ix,iy]=(x,y)
    """
    self.extent = np.asarray(extent);
    self.pixels = np.asarray(pixels);
    # cartesian sampling 
    xmax,ymax = self.extent/2.; nx,ny = self.pixels;    
    xbins = np.linspace(-xmax,xmax,nx+1);         # edges of nx pixels  
    ybins = np.linspace(-ymax,ymax,ny+1);
    x = 0.5*(xbins[:-1]+xbins[1:]);               # centers of nx pixels
    y = 0.5*(ybins[:-1]+ybins[1:]);    
    self.points = np.asarray(np.meshgrid(x,y,indexing='ij')); # shape: (2,numx,numy)
    self.intensity = np.zeros(self.pixels);                   # shape: (numx,numy)

  def add(self,mesh,bSkip=[],weight=1):
    """
    calculate footprint in image plane
      mesh ... instance of AdaptiveMesh 
      bSkip... logical array indicating simplices that should be skipped
      weight.. weight of contribution (intensity in Watt)
    """
    domain_area = mesh.get_area_in_domain();
    domain_area/= mesh.initial_domain_area;       # normalized weight in domain
    image_area  = mesh.get_area_in_image();       # size of triangle in image
    density = weight * abs( domain_area / image_area);
    for s,simplex in enumerate(mesh.simplices):
      if len(bSkip)>0 and bSkip[s]: continue
      triangle = mesh.image[simplex];
      mask = point_in_triangle(self.points,triangle);
      self.intensity += density[s]*mask;

  def show(self,fMask=None):
    """
    plotting 2D footprint and projected intensities in image plane
      fMask ... (opt) function bMask = fMask(x,y) specifies for each pixel, if
                      it contributes to the signal (bMask=0) or not (bMask=1)
    Return: figure handle
    """
    X,Y,intensity = self.get_footprint(fMask); 
    x,xprofile = self.x_projection(fMask);
    y,yprofile = self.y_projection(fMask);    
    # footprint    
    fig,(ax1,ax2)= plt.subplots(2);
    ax1.set_title("footprint in image plane");
    ax1.imshow(intensity.T,origin='lower',aspect='auto',interpolation='hanning',
             extent=np.outer(self.extent,[-0.5,0.5]).flatten());
    # projections    
    ax2.set_title("projected intensity in image plane");    
    ax2.plot(x,xprofile,label="along x");
    ax2.plot(y,yprofile,label="along y");
    ax2.legend(loc=0)
    # total intensity
    dx = x[1]-x[0]; dy = y[1]-y[0];
    if fMask is None: 
      assert(np.allclose(np.nansum(intensity)*dx*dy, 
                        [np.sum(xprofile)*dx,np.sum(yprofile)*dy])); # total power must be the same
    logging.info('total power: %5.3f W'%(np.sum(intensity)*dx*dy)); 
    return fig

  def get_footprint(self,fMask=None):
    """
    returns X,Y,intensity:
      fMask ... (opt) function bMask = fMask(x,y) specifies for each pixel, if
                      it contributes to the signal (bMask=0) or not (bMask=1)
    Return:
      X,Y      ... coordinates of the detector, shape (xnum,ynum)
      intensity... 2d intensity from detector,  shape (xnum,ynum)
    """
    Nx,Ny = self.pixels;
    X,Y   = self.points.reshape(2,Nx,Ny);
    image_intensity = self.intensity.reshape(Nx,Ny).copy();
    if fMask is not None:
      image_intensity[fMask(X,Y)] = 0
    return X,Y,image_intensity
    
  def x_projection(self,fMask=None):
    """
    Calculate projection on x-axis.
      fMask ... (opt) function bMask = fMask(x,y) specifies for each pixel, if
                      it contributes to the signal (bMask=0) or not (bMask=1)
    Return: x, xprofile, shape: (xnum,)
    """
    X,Y,intensity = self.get_footprint(fMask);
    xaxis = X[:,0];  yaxis = Y[0,:];  dy = yaxis[1]-yaxis[0]; 
    xprofile = np.nansum(intensity,axis=1)*dy; # integral over y
    return xaxis,xprofile;
    
  def y_projection(self,fMask=None):
    """
    Calculate projection on y-axis.
      fMask ... (opt) function bMask = fMask(x,y) specifies for each pixel, if
                      it contributes to the signal (bMask=0) or not (bMask=1)
    Return: y, yprofile, shape: (ynum,)
    """
    X,Y,intensity = self.get_footprint(fMask);
    xaxis = X[:,0];  yaxis = Y[0,:];  dx = xaxis[1]-xaxis[0]; 
    yprofile = np.nansum(intensity,axis=0)*dx; # integral over x  
    return yaxis,yprofile;
    
class PolarImageDetector(Detector):    
  """
  2D Image Detector with polar coordinates
  Todo: correct coordinates of pixels to coincide with center 
  (rmax denotes extent, i.e, pixel edge, while coordinates refer to pixel centers)
  """
  def __init__(self, rmax=1, nrings=100):
    """
     rmax ... radial size of detector in image space
     nrings.. number of rings
    """
    self.rmax = rmax;
    self.nrings = nrings;
    ret = hexapolar_sampling(nrings,rmax=rmax,ind=True); 
    self.points = np.asarray(ret[0:2]);     # shape: (2,nPixels)
    self.points_per_ring = ret[2];          # shape: (nrings,)
    self.weight_of_ring  = ret[3];          # shape: (nrings,)
    self.intensity = np.zeros(self.points.shape[1]);  # 1d array

  def add(self,mesh,bSkip=[],weight=1):
    """
    calculate footprint in image plane
      mesh ... instance of AdaptiveMesh 
      bSkip... logical array indicating simplices that should be skipped
      weight.. weight of contribution (intensity in Watt)
    """
    domain_area = mesh.get_area_in_domain(); 
    domain_area/= mesh.initial_domain_area;       # normalized weight in domain
    image_area  = mesh.get_area_in_image();       # size of triangle in image
    density = weight * abs( domain_area / image_area);
    for s,simplex in enumerate(mesh.simplices):
      if len(bSkip)>0 and bSkip[s]: continue
      triangle = mesh.image[simplex];
      mask = point_in_triangle(self.points,triangle);
      self.intensity += density[s]*mask;

  def show(self):
    " plotting 2D footprint in image plane, returns figure handle"
    fig,(ax1,ax2)= plt.subplots(2);
    # footprint
    ax1.set_title("footprint in image plane");
    ax1.tripcolor(self.points[0],self.points[1],self.intensity);
    # azimuthal average
    r, radial_profile, encircled_energy = self.radial_projection();
    ax2.set_title("radial profile");    
    ax2.plot(r,radial_profile,label='radial profile');
    ax2.plot(r,encircled_energy,label='encircled energy');
    ax2.legend(loc=0)
    logging.info('total power: %5.3f W'%(encircled_energy[-1])); 
    return fig

  def get_footprint(self):
    """
    returns X,Y,intensity:
      X,Y      ... coordinates of the detector, shape (nPoints)
      intensity... 2d intensity from detector,  shape (nPoints)
    """
    return self.points[0], self.points[1], self.intensity.copy()
    
  def radial_projection(self):
    """
    Calculate azimuthal avererage over detector (radial projection).
    Return: r, radial_profile, encircled_energy, shape: (nrings,)
    """
    Nr=np.insert(np.cumsum(self.points_per_ring),0,0); # index array for rings, size (nrings+1,)
    # radial profile    
    radial_profile = np.empty(self.nrings);
    r = np.empty(self.nrings);
    for i in xrange(self.nrings):
      radial_profile[i] = np.sum(self.intensity[Nr[i]:Nr[i+1]]) / self.points_per_ring[i];
      r2 = np.sum(self.points[:,Nr[i]:Nr[i+1]]**2,axis=0)
      assert np.allclose(r2[0],r2);
      r[i] = np.sqrt(r2[0]);
    # encircled energy (area of ring = weight of ring x total area of detector)
    encircled_energy = np.cumsum(radial_profile*self.weight_of_ring*np.pi*self.rmax**2);
    return r, radial_profile, encircled_energy


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
      
      # subdivision of invalid triangles (raytrace failed for some vertices)
      Mesh.refine_invalid_triangles(nDivide=100,bPlot=(ip==0));
      
      # iterative mesh refinement (subdivision of broken triangles)
      while True:  
        if ip==0: # plot mesh for first set of parameters
          skip = lambda(simplices): Mesh.get_broken_triangles(simplices=simplices,lthresh=lthresh)        
          Mesh.plot_triangulation(skip_triangle=skip);
        # refine mesh until nothing changes
        nNew = Mesh.refine_broken_triangles(is_broken,nDivide=100,bPlot=(ip==0));        
        if nNew==0: break 
          
      # update detectors
      broken = Mesh.get_broken_triangles(lthresh=lthresh);
      for d in self.detectors:
        d.add(Mesh,bSkip=broken,weight=self.weights[ip]);


