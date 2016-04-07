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
#from matplotlib.path import Path
import point_in_triangle

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
    self.__tri = Delaunay(initial_domain,incremental=True);
    self.simplices = self.__tri.simplices;
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
    simplices = self.simplices.copy();
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

  def refine_broken_triangles(self,is_broken,nDivide=10,bPlot=False):
    """
    subdivide triangles which contain discontinuities in the image mesh
      is_broken ... function mask=is_broken(triangles) that accepts a list of 
                     triangle vertices of shape (nTriangles, 3, 2) and returns 
                     a flag for each triangle indicating if it should be subdivided
      nDivide   ... (opt) number of subdivisions of each side of broken triangle
      bPlot     ... (opt) plot sampling and selected points for debugging 

    Note: The resulting mesh will be no longer a Delaunay mesh (identical points 
          might be present, circumference rule not guaranteed). Mesh functions, 
          that need this property (like refine_large_triangles()) will not work
          after calling this function.
    
    Todo: we might save 33% - 66% of the effort on calculating the mapping between
          domain and image (here: ray-trace) by recognizing identical sampling points
          and selecting the two broken sides of the triangle in advance
    """
    broken = is_broken(self.image[self.simplices]);
    nTriangles = np.sum(broken)
    if nTriangles==0: return;                 # noting to do!
    nPointsOrigMesh = self.image.shape[0];  
    
    # add new simplices:
    # segmentation of each broken triangle is generated in a cyclic manner,
    # starting with isolated point C and the two closest new sampling points
    # in image space, p1 + p2), continues with p3,p4,A,B.
    #    
    #             C
    #             /\
    #            /  \              \\\ largest segments of triangle in image space
    #        p1 *    *  p2          *  new sampling points
    #     ....///....\\\.............. discontinuity
    #      p3 *        * p4
    #        /          \          new triangles:
    #       /____________\           (C,p1,p2),             isolated point + closest two new points  
    #      A              B          (p1,p2,p3),(p2,p3,p4)  new broken triangles, only between new sampling points
    #                                (p3,p4,A), (p4,A,B):   rest
    # 
    # Note: one has to pay attention, that C,p1,p3,A are located on same side
    #       of the triangle, otherwise the partition will fail!     

    # identify the shortest edge of the triangle in image space (not cut)
    simplices = self.simplices[broken];                    # shape (nTriangles,3)
    triangles = self.image[simplices];                     # shape (nTriangles,3,2)
    vertices = np.concatenate((triangles,triangles[:,[0],:]),axis=1); # shape (nTriangles,4,2)
    edge_len = np.sum( np.diff(vertices,axis=1)**2, axis=2); # shape (nTriangles,3)
    min_edge = np.argmin( edge_len,axis=1);                # shape (nTriangles)
 
    # get indices of points ABC as shown above (C is isolated point)
    ind_triangle = np.arange(nTriangles)
    A = simplices[ind_triangle,min_edge];           # first point of shortest side
    B = simplices[ind_triangle,(min_edge+1)%3];     # second point of shortest side
    C = simplices[ind_triangle,(min_edge+2)%3];     # point opposit to shortest side
    
    # create dense sampling along C->B and C->A in domain space
    x = np.linspace(0,1,nDivide,endpoint=True);
    CA = np.outer(1-x,self.domain[C]) + np.outer(x,self.domain[A]);
    CB = np.outer(1-x,self.domain[C]) + np.outer(x,self.domain[B]);
                                                    # sampling of shape (Ndivide,nTriangles*2)    
    # map sampling on CA and CB to image space and measure length of segments in image space
    domain_points= np.hstack((CA,CB)).reshape(nDivide,2,nTriangles,2);
    image_points = self.mapping(domain_points.reshape(-1,2)).reshape(nDivide,2,nTriangles,2);
    len_segments = np.sum(np.diff(image_points,axis=0)**2,axis=-1); 
                                                    # shape (nDivide-1,2,nTriangle)
    # determine indices of broken segments (largest elements in CA and CB)
    largest_segments = np.argmax(len_segments,axis=0); # shape (2,nTriangle) 
    edge_points = np.asarray((largest_segments,largest_segments+1));
                                                    # shape (2,2,nTriangle)
 
    # set points p1 ... p4 for segmentation of triangle
    # see http://stackoverflow.com/questions/15660885/correctly-indexing-a-multidimensional-numpy-array-with-another-array-of-indices
    idx_tuple = (edge_points[...,np.newaxis],) + tuple(np.ogrid[:2,:nTriangles,:2]);
    new_domain_points = domain_points[idx_tuple];
    new_image_points  = image_points[idx_tuple];    
              # shape (2,2,nTriangle,2), indicating iDistance,iEdge,iTriangle,(x/y)
 
    # update points in mesh (points are no longer unique!)
    logging.info("refining_broken_triangles(): adding %d points"%(4*nTriangles));
    self.image = np.vstack((self.image,new_image_points.reshape(-1,2))); 
    self.domain= np.vstack((self.domain,new_domain_points.reshape(-1,2)));    
   
    if bPlot:   
      fig = self.plot_triangulation(skip_triangle=is_broken);
      ax1,ax2 = fig.axes;
      ax1.plot(domain_points[...,0].flat,domain_points[...,1].flat,'k.',label='test points');
      ax1.plot(new_domain_points[...,0].flat,new_domain_points[...,1].flat,'g.',label='selected points');
      ax1.legend(loc=0);      
      ax2.plot(image_points[...,0].flat,image_points[...,1].flat,'k.')
      ax2.plot(new_image_points[...,0].flat,new_image_points[...,1].flat,'g.',label='selected points');   
    
    # indices for points p1 ... p4 in new list of points self.domain 
    # (offset by number of points in the original mesh!)
    # Note: by construction, the order of p1 ... p4 corresponds exactly to the order
    #       shown above (first tuple contains points closest to C,
    #       first on CA, then on CB, second tuple beyond the discontinuity)
    (p1,p2),(p3,p4) = np.arange(4*nTriangles).reshape(2,2,nTriangles) + nPointsOrigMesh;
                                                    # shape (nTriangles,)
    # construct the five triangles from points
    t1=np.vstack((C,p1,p2));                        # shape (3,nTriangles)
    t2=np.vstack((p1,p2,p3));
    t3=np.vstack((p2,p3,p4));
    t4=np.vstack((p3,p4,A));
    t5=np.vstack((p4,A,B));
    new_simplices = np.hstack((t1,t2,t3,t4,t5)).T;  
       # shape (5*nTriangles,3), reshape as (5,nTriangles,3) to obtain subdivision of each triangle  

    # DEBUG subdivision of triangles
    if bPlot:
      t=7;  # select index of triangle to look at
      BCA=[B[t],C[t],A[t]]; subdiv=new_simplices[t::nTriangles,:];
      pt=self.domain[BCA]; ax1.plot(pt[...,0],pt[...,1],'g')
      pt=self.image[BCA];  ax2.plot(pt[...,0],pt[...,1],'g')
      pt=self.domain[subdiv]; ax1.plot(pt[...,0],pt[...,1],'r')
      pt=self.image[subdiv];  ax2.plot(pt[...,0],pt[...,1],'r')

    # we remove degenerated triangles (p1..4 identical ot A,B or C) 
    # and orient all triangles ccw in domain before adding them to the list of simplices
    area = get_area(self.domain[new_simplices]);
    new_simplices[area<0] = new_simplices[area<0,::-1]; # reverse cw triangles
    new_simplices = new_simplices[area<>0];             # remove degenerate triangles

    # sanity check that total area did not change after segmentation
    old = np.sum(np.abs(get_area(self.domain[simplices])));
    new = np.sum(np.abs(get_area(self.domain[new_simplices])))
    assert(abs((old-new)/old)<1e-10) # segmentation of triangle has no holes/overlaps
      
    # update simplices in mesh    
    self.__tri = None; # delete initial Delaunay triangulation        
    self.simplices=np.vstack((self.simplices[~broken], new_simplices)); # no longer Delaunay

   
    
    
  def refine_large_triangles(self,is_large):
    """
    subdivide large triangles in the image mesh
      is_large ... function mask=is_large(triangles) that accepts a list of 
                     triangle vertices of shape (nTriangles, 3, 2) and returns 
                     a flag for each triangle indicating if it should be subdivided
                     
    Note: Additional points are added at the center of gravity of large triangles
          and the Delaunay triangulation is recalculated. Edge flips can occur.
    """
    # check if mesh is still a Delaunay mesh
    if self.__tri is None:
      raise RuntimeError('Mesh is no longer a Delaunay mesh. Subdivision not implemented for this case.');
    
    ind = is_large(self.image[self.simplices]);
    if np.sum(ind)==0: return; # nothing to do
    
    # add center of gravity for critical triangles
    new_domain_points = np.sum(self.domain[self.simplices[ind]],axis=1)/3;
    self.__tri.add_points(new_domain_points);
    logging.info("refining_large_triangles(): adding %d points"%(new_domain_points.shape[0]))
    
    # calculate image points and update data
    new_image_points = self.mapping(new_domain_points);
    self.image = np.vstack((self.image,new_image_points));
    self.domain= np.vstack((self.domain,new_domain_points));
    self.simplices = self.__tri.simplices;
        
        
  def get_mesh(self):
    return self.domain,self.image,self.simplices;



class AnalyzeTransmission(object):

  def __init__(self, hDDE):
    self.hDDE=hDDE;


  def test(self):  
    # set up ray-trace parameters and image detector
    image_surface = 22;
    wavenum  = 1;
    image_size = np.asarray((0.2,0.05));
    image_size = np.asarray((0.2,0.05));
    image_shape = np.asarray((501,501));
    img_pixels = cartesian_sampling(*image_shape,rmax=2); # shape: (2,nPixels)
    img_pixels*= image_size[:,np.newaxis]/2;
    image_intensity = np.zeros(np.prod(image_shape)); # 1d array

    # field sampling
    xx,yy=cartesian_sampling(3,3,rmax=.1)  
    for i in xrange(len(xx)):
      x=xx[i]; y=yy[i];
      print("Field point: x=%5.3f, y=%5.3f"%(x,y))
      
      # init adaptive mesh for pupil sampling
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
        #path = Path( image_points[simplices[s]] );
        #mask = path.contains_points(img_pixels);
        print img_pixels.shape
        mask = point_in_triangle.points_in_triangle(img_pixels,triangle);
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