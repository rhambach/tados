# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 19:23:20 2016

@author: Hambach
"""
import numpy as np
import matplotlib.pylab as plt
import logging


class AdaptiveMesh(object):
  """
  Implementation of an adaptive mesh for a given mapping f:domain->image.
  We start from a Delaunay triangulation in the domain of f. This grid
  will be distorted in the image space. We refine the mesh by subdividing
  large or broken triangles. This process can be iterated.
  
  Note: currently, problems arise if a triangle is cut multiple times.
  ToDo: add unit tests
  """
  
  def __init__(self,initial_domain,mapping):
    """
    Initialize mesh, mapping and image points.
      initial_domain ... 2d array of shape (nPoints,2)
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
    
          
  def get_mesh(self):
    """ 
    return triangles and points in domain and image space
       domain,image:  coordinate array of shape (nPoints,2)
       simplices:     index array for vertices of each triangle, shape (nTriangles,3)
    Returns: (domain,image,simplices)
    """
    return self.domain,self.image,self.simplices;

  
  def plot_triangulation(self,skip_triangle=None):
    """
    plot current triangulation of adaptive mesh in domain and image space
      skip_triangle... (opt) function mask=skip_triangle(simplices) that accepts a list of 
                     simplices of shape (nTriangles, 3) and returns a flag 
                     for each triangle indicating that it should not be drawn
    returns figure handle;
    """ 
    simplices = self.simplices.copy();
    if skip_triangle is not None:
      skip = skip_triangle(simplices);
      skipped_simplices=simplices[skip];
      simplices=simplices[~skip];
          
    fig,(ax1,ax2)= plt.subplots(2);
    ax1.set_title("Sampling + Triangulation in Domain");
    if skip_triangle is not None and np.sum(skip)>0:
      ax1.triplot(self.domain[:,0], self.domain[:,1], skipped_simplices,'k:');
    ax1.triplot(self.domain[:,0], self.domain[:,1], simplices,'b-');    
    ax1.plot(self.initial_domain[:,0],self.initial_domain[:,1],'r.')
    
    ax2.set_title("Sampling + Triangulation in Image")
    ax2.triplot(self.image[:,0], self.image[:,1], simplices,'b-');
    ax2.plot(self.initial_image[:,0],self.initial_image[:,1],'r.')

    return fig;


  def get_area_in_domain(self,simplices=None):
    """
    calculate signed area of given simplices in domain space
      simplices ... (opt) list of simplices, shape (nTriangles,3)
    Returns:
      1d vector of size nTriangles containing the signed area of each triangle
      (positive: ccw orientation, negative: cw orientation of vertices)
    """
    if simplices is None: simplices = self.simplices;    
    x,y = self.domain[simplices].T;
    # See http://geomalgorithms.com/a01-_area.html#2D%20Polygons
    return 0.5 * ( (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]) );

  def get_area_in_image(self,simplices=None):
    """
    calculate signed area of given simplices in image space
    (see get_area_in_domain())
    """
    if simplices is None: simplices = self.simplices;    
    x,y = self.image[simplices].T;
    # See http://geomalgorithms.com/a01-_area.html#2D%20Polygons
    return 0.5 * ( (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]) );
  
  
  def get_broken_triangles(self,simplices=None,lthresh=None):
    """
    try to identify triangles that are cut or vignetted in image space
      simplices ... (opt) list of simplices, shape (nTriangles,3)  
      lthresh   ... (opt) threshold for longest side of broken triangle 
    Returns:
      1d vector of size nTriangles indicating if triangle is broken
    """
    if simplices is None: simplices = self.simplices;    
    # x and y coordinates for each vertex in each triangle 
    triangles = self.image[simplices]    
    # calculate maximum of (squared) length of two sides of each triangle 
    # (X[0]-X[1])**2 + (Y[0]-Y[1])**2; (X[1]-X[2])**2 + (Y[1]-Y[2])**2 
    max_lensq = np.max(np.sum(np.diff(triangles,axis=1)**2,axis=2),axis=1);
    # mark triangle as broken, if max side is 10 times larger than median value
    if lthresh is None: lthresh = 3*np.sqrt(np.median(max_lensq));
    return max_lensq > lthresh**2;
 
        
  def refine_large_triangles(self,is_large):
    """
    subdivide large triangles in the image mesh
      is_large ... function mask=is_large(triangles) that accepts a list of 
                     simplices of shape (nTriangles, 3) and returns a flag 
                     for each triangle indicating if it should be subdivided
                     
    Note: Additional points are added at the center of gravity of large triangles
          and the Delaunay triangulation is recalculated. Edge flips can occur.
    """
    # check if mesh is still a Delaunay mesh
    if self.__tri is None:
      raise RuntimeError('Mesh is no longer a Delaunay mesh. Subdivision not implemented for this case.');
    
    ind = is_large(self.simplices);
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
        
     

  def refine_broken_triangles(self,is_broken,nDivide=10,bPlot=False,bPlotTriangles=[7]):
    """
    subdivide triangles which contain discontinuities in the image mesh
      is_broken  ... function mask=is_broken(triangles) that accepts a list of 
                      simplices of shape (nTriangles, 3) and returns a flag 
                      for each triangle indicating if it should be subdivided
      nDivide    ... (opt) number of subdivisions of each side of broken triangle
      bPlot      ... (opt) plot sampling and selected points for debugging 
      bPlotTriangles (opt) list of triangle indices for which segmentation should be shown

    Note: The resulting mesh will be no longer a Delaunay mesh (identical points 
          might be present, circumference rule not guaranteed). Mesh functions, 
          that need this property (like refine_large_triangles()) will not work
          after calling this function.
    """
    broken = is_broken(self.simplices);
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
      for t in bPlotTriangles: # select index of triangle to look at
        BCA=[B[t],C[t],A[t]]; subdiv=new_simplices[t::nTriangles,:];
        pt=self.domain[BCA]; ax1.plot(pt[...,0],pt[...,1],'g')
        pt=self.image[BCA];  ax2.plot(pt[...,0],pt[...,1],'g')
        pt=self.domain[subdiv]; ax1.plot(pt[...,0],pt[...,1],'r')
        pt=self.image[subdiv];  ax2.plot(pt[...,0],pt[...,1],'r')

    # we remove degenerated triangles (p1..4 identical ot A,B or C) 
    # and orient all triangles ccw in domain before adding them to the list of simplices
    area = self.get_area_in_domain(new_simplices);
    new_simplices[area<0] = new_simplices[area<0,::-1]; # reverse cw triangles
    new_simplices = new_simplices[area<>0];             # remove degenerate triangles

    # sanity check that total area did not change after segmentation
    old = np.sum(np.abs(self.get_area_in_domain(simplices)));
    new = np.sum(np.abs(self.get_area_in_domain(new_simplices)));
    assert(abs((old-new)/old)<1e-10) # segmentation of triangle has no holes/overlaps
      
    # update simplices in mesh    
    self.__tri = None; # delete initial Delaunay triangulation        
    self.simplices=np.vstack((self.simplices[~broken], new_simplices)); # no longer Delaunay
