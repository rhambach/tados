# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 10:27:31 2016

interesting links:
http://toblerity.org/shapely/manual.html
http://stackoverflow.com/questions/14697442/faster-way-of-polygon-intersection-with-shapely

@author: Hambach
"""
import abc
import numpy as np

from PyOptics.raytrace2d.raytrace import Rays
from PyOptics.raytrace2d.common import init_list1d

class Surface(object):
  " abstract base class for all surfaces, defines interface only "

  __metaclass__ = abc.ABCMeta
  
  @abc.abstractmethod
  def info(self, verbosity=0):
    """ 
    return info string describing the surface in a lens data editor,
    verbosity ... 0: one-liner, 1: +parameters, 2: debug
    """
    return;
    
  @abc.abstractmethod    
  def trace_behind(self,rays,n_before):
    " trace given set of rays just behind the surface and return new rays "
    return; 

  def get_surface_data(self):
    " return (y,z) coordinates specifying the surface or (None,None) "
    return None,None;
   
  def get_refractive_index(self,n_before):
    " index of refraction after surface, same as before for any dummy surface"
    return n_before; 
   

class PropagateDistance(Surface):
  " not an actual surface, propagate rays in the current medium by a distance d"
  def __init__(self,d):
    """
      d  : float
        propagation distance along current direction of the ray
    """
    self.d = d;
    
  def info(self,verbosity=0):
    descr = "Propagation by a distance"
    if verbosity>0: descr += " d=%f"%self.d;
    return descr;
 
  def trace_behind(self,rays, n_before): 
    zp = rays.z + self.d*rays.vz;
    yp = rays.y + self.d*rays.vy;
    vig= np.zeros(rays.num,dtype=np.bool);   # no vignetting
    return Rays(zp,yp,rays.vz,rays.vy), vig;
 



class PlaneSurface(Surface):

  def __init__(self, A, B, n=None):
    """
      tilted, planar surface segment with end-points A and B
    
      Parameters
      ----------
        A,B :  pairs of floats
          A=(y1,z1) and B=(y2,z2) are the two end points of the surface
        n   : float, optional
          refractive index of the medium after the surface, default: same as before
    """
    self.A=tuple(A);
    self.B=tuple(B);
    self.n_after=n;
    
  def info(self,verbosity=0):
    descr = "Plane Surface";
    if self.n_after is None: descr="Dummy " + descr;
    if verbosity>0: descr += " passing through points A=(%f,%f)"%self.A + \
                             " and B=(%f,%f),"%self.B;
    return descr;    
    
  def trace_behind(self, ray, n_before):
    """
      perform sequential raytrace behind the plane surface     
    
      Parameters
      ----------
        rays     : Rays object
          list of rays before surface
        n_before : float
          index of refraction before surface
          
      Returns
      -------
        new_rays : Rays object
          list of rays after surface
        vig      : list if booleans
          indicates for each ray, if it is vignetted (True) or not (False)
          
      Notes
      -----
      see: https://www.topcoder.com/community/data-science/data-science-tutorials/geometry-concepts-line-intersection-and-its-applications/
    """    
    # propagate rays right behind the plane interface
    (Ay,Az)=self.A; (By,Bz)=self.B;
    n_after = self.get_refractive_index(n_before);

    # calculate surface direction vector s=(B-A)
    sz = Bz-Az;     
    sy = By-Ay;
     
    # determine intersection point (zp,yp) of the ray with the line segment (in barycentric coords)
    #   r + alpha*v = A + beta*s;    r = (ray.z,ray.y);
    # to this end, we solve the matrix eq.
    #   |vz sz| |-alpha| = |rz-Az|     det*|-alpha| =  | sy -sz| |rz-Az|
    #   |vy sy| | beta |   |ry-Ay| ;       | beta |    |-vy  vz| |ry-Ay|
    det = ray.vz*sy-ray.vy*sz;      # corresponds to cross product (vxs) between ray and surface segment, 
                                    # should be >  0 in standard case  
    beta = (-ray.vy*(ray.z-Az)+ray.vz*(ray.y-Ay))/det;
    # if det==0: ray misses surface (both are parallel), beta/det is inf
    # if beta/det<0 or beta/det>1, the ray does not intersect between A and B
    vig = ~np.logical_and(beta>=0, beta<1);    
    zp = Az + beta*sz;
    yp = Ay + beta*sy;
    
    # change ray angle according to law of refraction
    # we avoid the calculation of sin(theta) altogether by evaluating
    #   v*s/|s| = cos(angle(v,s))=sin(theta),
    # where the sign indicates the orientation of theta measured from the surface normal
    # pointing into the medium before the interface
    s = np.sqrt(sz**2+sy**2);
    sin_theta = (ray.vz*sz+ray.vy*sy)/s;
    sin_thetap= (n_before/n_after)*sin_theta;    # law of refraction
    # theta' is measured from surface normal pointing into medium after interface
  
    # determine new direction of outgoing ray by rotating outgoing surface normal
    # by theta' in ccw direction (theta' can be negative!)
    # Note: cos theta' > 0 (outgoing ray points in direction of outgoing surface normal)
    nz = sy/s; ny = -sz/s;
    cos_thetap= np.sqrt(1-sin_thetap**2);  
    vpz = cos_thetap*nz - sin_thetap*ny;
    vpy = sin_thetap*nz + cos_thetap*ny;
    
    # append to raypath
    return Rays(zp,yp,vpz,vpy), vig;
    
  def get_refractive_index(self,n_before):
    return self.n_after if self.n_after is not None else n_before;   
    
  def get_surface_data(self):
    " return (y,z) coordinates specifying the surface or (None,None) "
    return np.transpose([self.A,self.B]);
    
    
    
    
class SegmentedSurface(Surface):

  def __init__(self, y, z, n=None, allow_virtual=True):
    """
      piecewise linear surface sampled by the vertices (y,z)
    
      Parameters
      ----------
        y,z   :  arrays of floats,
          global coordinates of the vertices of the surface
        n     : float, optional
          refractive index of the medium after the surface, default: same as before
        allow_virtual : flag, optional
          rays can be propagated backwards by default, set flag to false to 
          allow only propagation of real rays
    """
    y,z = np.atleast_1d(y,z);
    self.num = max(y.size,z.size);
    self.y = init_list1d(y,self.num,np.double,'y');
    self.z = init_list1d(z,self.num,np.double,'z');
    self.n_after=n;
    self.allow_virtual = allow_virtual;

  def info(self,verbosity=0):
    descr = "Segmented Surface";
    if self.n_after is None: descr="Dummy " + descr;
    if verbosity>0: descr += " passing through %d sampling points"%self.num;
    return descr;    
    
  def trace_behind(self, rays, n_before):
    """
      perform sequential raytrace behind the plane surface     
    
      Parameters
      ----------
        rays : Rays object
          list of rays before surface
        n_before : float
          index of refraction before surface
          
      Returns
      -------
        new_rays : Rays object
          list of rays after surface
        vig : list if booleans
          indicates for each ray, if it is vignetted (True) or not (False)
    """
    n_after = self.get_refractive_index(n_before);

    # initialize output rays and vignetting flag    
    ret = Rays.empty(rays.num);            
    vig = np.ones(rays.num,dtype=np.bool);
    alpha_last = np.full(rays.num,np.inf);
       
    # iterate over all segments
    for i in xrange(self.num-1):

      Ay = self.y[i];   Az = self.z[i];
      By = self.y[i+1]; Bz = self.z[i+1];
    
      # propagate rays right behind the segment
      # see PlaneSurface.trace_behind() for explanation
      sy = By-Ay;       sz = Bz-Az;
      det = rays.vz*sy-rays.vy*sz;      # corresponds to cross product (vxs) between ray and surface segment, 
      with np.errstate(divide='ignore'):# divide by zero if ray and surface is parallel
        beta = (-rays.vy*(rays.z-Az)+rays.vz*(rays.y-Ay))/det;  
            
      # select only rays that intersect the current segment, stop if no ray hits segment
      bHit = np.logical_and(beta>=0, beta<1);    
      if not np.any(bHit): continue;
      beta = beta[bHit]
      rvz = rays.vz[bHit]; rvy = rays.vy[bHit];
      
      # calculate intersection points
      zp = Az + beta*sz;
      yp = Ay + beta*sy;

      # check if ray propagates forward
      if not self.allow_virtual:
        # propagation length alpha along ray r + alpha * v (only for rays that hit segment)
        alpha = ( -sy*(rays.z[bHit]-Az) + sz*(rays.y[bHit]-Ay) ) / det[bHit];
        # check if ray propagates forward (for n>0) or backward (for n<0), at least a little bit
        bForward = alpha*n_before > 1e-10 
        # accept ray intersection only if ray is pointing forward and 
        # absolute path length alpha is smaller than before        
        alpha = np.abs(alpha)
        bHit[bHit]=ind=np.logical_and(bForward, alpha<alpha_last[bHit]);
        # update propagation length and all other quantities for accepted rays        
        if not np.any(bHit): continue;
        alpha_last[bHit]=alpha[ind];        
        rvz=rvz[ind]; rvy=rvy[ind];
        zp =zp[ind];  yp =yp[ind];
        
      # change ray angle according to law of refraction
      s = np.sqrt(sz**2+sy**2);
      sin_theta = (rvz*sz+rvy*sy)/s;
      sin_thetap= (n_before/n_after)*sin_theta;    # law of refraction
      
      # determine new direction of outgoing ray by rotating outgoing surface normal
      nz = sy/s; ny = -sz/s;
      cos_thetap= np.sqrt(1-sin_thetap**2);  
      vzp = cos_thetap*nz - sin_thetap*ny;
      vyp = sin_thetap*nz + cos_thetap*ny;
    
      # append rays
      ret.z[bHit]=zp; ret.y[bHit]=yp;
      ret.vz[bHit]=vzp; ret.vy[bHit]=vyp;
      vig[bHit]=False;   # not vignetted if we hit segment
    
    # set vignetted rays to initial starting point
    ret.z[vig] = rays.z[vig]; ret.vz[vig] = rays.vz[vig];
    ret.y[vig] = rays.y[vig]; ret.vy[vig] = rays.vy[vig];
    return ret,vig;
    
  def get_refractive_index(self,n_before):
    return self.n_after if self.n_after is not None else n_before;   
    
  def get_surface_data(self):
    " return (y,z) coordinates specifying the surface or (None,None) "
    return np.vstack([self.y,self.z]);