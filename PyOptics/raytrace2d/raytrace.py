# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

from PyOptics.raytrace2d.common import init_list1d

class Rays(object):
  
  def __init__(self,z=None,y=None,vz=None,vy=None):
    """
    describe list of rays starting from point (z,y) in direction (vz,vy)
    
    Parameters
    ----------
      z : scalar or list of floats
        position along optical axis
      y : scalar or list of floats
        ray height (distance to optical axis)
      vz : scalar or list of floats, optional
        z-component of normalized ray direction kz/k0 (cosine of the ray inclination angle)
        if None, a component along +z is calculated from vy.
      vy : scalar or list of floats, optional
        y-component of normalized ray direction ky/k0 (sine of ray inclination angle)
        if None, a component along +y is calculated from vz.
    """
    # calculate missing direction coordinates
    if vz is None and vy is None: raise RuntimeError("direction of rays (vz,vy) is not given");
    if vz is None: vz = np.sqrt(1-vy**2);   # choose vz>0 
    if vy is None: vy = np.sqrt(1-vz**2);   # choose vy>0
    # make all arrays 1d of same length
    z,y,vz,vy = np.atleast_1d(z,y,vz,vy);
    self.num = max(z.size,y.size,vz.size,vy.size);
    self.z = init_list1d(z,self.num,np.double,'z');
    self.y = init_list1d(y,self.num,np.double,'y');
    self.vz = init_list1d(vz,self.num,np.double,'vz');
    self.vy = init_list1d(vy,self.num,np.double,'vy');

  @classmethod
  def empty(cls,nRays):
    " create empty Ray object for nRays "
    z = np.full(nRays,np.nan);
    y = np.full(nRays,np.nan);
    vz= np.ones(nRays);
    return cls(z,y,vz=vz);


  def __iter__(self):
    " return stacked numpy array for iteration over all rays "
    return np.stack((self.y,self.z,self.vy,self.vz));

class Raytracer(object):

  def __init__(self,source,system):
    """
      source  : class Source()
        source of the optical system
      system  : list of Surface-objects
        sequential optical system defined as list of Surface-objects
    """
    self.system=system;
    self.source=source;
    self.raypath=[];
    self.vignetted_at_surf=[];
    
  def trace(self,nRays=7,calc_opl=True):
    " trace number of rays"
    # initial rays emerging from source
    n_before = self.source.get_refractive_index();
    rays=self.source.get_rays(nRays);
    self.vignetted_at_surf = np.full(nRays,len(self.system),dtype=np.int);
    self.raypath=[rays];
    self.n = np.empty(len(self.system));

    # trace rays behind each surface in the optical system
    for num,surface in enumerate(self.system):
      # raytrace      
      rays,vig=surface.trace_behind(rays,n_before);
      # check if ray is vignetted
      if np.any(vig):
        self.vignetted_at_surf[vig] = np.minimum(self.vignetted_at_surf[vig], num);
      # save data
      self.raypath.append(rays);
      self.n[num]=n_before;
      n_before=surface.get_refractive_index( n_before );  
        
  
  def print_system(self,verbosity=0):
    print("Source   %s"%self.source.info(verbosity))
    for i,surface in enumerate(self.system):
      print("Surf%2d   %s"%(i,surface.info(verbosity)));
    
 