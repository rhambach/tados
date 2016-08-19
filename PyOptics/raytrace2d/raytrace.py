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
    
  def trace(self,nRays=7):
    " trace number of rays"
    # initial rays emerging from source
    n_before = self.source.get_refractive_index();
    rays=self.source.get_rays(nRays);
    self.vignetted_at_surf = np.full(nRays,len(self.system),dtype=np.int);
    self.raypath=[rays];

    # trace rays behind each surface in the optical system
    for num,surface in enumerate(self.system):
      rays,vig=surface.trace_behind(rays,n_before);
      self.raypath.append(rays);
      n_before = surface.get_refractive_index(n_before);  
      if np.any(vig):
        self.vignetted_at_surf[vig] = np.minimum(self.vignetted_at_surf[vig], num);
  
      
  def plot_raypath(self,show_vignetted=True,show_surfaces=True,**kwargs):
    """
    Plot system layout and results of last raytrace
    
    Parameters
    ----------
      show_vignetted : bool, optional
        shows raypath of vignetted rays, default: True
      show_surfaces : bool, optional
        shows surfaces of system, default: True
      **kwargs : keyword arguments
        further arguments are passed on to the matplotlib plot() function
    """   
    if not self.raypath: 
      raise RuntimeError("raypath is empty. First run a raytrace before plotting.");
    
    from matplotlib.lines import Line2D      
    # setup plot
    fig,ax = plt.subplots(1,1);
    ax.set_title("System Layout");
    ax.set_xlabel("position z along optical axis");
    ax.set_ylabel("ray height y");
    # extract list of z and y values from raypath
    pos = np.array([(rays.z,rays.y) for rays in self.raypath]);  # shape (nPoints,2,nrays)   
    # iterate over all rays
    for i in xrange(pos.shape[2]):
      s = self.vignetted_at_surf[i];  # last surface of unvignetted ray
                                      # corresponds to index s+1 in raypath (source is 0)
      if s<=len(self.system):
        ax.add_line(Line2D(pos[:s+1,0,i],pos[:s+1,1,i],ls='-',**kwargs));
        if show_vignetted: 
          ax.add_line(Line2D(pos[s:,0,i],pos[s:,1,i],ls=':',**kwargs));
      else:
        ax.add_line(Line2D(pos[:,0,i],pos[:,1,i],**kwargs));
    
    # plot surfaces
    if show_surfaces:    
      for surface in self.system:
        y,z = surface.get_surface_data();
        if y is not None:
          ax.plot(z,y,'k-');
      ax.set_aspect('equal')  
    return ax;
    
  def plot_footprint(self,surf=-1,**kwargs):
    """
    Plot intersection points of all rays with specified surface.
    
    Parameters
    ----------
      surf : integer, optional
        surface number in system, default: -1 corresponding to last surface 
      **kwargs : keyword arguments
        further arguments are passed on to the matplotlib hist() function
    """
    if not self.raypath: 
      raise RuntimeError("raypath is empty. First run a raytrace before plotting."); 

    if surf<0: surf+=len(self.system);    
    # setup plot
    fig,(ax1,ax2) = plt.subplots(1,2,sharey=True);
    fig.suptitle("Surf%2d: %s"%(surf,self.system[surf].info()));
    ax1.set_xlabel("position z along optical axis");    
    ax2.set_ylabel("ray height y at surface");    
    ax2.set_xlabel("counts");
    # extract list of z and y values from raypath    
    rays=self.raypath[surf+1];
    ax1.plot(rays.z,rays.y,'.',alpha=0.1);
    ax2.hist(rays.y,orientation="horizontal",**kwargs);
    return (ax1,ax2);    

    
  def print_system(self,verbosity=0):
    print("Source   %s"%self.source.info(verbosity))
    for i,surface in enumerate(self.system):
      print("Surf%2d   %s"%(i,surface.info(verbosity)));
    
 