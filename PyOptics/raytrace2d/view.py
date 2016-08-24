# -*- coding: utf-8 -*-

import abc
import numpy as np
import matplotlib.pylab as plt

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

class View(object):
  __metaclass__ = abc.ABCMeta
  
  def __init__(self,tracer,ax=None):
    if not tracer.raypath: 
      raise RuntimeError("raypath is empty. First run a raytrace before plotting.");
    self.source=tracer.source;
    self.system=tracer.system;
    self.vignetted_at_surf=tracer.vignetted_at_surf;               # shape (nRays,)
    # extract list of z and y values from raypath
    pos = np.array([(rays.z,rays.y) for rays in tracer.raypath]);  # shape (nPoints,2,nrays)   
    self.points = np.rollaxis(pos,2);                              # shape (nRays,nPoints,2)
    self.nRays,self.nPoints,_, = self.points.shape;
    
    if ax is None: fig,ax = plt.subplots(1,1);
    ax.set_xlabel("position z along optical axis");
    ax.set_ylabel("ray height y");        
    self.ax=ax;
    
  def plot_system(self, ind=None, hind=None, **kwargs):    
    """
    Plot system layout for specified surfaces
        
    Parameters
    ----------
      ind : slice object, optional
        indicates indices of surfaces to plot, default: all
      hind : slice object, optional
        indicates indices of surfaces to highlight, default: none
      **kwargs : keyword arguments
        further arguments are passed on to the matplotlib plot() function 
    """
    surf_normal    = self.system[ind] if ind is not None else self.system;
    surf_highlight = self.system[hind] if hind is not None else [];
    for (color,surfaces) in [('black',surf_normal), ('darkorange',surf_highlight)]:
      for surface in surfaces:
        y,z = surface.get_surface_data(); 
        if y is not None:
          self.ax.plot(z,y,color=color,**kwargs);
    
    self.ax.set_aspect('equal')  
    return self.ax;    

  @abc.abstractmethod
  def plot_rays(self,show_vignetted=True,**kwargs):
    """
    Plots raypath only.
        
    Parameters
    ----------
      show_vignetted : bool, optional
        shows raypath of vignetted rays, default: True
      **kwargs : keyword arguments
        further arguments are passed on to the matplotlib plot() function
    """   
    return;
   
  def plot(self, ind=None, hind=None, show_vignetted=True, **kwargs):
    """
    Plot system layout and results of last raytrace.
        
    Parameters
    ----------
      show_vignetted : bool, optional
        shows raypath of vignetted rays, default: True
      **kwargs : keyword arguments
        further arguments are passed on to the matplotlib plot() function for the ray-plot
    """   
    self.plot_rays(show_vignetted=show_vignetted,**kwargs);
    self.plot_system(ind=ind,hind=hind);
    return self.ax;    
    
    
class SimpleLayout(View):

  def __init__(self,tracer,ax=None):
    """
    Plot system layout and results of last raytrace.
    
    Parameters
    ----------
      tracer : instance of Raytrace
        optical system and results of last raytrace (Raytrace.trace() must be executed once)
      ax     : instance of matplotlib.Axes, optional
        axes for plotting, default: a new figure is created
    """
    super(SimpleLayout,self).__init__(tracer,ax=ax);
    self.ax.set_title("System Layout");
    
  def plot_rays(self,show_vignetted=True,**kwargs):
    pos=self.points;
    # iterate over all rays
    for i in xrange(self.nRays):
      s = self.vignetted_at_surf[i];  # last surface of unvignetted ray
                                      # corresponds to index s+1 in raypath (source is 0)
      if s<=len(self.system):
        self.ax.add_line(Line2D(pos[i,:s+1,0],pos[i,:s+1,1],ls='-',**kwargs));
        if show_vignetted: 
          self.ax.add_line(Line2D(pos[i,s:,0],pos[i,s:,1],ls=':',**kwargs));
      else:
        self.ax.add_line(Line2D(pos[i,:,0,i],pos[i,:,1],**kwargs));
    
  

class Footprint(View):

  def __init__(self,tracer,surf=-1):
    """
    Plot intersection points of all rays with specified surface.
    
    Parameters
    ----------
      tracer : instance of Raytrace
        optical system and results of last raytrace (Raytrace.trace() must be executed once)
      surf : integer, optional
        surface number in system, default: -1 corresponding to last surface 
    """
    fig,(ax1,ax2) = plt.subplots(1,2);
    super(Footprint,self).__init__(tracer,ax=ax1);
    ax2.set_xlabel("counts");
    ax2.set_ylabel("ray height y at surface");  
    ax2.yaxis.set_label_position("right")       
    ax2.yaxis.tick_right()
    self.ax1=ax1; self.ax2=ax2; 
    # set surface to plot    
    if surf<0: surf+=len(self.system);    
    self.surf=surf;
    fig.suptitle("Surf%2d: %s"%(surf,self.system[surf].info()));
 
  def plot_system(self,ind=None,hind=None,**kwargs):
    self.ax=self.ax1;
    self.ax.set_title("System Layout");
    super(Footprint,self).plot_system(ind=slice(0,self.surf+1),hind=slice(self.surf,self.surf+1),**kwargs);
      
  def plot_rays(self,show_vignetted=True,**kwargs):
    # extract list of z and y values from raypath    
    x,y=self.points[:,:self.surf+2].T;             # shape (nSurfaces,nRays)
    vig = self.vignetted_at_surf <= self.surf;
    alpha = max(0.01, min(1.,256./self.nRays));    # estimate opacity between 1 and 0.01
    self.ax1.plot(x[:,~vig],y[:,~vig],'b-',alpha=alpha);
    if show_vignetted: self.ax1.plot(x[0,vig],y[0,vig],'r.');
    self.ax2.hist(y[-1,~vig],orientation="horizontal",**kwargs);
   
    
    
    