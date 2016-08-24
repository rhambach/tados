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
  def plot_rays(self,show_vignetted=True,ind=None):
    """
    Plots raypath only.
        
    Parameters
    ----------
      show_vignetted : bool, optional
        shows raypath of vignetted rays, default: True
      ind : slice object, optional
        indicates indices of ray segments to plot, default: all
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
      ind : slice object, optional
        indicates indices of surfaces to plot, default: all
      hind : slice object, optional
        indicates indices of surfaces to highlight, default: none
      **kwargs : keyword arguments
        further arguments are passed on to the matplotlib plot() function for the ray plot
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
    self.ax.set_title("System Layout (%d rays)"%self.nRays);
    
  def plot_rays(self,show_vignetted=True,alpha=None,ind=None,**kwargs):
    
    # estimate opacity between 1 and 0.01        
    if alpha is None: alpha = 256./self.nRays/self.nPoints;    
    alpha = min(max(alpha,1./256),1);

    # plot many ray semgemnts efficiently with matplotlib's LineCollection    
    # see http://exnumerus.blogspot.de/2011/02/how-to-quickly-plot-multiple-line.html      
    lines = np.stack([self.points[:,:-1,:],self.points[:,1:,:]],axis=2); # shape (nRays,nPoints-1,2,2)

    # index vignetted ray segments
    surf_num = np.arange(self.nPoints-1);
    vig = surf_num[np.newaxis,:] >= self.vignetted_at_surf[:,np.newaxis]; # shape (nRays,nPoints-1);
     
    # plot unvignetted rays
    linecol = LineCollection(lines[~vig],alpha=alpha,color='blue',**kwargs);
    self.ax.add_collection(linecol)

    # plot vignetted rays    
    if show_vignetted:
      linecol = LineCollection(lines[vig],alpha=alpha,color='red',**kwargs);
      self.ax.add_collection(linecol)
      
    # update
    self.ax.figure.canvas.draw();

    
class Footprint(SimpleLayout):

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
    super(Footprint,self).plot_system(ind=slice(0,self.surf+1),hind=slice(self.surf,self.surf+1),**kwargs);
      
  def plot_rays(self,show_vignetted=True,**kwargs):
    # plot rays in SystemLayout
    self.ax=self.ax1;
    super(Footprint,self).plot_rays(show_vignetted=show_vignetted,ind=slice(0,self.surf+1));
    # plot Footprint
    y=self.points[:,self.surf+1,1];                # shape (nRays,)
    vig = self.vignetted_at_surf <= self.surf;
    self.ax2.hist(y[~vig],orientation="horizontal",**kwargs);
   
   
class PlotPropagation(SimpleLayout):

  def __init__(self,tracer,ax=None):
    """
    Plot propagation of light according to the results of last raytrace up to fixed OPL.
    
    Parameters
    ----------
      tracer : instance of Raytrace
        optical system and results of last raytrace (Raytrace.trace() must be executed once)
      ax     : instance of matplotlib.Axes, optional
        axes for plotting, default: a new figure is created
    """
    super(PlotPropagation,self).__init__(tracer,ax=ax);
    self.points_orig = self.points.copy();            # shape (nRays,nPoints,2)
    # calculate OPL
    self.opl_segment = np.linalg.norm( self.points[:,:-1,:]-self.points[:,1:,:], axis=2); 
    self.opl_segment*= np.abs(tracer.n);              # shape (nRays,nPoints-1)
    self.opl = np.cumsum(self.opl_segment,axis=1);    # shape (nRays,nPoints-1)
      
  def get_max_opl(self):
    vig = self.vignetted_at_surf < len(self.system);
    return np.min(self.opl[~vig,-1]);

  def __restrict_raypath_to_opl(self,opl):
    " change self.points such that all rays stop at same maximum OPL"
    self.points=self.points_orig.copy();  # reset to initial state
    if opl > self.get_max_opl():
      raise RuntimeError('opl should be larger than %f.'%self.get_max_opl());
    # search first ray segment at which opl is exceeded (between points[ind] and points[ind+1])
    ind = np.argmax(opl<=self.opl,axis=1);   # 0 <= ind < nPoints-1
    ind_rays = np.arange(self.nRays);
    exceed_opl = self.opl[ind_rays,ind]-opl 
    assert np.all(exceed_opl>=0), "exceed_opl should be larger than zero"
    assert np.all(exceed_opl<self.opl_segment[ind_rays,ind]), "exceed_opl should be smaller than opl of next segment"
    # correct end-point of first ray-segment beyond OPL to match exactly the required opl
    end = self.points[ind_rays,ind+1];
    start = self.points[ind_rays,ind];
    t = exceed_opl / self.opl_segment[ind_rays,ind];
    t = t[:,np.newaxis];
    self.points[ind_rays,ind+1] = t*start + (1-t)*end;
    # remove points beyond opl
    self.vignetted_at_surf = ind+1;    
        
  def plot_rays(self,opl,show_vignetted=False,**kwargs):
    self.__restrict_raypath_to_opl(opl);
    super(PlotPropagation,self).plot_rays(show_vignetted=False);
    

    