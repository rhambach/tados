# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from _context import PyOptics
import PyOptics.raytrace2d as rt

def plot_polygon(Nverts,Nreflections,Nrays,intensity):

  print("Setting up system ...")
  # source
  source = rt.CollimatedBeam(0.1025,-0.2,0,n=1)
  
  # optical system (sequential list of surfaces, position in global coordinates)
  theta = 2*np.pi/Nverts * (np.arange(0,Nverts+1)+0.5);
  r = 0.125/2;  
  x =-r*np.cos(theta); x+=0.;
  y =-1.8*r*np.sin(theta); 
  
  # s0: ccw orientation, transition into negativ index medium to simulate mirror
  # s1: cw orientation (as light travels backwards in negative index medium)s=[0,0];
  s=[0,0];
  s[0] = rt.SegmentedSurface(x,y,n=1.4701,allow_virtual=False);
  s[1] = rt.SegmentedSurface(x[::-1],y[::-1],n=-1.4701,allow_virtual=False);
  system = [s[i%2] for i in range(Nreflections)];
    
  # raytrace
  print("Perform Raytrace ...")
  tracer = rt.Raytracer(source,system);
  tracer.trace(nRays=Nrays);
  
  # plotting
  # see http://exnumerus.blogspot.de/2011/02/how-to-quickly-plot-multiple-line.html      
  print("Plot Results ...")
  from matplotlib.collections import LineCollection
  # setup plot
  fig,ax = plt.subplots(1,1);
  ax.set_title("Billiard for Ellipse (%d rays, %d reflections)"%(Nrays,Nreflections));
  # extract list of z and y values from raypath
  pos = np.array([(rays.z,rays.y) for rays in tracer.raypath]);  # shape (nPoints,2,nrays)   
  pos = np.rollaxis(pos,2,start=1);                              # shape (nPoints,nRays,2)
  # iterate over all ray-segments and draw individual image with matplotlib
  nPoints,nRays,_ = pos.shape;
  for i in range(nPoints-1):
    lines = np.stack([pos[i,:-1,:],pos[i+1,1:,:]],axis=1); # shape (nRays,2,2)
    linecol = LineCollection(lines.reshape(-1,2,2),lw=1,alpha=intensity[i]);
    ax.add_collection(linecol)
  
  # plot surfaces
  y,x = system[0].get_surface_data();
  ax.plot(x,y,'k-');
  ax.set_aspect('equal')  
  plt.show();
  
  
if __name__ == '__main__':
  
  Nverts=10000;
  Nreflections=3;        # reflections inside polygon
  Nrays=500;               # number of rays

  # alpha should not be too small: https://github.com/matplotlib/matplotlib/issues/2287/
  alpha = max(0.01, min(1.,256./Nrays/Nreflections));    # estimate opacity between 1 and 0.01
  alpha = 0.05
  plot_polygon(Nverts,Nreflections,Nrays,np.asarray([20,20,1])/256.);