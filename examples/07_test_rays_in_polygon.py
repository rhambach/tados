# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

import _set_pkgdir
import PyOptics.raytrace2d as rt

def plot_polygon(Nverts,Nreflections,Nrays,start=(0.5,0.137)):

  print("Setting up system ...")
  # source
  if Nrays==1:
    source = rt.SingleRay(start,10,n=1);
  else:
    source = rt.PointSource(start,n=1)
  
  # optical system (sequential list of surfaces, position in global coordinates)
  theta = 2*np.pi/Nverts * (np.arange(0,Nverts+1)+0.5);
  x =-np.cos(theta);
  y =-np.sin(theta);
  
  # s0: ccw orientation, transition into negativ index medium to simulate mirror
  # s1: cw orientation (as light travels backwards in negative index medium)s=[0,0];
  s=[0,0];
  s[0] = rt.SegmentedSurface(x[::-1],y[::-1],n=-1,allow_virtual=False);
  s[1] = rt.SegmentedSurface(x,y,n= 1,allow_virtual=False);
  system = [s[i%2] for i in xrange(Nreflections)];
    
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
  ax.set_title("Billiard for regular %d-Polygon (%d rays, %d reflections)"%(Nverts,Nrays,Nreflections));
  # extract list of z and y values from raypath
  pos = np.array([(rays.z,rays.y) for rays in tracer.raypath]);  # shape (nPoints,2,nrays)   
  # iterate over all rays
  nPoints,_,nRays = pos.shape;
  
  # alpha should not be too small: https://github.com/matplotlib/matplotlib/issues/2287/
  pos = np.rollaxis(pos,2); # shape (nRays,nPoints,2)
  lines = np.stack([pos[:,:-1,:],pos[:,1:,:]],axis=2); # shape (nRays,nPoints,2,2)
  alpha = max(0.01, min(1.,256./Nrays/Nreflections));    # estimate opacity between 1 and 0.01
  linecol = LineCollection(lines.reshape(-1,2,2),alpha=alpha);
  ax.add_collection(linecol)
  
  # plot surfaces
  y,x = system[0].get_surface_data();
  ax.plot(x,y,'k-');
  ax.set_aspect('equal')  
  plt.show();
  
if __name__ == '__main__':
  
  for Nverts in (5,8,1000):  # number of vertices of polygon
    Nreflections=100;        # reflections inside polygon
    Nrays=100;               # number of rays
    start=(0.5,0.137);       # coordinates of starting point
    print("###### %d Polygon ####################################"%Nverts)
    plot_polygon(Nverts,Nreflections,Nrays,start);