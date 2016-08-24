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
  print("Plot Results ...")
  PL=rt.PlotPropagation(tracer);  
  opl = PL.get_max_opl();
  alpha = max(0.01, min(1.,256./Nrays/Nreflections));    # estimate opacity between 1 and 0.01
  PL.plot_rays(opl,alpha=alpha);
  PL.plot_system(ind=slice(0,1));  
  PL.ax.set_title("Billiard for regular %d-Polygon (%d rays, %d reflections)"%(Nverts,Nrays,Nreflections));
   
  return PL
  
if __name__ == '__main__':
  
  for Nverts in (5,):#8,1000):  # number of vertices of polygon
    Nreflections=3;        # reflections inside polygon
    Nrays=1001;               # number of rays
    start=(0.5,0.137);       # coordinates of starting point
    print("###### %d Polygon ####################################"%Nverts)
    plot_polygon(Nverts,Nreflections,Nrays,start);
    
  plt.show();
  