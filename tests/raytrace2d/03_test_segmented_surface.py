# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:03:53 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt

import _set_pkgdir
import PyOptics.raytrace2d as rt

def get_reference_system(Nreflections):
  " build zigsag system from single Segments"
  y =np.arange(Nreflections/2+1,dtype=np.float);
  z =y%2;
  y-=0.5;
  system=[];
  for ir in xrange(Nreflections-1):
    i = int((ir+1)/2); # segment number at reflection: sequence 0,1,1,2,2,3,3,...
    offset = 1 if ir%4==0 or ir%4==1 else 0;  # offset sequence  1,1,0,0,1,1,0,...
    reverse= 1 if ir%4==1 or ir%4==2 else 0;  # orientation of surface, seq:  0,1,1,0,0,1,1,...
    n = -1 if ir%2==0 else 1;                 # index of refaction after surface, seq: -1,1,-1,...    
    A = (y[i],z[i]+offset); B = (y[i+1],z[i+1]+offset);
    if reverse: A,B=B,A;
    system.append( rt.PlaneSurface(A,B,n=n));
  return system;
  
def get_test_system(Nreflections):
  " build zigsag system as repeated segmented surfaces"
  # single zig-zag line  
  y =np.arange(Nreflections/2+1,dtype=np.float);
  z =y%2;
  y-=0.5;
  # double zig-sag in one loop
  y = np.hstack([y,y[::-1]]);
  z = np.hstack([z+1,z[::-1]]);
  
  # s0: ccw orientation, transition into negativ index medium to simulate mirror
  # s1: cw orientation (as light travels backwards in negative index medium)s=[0,0];
  s=[0,0];
  s[0] = rt.SegmentedSurface(y,      z      ,n=-1,allow_virtual=False);
  s[1] = rt.SegmentedSurface(y[::-1],z[::-1],n= 1,allow_virtual=False);
  
  return [s[i%2] for i in xrange(Nreflections)];

def test_zigsag_mirror_system(Nreflections,nRays=1):

  # setup plot
  fig,ax = plt.subplots(1,2,sharex=True,sharey=True);
  fig.suptitle("Test Reflection in Segmented Surfaces (%d reflections)"%(Nreflections));
  
  # extract list of z and y values from raypath

  print("Setting up system ...")
  # source
  source = rt.CollimatedBeam(0.1,1,0,n=1);
  # optical system (sequential list of surfaces, position in global coordinates)
  systems = [get_reference_system(Nreflections), 
             get_test_system(Nreflections)       ];
  
  for n,system in enumerate(systems):
    # raytrace
    print("Perform Raytrace #%d ..."%n)
    tracer = rt.Raytracer(source,system);
    tracer.trace(nRays=nRays);
    rt.SimpleLayout(tracer,ax=ax[n]).plot();
    
  ax[0].set_title("Reference: SingleSurface");
  ax[1].set_title("Test: SegmentedSurface")
  return fig;
  
if __name__ == '__main__':
  
  test_zigsag_mirror_system(10,nRays=3);
  plt.show();
  