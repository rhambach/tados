# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 10:37:13 2016

@author: Hambach
"""

from __future__ import division
import logging
import numpy as np
import matplotlib.pylab as plt
from point_in_triangle import point_in_triangle
from adaptive_mesh import *
from zemax_dde_link import *

def transmission_field_sampling():
  pass


def analyze_transmission(hDDE):  
  # set up ray-trace parameters and image detector
  image_surface = 22;
  wavenum  = 4;
  image_size = np.asarray((0.2,0.05));
  image_size = np.asarray((0.2,0.05));
  image_shape = np.asarray((201,401));
  img_pixels = cartesian_sampling(*image_shape,rmax=2); # shape: (2,nPixels)
  img_pixels*= image_size[:,np.newaxis]/2;
  image_intensity = np.zeros(np.prod(image_shape)); # 1d array

  # field sampling (octagonal fiber)
  xx,yy=cartesian_sampling(7,7,rmax=.02);  # low: 11x11, high: 7x7
  ind = (np.abs(xx)<=1) & (np.abs(yy)<=1) & \
              (np.abs(xx+yy)<=np.sqrt(2)) & (np.abs(xx-yy)<=np.sqrt(2));
  xx=xx[ind]; yy=yy[ind];
  plt.figure(); plt.title("field sampling (normalized coordinates)");
  plt.plot(xx.flat,yy.flat,'.')
  plt.xlabel('x'); plt.ylabel('y');
  
  
  for i in xrange(len(xx)):
    x=xx[i]; y=yy[i];
    x=0.3333333333333; y=0.666666666666;
    print("Field point: x=%5.3f, y=%5.3f"%(x,y))
    
    # init adaptive mesh for pupil sampling☺
    px,py=fibonacci_sampling_with_circular_boundary(50,20) # low: (50,20), high: (200,50)
    initial_sampling = np.vstack((px,py)).T;         # size (nPoints,2)
    def raytrace(pupil_points):        # local function for raytrace
      px,py = pupil_points.T;
      ret = hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
      vigcode = ret[:,[6]]<>0;        # include vignetting by shifting ray outside image
      return ret[:,[0,1]]+image_size*vigcode;
    Mesh=AdaptiveMesh(initial_sampling, raytrace);

    # segmentation of triangles along cutting line
    lthresh = 0.5*image_size[1];
    Athresh = np.pi/1000;       # 0.1% of pupil area
    def is_broken(simplices):
      broken = Mesh.get_broken_triangles(simplices=simplices,lthresh=lthresh);
      area   = Mesh.get_area_in_domain(simplices=simplices);
      return broken & (area>Athresh)  
    while True:  # iteratively refine broken triangles
      nNew = Mesh.refine_broken_triangles(is_broken,nDivide=100,bPlot=True);
      if nNew==0: break # no additional triangles added
      if i==0: Mesh.plot_triangulation(skip_triangle=is_broken);
    pupil_points, image_points, simplices = Mesh.get_mesh();

    # analysis of beam intensity in each triangle (conservation of energy!) 
    broken = Mesh.get_broken_triangles(lthresh=lthresh)  
    pupil_area = Mesh.get_area_in_domain(); 
    pupil_area_tot = np.sum(pupil_area);
    assert(all(pupil_area>0));  # triangles should be oriented ccw in pupil
    err_circ = 1-pupil_area_tot/np.pi;    
    err_broken = np.sum(pupil_area[broken])/pupil_area_tot;
    logging.info('error of triangulation: \n' +
     '  %5.3f%% due to approx. of circular pupil boundary \n'%(err_circ*100) +
     '  %5.3f%% due to broken triangles' %(err_broken*100));
    image_area = Mesh.get_area_in_image();
    if any(image_area<0) and any(image_area>0):
      logging.warning('scambling of rays, triangulation may not be working')
    
    # footprint in image plane (assuming beam of 1W)
    density = abs(pupil_area / pupil_area_tot / image_area );  # [W/mm^2]
    for s in np.where(~broken)[0]:
      triangle = image_points[simplices[s]];
      mask = point_in_triangle(img_pixels,triangle);
      image_intensity += density[s]*mask;
    
  # plotting of footprint in image plane
  img_pixels_2d = img_pixels.reshape(2,image_shape[1],image_shape[0]);
  image_intensity = image_intensity.reshape((image_shape[1],image_shape[0]))   
  xaxis = img_pixels_2d[1,:,0]; dx=xaxis[1]-xaxis[0];
  yaxis = img_pixels_2d[0,0,:]; dy=yaxis[1]-yaxis[0];
  
  fig,(ax1,ax2)= plt.subplots(2);
  ax1.set_title("footprint in image plane (surface: %d)"%image_surface);
  ax1.imshow(image_intensity,origin='lower',aspect='auto',interpolation='hanning',
             extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]]);
  ax2.set_title("integrated intensity in image plane");    
  ax2.plot(xaxis,np.sum(image_intensity,axis=1)*dy,label="along y");
  ax2.plot(yaxis,np.sum(image_intensity,axis=0)*dx,label="along x");
  ax2.legend(loc=0)
  
  logging.info('total intensity: %5.3f W'%(np.sum(image_intensity)*dx*dy));
  
  return 1



if __name__ == '__main__':
  import os as os
  import sys as sys
  logging.basicConfig(level=logging.INFO);
  
  with DDElinkHandler() as hDDE:
  
    ln = hDDE.link;
    # load example file
    #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
    #                        'Cooke 40 degree field.zmx')
    filename= os.path.realpath('./tests/pupil_slicer.ZMX');
    hDDE.load(filename);
    analyze_transmission(hDDE);
    