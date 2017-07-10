# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 10:37:13 2016

@author: Hambach
"""


import logging
import numpy as np
import matplotlib.pylab as plt

import _set_pkgdir
from PyOptics.illumination.point_in_triangle import point_in_triangle
from PyOptics.illumination.adaptive_mesh import AdaptiveMesh
from PyOptics.zemax import sampling, dde_link

def analyze_transmission(hDDE):  
  # set up ray-trace parameters and image detector
  image_surface = 21;
  wavenum  = 1;
  image_size = np.asarray((0.2,0.05));
  image_size = np.asarray((0.2,0.05));
  image_shape = np.asarray((201,501));
  img_pixels = sampling.cartesian_sampling(*image_shape,rmax=2); # shape: (2,nPixels)
  img_pixels*= image_size[:,np.newaxis]/2;
  image_intensity = np.zeros(np.prod(image_shape)); # 1d array

  # field sampling
  xx,yy=sampling.cartesian_sampling(3,3,rmax=.1)  
  for i in range(len(xx)):
    x=xx[i]; y=yy[i];
    print("Field point: x=%5.3f, y=%5.3f"%(x,y))
    
    # init adaptive mesh for pupil sampling☺
    px,py=sampling.fibonacci_sampling_with_circular_boundary(500)  
    initial_sampling = np.vstack((px,py)).T;         # size (nPoints,2)
    def raytrace(pupil_points):        # local function for raytrace
      px,py = pupil_points.T;
      ret = hDDE.trace_rays(x,y,px,py,wavenum,surf=image_surface);
      vigcode = ret[:,[1]]!=0;        # include vignetting by shifting ray outside image
      return ret[:,[2,3]]+image_size*vigcode;
    Mesh=AdaptiveMesh(initial_sampling, raytrace);

    # mesh refinement  
    if False:  
      # iterative refinement of broken triangles
      lthresh = 0.5*image_size[1];
      is_large= lambda simplices: Mesh.find_broken_triangles(simplices=simplices,lthresh=lthresh);    
      for it in range(4): 
        Mesh.refine_large_triangles(is_large);
        if i==0: Mesh.plot_triangulation(skip_triangle=is_large);
    else:
      # segmentation of triangles along cutting line
      lthresh = 0.5*image_size[1];
      is_broken = lambda simplices: Mesh.find_broken_triangles(simplices=simplices,lthresh=lthresh);  
      Mesh.refine_broken_triangles(is_broken,nDivide=7,bPlot=True);
      if i==0: Mesh.plot_triangulation(skip_triangle=is_broken);      
    pupil_points, image_points, simplices = Mesh.get_mesh();

    # analysis of beam intensity in each triangle (conservation of energy!) 
    broken = Mesh.find_broken_triangles(lthresh=lthresh)  
    pupil_area = Mesh.get_area_in_domain(); 
    assert(all(pupil_area>0));  # triangles should be oriented ccw in pupil
    err_circ = 1-np.sum(pupil_area)/np.pi;    
    err_broken = np.sum(pupil_area[broken])/np.sum(pupil_area);
    logging.info('error of triangulation: \n' +
     '  %5.3f%% due to approx. of circular pupil boundary \n'%(err_circ*100) +
     '  %5.3f%% due to broken triangles' %(err_broken*100));
    image_area = Mesh.get_area_in_image();
    if any(image_area<0) and any(image_area>0):
      logging.warning('scambling of rays, triangulation may not be working')
    
    # footprint in image plane
    density = abs(pupil_area / image_area);
    for s in np.where(~broken)[0]:
      triangle = image_points[simplices[s]];
      mask = point_in_triangle(img_pixels,triangle);
      image_intensity += density[s]*mask;
    
    if i==0:
      plt.figure();
      plt.title("Intensity in each triangle of the Pupil Triangulation");
      plt.plot(pupil_area,'b',label="$A_{pupil}$")
      plt.plot(image_area,'g',label="$A_{image}$") 
      plt.legend(loc='best');
 
  # plotting of footprint in image plane
  img_pixels_2d = img_pixels.reshape(2,image_shape[1],image_shape[0]);
  image_intensity = image_intensity.reshape((image_shape[1],image_shape[0]))   
  xaxis = img_pixels_2d[1,:,0];
  yaxis = img_pixels_2d[0,0,:];
  
  fig,(ax1,ax2)= plt.subplots(2);
  ax1.set_title("footprint in image plane (surface: %d)"%image_surface);
  ax1.imshow(image_intensity,origin='lower',aspect='auto',interpolation='hanning',
             extent=[xaxis[0],xaxis[-1],yaxis[0],yaxis[-1]]);
  ax2.set_title("integrated intensity in image plane");    
  ax2.plot(xaxis,np.sum(image_intensity,axis=1),label="along y");
  ax2.plot(yaxis,np.sum(image_intensity,axis=0),label="along x");
  ax2.legend(loc=0)
  
  return 1



if __name__ == '__main__':
  import os as os
  logging.basicConfig(level=logging.INFO);
  
  with dde_link.DDElinkHandler() as hDDE:
  
    ln = hDDE.link;
    # load example file
    #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
    #                        'Cooke 40 degree field.zmx')
    filename= os.path.realpath('../tests/zemax/pupil_slicer.ZMX');
    hDDE.load(filename);
    analyze_transmission(hDDE);
    