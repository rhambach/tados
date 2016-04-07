# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 13:50:00 2016

@author: Hambach
"""
import numpy as np

def points_in_triangle(points,triangle):
  """
  determines, if point is in given triangle (can be both arrays)
    points   ... coordinates of all points, array of shape (2,nPoints)
    triangle ... coordinates of a triangle, array of shape (3,2)
  
  implementation follows http://www.blackpawn.com/texts/pointinpoly/
  """
  assert (points.shape[0] == 2)
  assert (triangle.ndim<3 and triangle.shape == (3,2))
  return PIT_crossproduct(points,triangle);
  
  
  
# Using cross product ----------------------------------------------------- 
def same_side(p1,p2, a,b):
  """
    check if p1 and p2 are on the same side of line through a and b
    p1,p2, a,b ... coordinates of points, shape (2,nPoints)
  """
  # is orientation of triangle (a,b,p1) and (a,b,p2) the same ?
  return np.cross(b-a, p1-a,axis=0) * np.cross(b-a, p2-a,axis=0) >0;

def PIT_crossproduct(p, triangle):
  """
    implementation of point-in-triangle by determining,
    if point is on right side of each edge of the triangle
  """
  a,b,c = triangle.reshape((3,-1)+(1,)*(p.ndim-1));
  return same_side(p,a, b,c) & same_side(p,b, a,c) & same_side(p,c, a,b);
   

# Helper functions -------------------------------------------------------    
def cyclic_path(triangles):
  assert (triangles.ndim < 4 and triangles.shape[1:3] == (3,2))
  return np.concatenate((triangles.reshape((-1,3,2)),triangles[:,0].reshape((-1,1,2))),axis=1);    

def test_few_points(points,triangles,fPIT):
  is_inside=np.zeros(points.shape[1],dtype=bool)
  for t in triangles:  # test each point separately
    is_inside |= fPIT(points,t);
  border=cyclic_path(triangles);
  plt.figure()
  plt.plot(border[:,:,0].T,border[:,:,1].T,'b-')
  plt.plot(points[0][~is_inside],points[1][~is_inside],'r.');
  plt.plot(points[0][is_inside], points[1][is_inside],'g.');

def test_image(img_points,triangles,fPIT):
  is_inside=np.zeros(img_points.shape[1:],dtype=bool)
  for t in triangles:  # test each point separately
    is_inside |= fPIT(img_points,t);
  border=cyclic_path(triangles);
  plt.figure()
  plt.imshow(is_inside,origin='lower',
      extent=[img_points[0].min(),img_points[0].max(),img_points[1].min(),img_points[1].max()]);  
  plt.plot(border[:,:,0].T,border[:,:,1].T,'g--',linewidth=2)
 
    
if __name__ == '__main__':
  import matplotlib.pylab as plt  
  triangles = np.asarray([[[-1.,   -.2], [-1.5, -2.1], [-0.5, -1.1]],
                          [[ 0.,   1.2], [ 1.,   0.1], [ 0.1,  0.2]],
                          [[ 0.1, -1.2], [-0.1,  0.2], [-1.1,  0.1]],
                          [[-0.5,    1], [  0,   0.3], [  -1, 0.2]],]);
  
  points=np.random.randn(2,100);
  image =np.asarray(np.meshgrid(np.linspace(-2,1,1000),np.linspace(-3,2,1000)));
  
  test_few_points(points,triangles,PIT_crossproduct);
  test_image(image,triangles,PIT_crossproduct);