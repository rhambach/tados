# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 19:45:55 2016

@author: Hambach
"""
import numpy as np
import matplotlib.pylab as plt


def cartesian_sampling(nx,ny,rmax=1.):
  """
  cartesian sampling in reduced coordinates (between -1 and 1)
  
  Parameters 
  ----------  
   nx,ny :  integer
     number of points along x and y
   rmax :  float, optional
     radius of circular aperture, default 1
  
  Returns
  -------
   x,y :  1d-arrays of size N
     x and y coordinates for each point
  """
  x = np.linspace(-1,1,nx);
  y = np.linspace(-1,1,ny);
  x,y=np.meshgrid(x,y);   
  ind = x**2 + y**2 <= rmax**2;
  return x[ind],y[ind]

def hexapolar_sampling(Nr,rmax=1.,ind=False):
  """
  hexapolar sampling with roughly equi-area sampling point distribution
  
  Parameters
  ----------
   Nr : integer
     number of rings, last ring has index (Nr-1)
   rmax: float, optional
     radius of circular aperture, default 1
   ind : bool, optional
     if true, number of points and weights of each ring are returned
     
  Returns
  ------
   x,y :  1d-arrays of size N
     x and y coordinates for each point 
   Ntet : 1d-array of size Nr, optional
     number of points in each ring
   weight : 1d-array of size Nr, optional
     weight of each ring     
  """
  r = np.arange(1,Nr,dtype=np.double)/Nr*rmax;               # ring centers
  Ntet = 6*np.arange(1,Nr);                  # number of points on each ring
  # construct grid points in each ring  
  x=[0]; y=[0];                              # first ring
  for i in xrange(Nr-1):
    tet = np.linspace(0,2*np.pi,Ntet[i],endpoint=False);
    x.extend(r[i]*np.cos(tet));
    y.extend(r[i]*np.sin(tet));
  x=np.asarray(x).flatten();
  y=np.asarray(y).flatten();
  # calculate number of points and weight of each ring, if desired
  if ind==False: 
    return x,y;
  elif Nr==1:
    return x,y,[1],[1];
  else:
    # calculate area of each ring Nr>1
    dr = r[1]-r[0];    
    area = ( (r+dr/2.)**2 - (r-dr/2.)**2 );
    # add first ring    
    area = np.insert(area,0, (dr/2.)**2);
    Ntet = np.insert(Ntet,0, 1);
    return x,y,Ntet,area/rmax**2;
    
      
def fibonacci_sampling(N,rmax=1.):
  """
  Fibonacci sampling in reduced coordinates (normalized to 1)

  Parameters
  ----------
   N : integer
     number of points
   rmax: float, optional
     radius of circular aperture, default 1
     
  Returns
  ------
   x,y :  1d-arrays of size N
     x and y coordinates for each point 
  """
  k = np.arange(N)+0.5;
  theta = 4*np.pi*k/(1+np.sqrt(5));
  r = rmax*np.sqrt(k/N)
  x = r * np.cos(theta);
  y = r * np.sin(theta);
  return x,y
  
def fibonacci_sampling_with_circular_boundary(N,Nboundary=None,rmax=1.):
  """
  Fibonacci sampling with additional points on the circular boundary

  Parameters
  ----------
   N : integer
     number of points
   Nboundary: integer, optional
     number of points on boundary of circular aperture
   rmax: float, optional
     radius of circular aperture, default 1
     
  Returns
  ------
   x,y :  1d-arrays of size N
     x and y coordinates for each point 
  """  
  
  # average distance d between points : pi*(d/2)^2 = pi/N (= area per point)
  d = 2/np.sqrt(N);  
  # estimate number of boundary points: 2*pi/d
  if Nboundary is None: Nboundary = int(2*np.pi/d);
  assert(N>Nboundary);  
  
  # fibonacci sampling up to 1-d/3 (empirical value)
  x,y = fibonacci_sampling(N-Nboundary,rmax=(1-d/3)*rmax);
  theta = 2*np.pi*np.arange(Nboundary)/Nboundary;
  xp = rmax * np.cos(theta);
  yp = rmax * np.sin(theta);
  return np.hstack((x, xp)), np.hstack((y, yp));
  

def __test_sampling((x,y),title=""):
  fig,ax=plt.subplots(1,1);
  t = np.linspace(0,2*np.pi,100,endpoint=True);  
  ax.plot(np.sin(t),np.cos(t),'r',lw=2)
  ax.scatter(x,y);  
  ax.set_title(title);
  ax.set_aspect('equal');
  

if __name__ == '__main__':
  __test_sampling( cartesian_sampling(21,21), 'Cartesian');
  __test_sampling( hexapolar_sampling(11),    'Hexapolar');
  __test_sampling( fibonacci_sampling(500),   'Fibonacci');
  __test_sampling( fibonacci_sampling_with_circular_boundary(500), 'Fibonacci+boundary');
 