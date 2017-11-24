# -*- coding: utf-8 -*-

import abc, six
import numpy as np
from tados.raytrace2d import raytrace

# ToDo: add PointSource

@six.add_metaclass(abc.ABCMeta)    # backward compatible to 2.7
class Source(object):
  " abstract base class for all sources, defines interface only "
  
  @abc.abstractmethod
  def info(self, verbosity=0):
    """ 
    return info string describing the source in a lens data editor,
    verbosity ... 0: one-liner, 1: +parameters, 2: debug
    """
    return;
    
  @abc.abstractmethod    
  def get_rays(self,nRays): return; 

  @abc.abstractmethod 
  def get_refractive_index(self): return;



class CollimatedBeam(Source):
  
  def __init__(self,diameter,z,angle,n=1.):
    """
    Parameters
    ----------
      diameter : float
        diameter of the beam
      z : float
        source position along axis
      angle : float
        ray inclination angle of the beam [degree]
        measured against positive z-direction
      n : float
        refractive index of the medium around the source
    """
    self.diameter=diameter;
    self.z=z;
    self.angle=angle;
    self.n=n;
   
  def info(self,verbosity=0):
    descr = "Collimated Beam";
    if verbosity>0: descr += " (diameter: %f, z: %f, angle: %f, n: %f)" \
                          %(self.diameter,self.z,self.angle,self.n);
    return descr;
   
  def get_rays(self,nRays):
    """
    Parameters
    ----------
      nRays : float
        number of rays
        
    Returns
    ------
      rays : instance of class Rays()
        list of collimated rays, beam center on axis at z, ray angle u
    """
    y = self.diameter * np.linspace(-0.5,0.5,nRays); # ray heights at z
    vz= np.cos(np.deg2rad(self.angle));
    vy= np.sin(np.deg2rad(self.angle));
    return raytrace.Rays(z=self.z,y=y,vz=vz,vy=vy);
    
  def get_refractive_index(self):
    return self.n;
    
    
class PointSource(Source):
  
  def __init__(self,pos,amin=0,amax=360, n=1.):
    """
    Parameters
    ----------
      pos : (z,y)-tuple of floats
        coordinates of starting postion (z,y)
      amin,amax : floats, optional
       ray inclination angle of the beam [degree]
       measured against positive z-direction is restricted to (amin,amax) 
      n : float
        refractive index of the medium around the source
    """
    self.z=pos[0];
    self.y=pos[1];    
    self.amin=amin;
    self.amax=amax;
    self.n=n;
   
  def info(self,verbosity=0):
    descr = "Point Sourcem";
    if verbosity>0: descr += " (position: z=%f, y=%f, angles: %f < alpha < %f, n: %f)" \
                          %(self.z,self.y,self.amin,self.amax,self.n);
    return descr;
   
  def get_rays(self,nRays):
    """
    Parameters
    ----------
      nRays : float
        number of rays
        
    Returns
    ------
      rays : instance of class Rays()
        list of collimated rays, beam center on axis at z, ray angle u
    """
    bFullAngle = np.allclose(self.amax-self.amin,360);
    angles = np.linspace(self.amin,self.amax,nRays,endpoint=not bFullAngle);
    vz= np.cos(np.deg2rad(angles));
    vy= np.sin(np.deg2rad(angles));
    return raytrace.Rays(z=self.z,y=self.y,vz=vz,vy=vy);
    
  def get_refractive_index(self):
    return self.n;    
    
    

class SingleRay(Source):
   
  def __init__(self,pos,angle,n=1.):
    """
    Parameters
    ----------
      pos : (z,y)-tuple of floats
        coordinates of starting postion (z,y)
      angle : float
       ray inclination angle of the beam [degree]
       measured against positive z-direction
      n : float
       refractive index of the medium around the source  
    """
    self.n = n;
    self.angle = angle;
    vz= np.cos(np.deg2rad(self.angle));
    vy= np.sin(np.deg2rad(self.angle));
    self.ray = raytrace.Rays(z=pos[0],y=pos[1],vz=vz,vy=vy);
    
  def info(self,verbosit=0):
    descr = "SingleRay";
    return descr
    
  def get_rays(self,nRays):
    return self.ray;
    
  def get_refractive_index(self):
    return self.n;