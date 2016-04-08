# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 17:47:19 2016

@author: Hambach
"""
import numpy as np

class ToleranceSystem(object):
  """
  Helper class for tolerancing the system
  """

  def __init__(self,hDDE,filename):
    self.hDDE=hDDE;
    self.ln = hDDE.link;
    self.filename=filename;
    
    hDDE.load(filename);
    self.ln.zPushLens(1);

    self.numSurf = self.ln.zGetNumSurf();
    self.__is_real = np.ones(self.numSurf,dtype=bool); # index array for real surfaces
    self.__R0, self.__t0 = self.__get_surface_coordinates();
    
    
  def __get_surface_coordinates(self):
    """
    returns for each surface in the system, the global rotation matrix 
           | R11  R12  R13 |                   | X |
      R =  | R21  R22  R23 | and shift   t =   | Y |
           | R31  R32  R33 |                   | Z |
    Returns: (R,t) of shape (numSurf,3,3) and (numSurf,3) respectively
    """    
    real_surfaces = np.where(self.__is_real)[0];  # list of real surfaces (no additional coordinate breaks)
    assert( len(real_surfaces)==self.numSurf);    # number of real surfaces should not change
    coords = [ self.ln.zGetGlobalMatrix(s) for s in real_surfaces ]
    coords = np.asarray(coords);
    R = coords[:,0:9].reshape(self.numSurf,3,3);
    t = coords[:,9:12];
    return R,t
    
  def __get_surface_comments(self):
    real_surfaces = np.where(self.__is_real)[0];
    return [ self.ln.zGetComment(s) for s in real_surfaces];
    
  def print_LDE(self,bShowDummySurfaces=False):
    print self.ln.ipzGetLDE();

  def print_current_geometric_changes(self):
    """
    print all surfaces that have been decentered or rotated
    """  
    R,t = self.__get_surface_coordinates();
    dR = R - self.__R0; 
    dt = t - self.__t0;
    numSurf = dR.shape[0];
    comment = self.__get_surface_comments();
    
    if np.all(dR==0) and np.all(dt==0):
      print " system unchanged."
      return
  
    print "   surface     shift           tilt "
    for s in xrange(numSurf):
      print "  %2d: "%s,
      # check for translation
      x,y,z=dt[s]
      if   x==0 and y==0 and z==0: print "          ",
      elif x==0 and y==0 and z<>0: print "DZ: %5.3f,"%z,
      elif x==0 and y<>0 and z==0: print "DY: %5.3f,"%y,
      elif x<>0 and y==0 and z==0: print "DX: %5.3f,"%x,
      else:                        print "(%3.2f,%3.2f,%5.2f)"%(x,y,z),
      
      # check for rotations
      x,y,z = np.linalg.norm(dR[s],axis=1);
      if   x==0 and y==0 and z==0: print "    ",
      elif x==0:                   print "XROT",
      elif x==0 and y<>0 and z==0: print "YROT",
      elif x<>0 and y==0 and z==0: print "ZROT",
      else:                        print " ROT"
      
      if comment[s]: print "  (%s)"%(comment[s]);
      else: print ""
    
    
  def tilt_decenter_elements(self,*args,**kwargs):
    """
    Wrapper for pyzDDE.zTiltDecenterElements(firstSurf, lastSurf, xdec=0.0, ydec=0.0, 
      xtilt=0.0, ytilt=0.0, ztilt=0.0, order=0, cbComment1=None, cbComment2=None, dummySemiDiaToZero=False)
    The function tilts/decenters optical elements between two surfaces
    by adding appropriate coordinate breaks and dummy surfaces.
    
    returns surface numbers of added surfaces (triple of ints)     
    """
    s1,s2,s3 = np.sort(self.ln.zTiltDecenterElements(*args,**kwargs));
    self.__is_real = np.insert(self.__is_real,(s1,s2-1,s3-2),False);
    assert(np.all(self.__is_real[[s1,s2,s3]]==False));
    ret = self.ln.zPushLens(1);
    if ret==0:  return s1,s2,s3;
    else: return ret;

  def change_thickness(self,surf=0,adjust_surf=0,value=0):
    """
    Wrapper for TTHI operand. Change thickness of given surface by a given value 
    in lens units. An adjustment surface can be specified, whose thickness is
    adjusted to compensate the z-shift. All surfaces after the adjustment surface
    will not be altered (imitates the behaviour fo TTHI(n,m).
      surf       ... surface number after which the thickness is changed
      adjust_surf... (opt) adjustment surface, ineffective if surf==adjust_surf
      value      ... thickness variation in lens units
          
    """
    self.ln.zSetSurfaceData(surfNum=surf, code=self.ln.SDAT_THICK, value=value);
    if adjust_surf>surf:  
      self.ln.zSetSurfaceData(surfNum=adjust_surf, code=self.ln.SDAT_THICK, value=-value);
    return self.ln.zPushLens(1);

  # Wrapper for simulating ZEMAX operands follow
  def TTHI(self,surf,adjust_surf,val):
    return self.change_thickness(val,surf,adjust_surf=adjust_surf);
  def TEDX(self,firstSurf,lastSurf,val):
    return self.tilt_decenter_elements(firstSurf,lastSurf,xdec=val)
  def TEDY(self,firstSurf,lastSurf,val):
    return self.tilt_decenter_elements(firstSurf,lastSurf,ydec=val)
  def TETX(self,firstSurf,lastSurf,val):
    return self.tilt_decenter_elements(firstSurf,lastSurf,xtilt=val)
  def TETY(self,firstSurf,lastSurf,val):
    return self.tilt_decenter_elements(firstSurf,lastSurf,ytilt=val)
  def TETZ(self,firstSurf,lastSurf,val):
    return self.tilt_decenter_elements(firstSurf,lastSurf,ztilt=val)
    


if __name__ == '__main__':
  import os
  from zemax_dde_link import *
  
  with DDElinkHandler() as hDDE:
    
    # start tolerancing
    filename= os.path.realpath('./tests/pupil_slicer.ZMX');
    tol=ToleranceSystem(hDDE,filename);
    tol.print_LDE();
  
    tol.change_thickness(5, value=2)
    tol.tilt_decenter_elements(1,3,ydec=0.02);  # [mm]
    tol.TETX(1,3,0.001)
    tol.print_current_geometric_changes();
    #  changes to system by hand:
    #
    #  ln.zSetSurfaceData(surfNum=5, code=ln.SDAT_THICK, value=2);
    #  TEDY=20./1000;    # [mm=1000um]
    #  ln.zSetSurfaceData(surfNum=1, code=ln.SDAT_DCNTR_Y_BEFORE, value=TEDY);
    #  ln.zSetSurfaceData(surfNum=3, code=ln.SDAT_DCNTR_Y_AFTER, value=-TEDY);
    #
    #  TETX=3./60;  # [deg=60'=17mrad]
    #  ln.zTiltDecenterElements(15,18,xtilt=TETX)
