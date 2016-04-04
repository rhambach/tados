# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 10:37:13 2016

@author: Hambach
"""

from __future__ import division
import pyzdde.arraytrace as at  # Module for array ray tracing
import pyzdde.zdde as pyz
import logging
import numpy as np
import matplotlib.pylab as plt

class DDElinkHandler(object):
  """
  ensure that DDE link is always closed, see discussion in 
  http://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
  """
  def __init__(self):
    self.link=None;
  
  def __enter__(self):
    " initialize DDE connection to Zemax "
    self.link = pyz.createLink()
    if self.link is None:
      raise RuntimeError("Zemax DDE link could not be established.");
    return self;
    
  def __exit__(self, exc_type, exc_value, traceback):
    " close DDE link"
    self.link.close();

  def load(self,zmxfile):
    " load ZMX file with name 'zmxfile' into Zemax "
    ln = self.link;   

    # load file to DDE server
    ret = ln.zLoadFile(zmxfile);
    if ret<>0:
        raise IOError("Could not load Zemax file '%s'. Error code %d" % (zmxfile,ret));
    logging.info("Successfully loaded zemax file: %s"%ln.zGetFile())
    
    # try to push lens data to Zemax LDE
    ln.zGetUpdate() 
    if not ln.zPushLensPermission():
        raise RuntimeError("Extensions not allowed to push lenses. Please enable in Zemax.")
    ln.zPushLens(1)


class AnalyzeTransmission(object):

  def __init__(self, hDDE):
    self.DDElink=hDDE.link;

    
  def test(self):
        
    ln = self.DDElink;
    
    # pupil sampling
    px = np.linspace(-1,1,201); py=px;
    px,py=np.meshgrid(px,py);   
    ind = px**2 + py**2 <= 1;
    px=px[ind]; py=py[ind];
    nRays = px.size;
    print("number of rays: %d"%nRays);
        
    x = np.zeros(nRays);
    y = np.zeros(nRays)
        
        
    # fill in ray data array (following Zemax notation!)
    t = time.time();    
    rays = at.getRayDataArray(nRays, tType=0, mode=0, endSurf=-1)
    for k in xrange(nRays):
      rays[k+1].x = x[k]      
      rays[k+1].y = y[k]
      rays[k+1].z = px[k]
      rays[k+1].l = py[k]
      rays[k+1].intensity = 1.0
      rays[k+1].wave = 1;

    print("set pupil values: %ds"%(time.time()-t))

    # Trace the rays
    ret = at.zArrayTrace(rays, timeout=5000)
    print("zArrayTrace: %ds"%(time.time()-t))
#
    # collect results
    results = np.asarray( [(r.x,r.y,r.l,r.m,r.intensity) for r in rays[1:]] );

    
    print("retrive data: %ds"%(time.time()-t))    

    plt.figure();
    plt.plot(results[:,0], results[:,1],'.')    
    
    return results;






if __name__ == '__main__':
  import os as os
  import sys as sys
  logging.basicConfig(level=logging.INFO);
  
  with DDElinkHandler() as hDDE:
  
    ln = hDDE.link;
    # load example file
    filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
                            'Cooke 40 degree field.zmx')
    hDDE.load(filename);
    
    AT = AnalyzeTransmission(hDDE);
    rd = AT.test();  