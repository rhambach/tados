# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:55:26 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt
import logging
import os
from tolerancing import *
from transmission import RectImageDetector
from zemax_dde_link import *
import cPickle as pickle
import gzip

def GeometricImageAnalysis(hDDE, testFileName=None):
  """
  perform Geometric Image Analysis in Zemax and return Detector information
    textFileName ... (opt) name of the textfile used for extracting data from Zemax
    
  Returns: (img,params)
    img   ... RectImageDetector object containing detector data
    params... dictionary with parameters of the image analysis
  """  
  data,params = hDDE.zGeometricImageAnalysis(testFileName,timeout=1000);
  imgSize = float(params['Image Width'].split()[0]);
  
  # save results in RectImageDetector class
  img = RectImageDetector(extent=(imgSize,imgSize), pixels=data.shape);
  img.intensity = data.copy();
  return img,params
  
def find_quantiles(x,F,thresh):
  thresh = np.atleast_1d(thresh); 
  # offset by half bin width necessary, to have same results if x and F are reversed  
  return np.interp([thresh, F[-1]-thresh],F,x); 

def intensity_in_box(x,F,min_width=None,max_width=None):
  """
  calculate maximum intensity in a box of different width
    x: x-axis (pixel centers)
    F: cumulative distribution function (intensity)
    min_width: minimal width of box
    max_width: maximal width of box
  returns:
    w: widht of box
    I: intensity inside box
  """  
  dx=x[1]-x[0];
  if min_width is None: min_width=dx*1.5;          # nBox>0
  if max_width is None: max_width=dx*(x.size-0.5); # nBox<x.size
  assert min_width>dx, 'min_widht too small';
  assert max_width<dx*(x.size-1), 'max_width too large';
  nBox = np.arange(np.floor(min_width/dx),np.ceil(max_width/dx));
  w= dx*(nBox+1);           # width of the box with nBox pixels
  I=np.asarray([np.max(F[n:]-F[:-n]) for n in nBox]);
  return (w,I)
 
def compensator_rotz(tol,angle):  
  " apply rotation of slicer about surface nomal by given angle [deg]"
  ln = tol.ln; 
  tol.tilt_decenter_elements(6,8,ztilt=angle,cbComment1="compensator",cbComment2="~compensator");
  surf = tol.get_orig_surface(20);
  assert ln.zGetComment(surf)=="rotate image plane";
  # correct rotation of image plane
  angle_img = 90+np.rad2deg(np.arctan(np.sqrt(2)*np.tan(np.deg2rad(-22.20765+angle))));
  ln.zSetSurfaceParameter(surf,5,angle_img);   #  5: TILT ABOUT Z 
  


def tilt_obj(tol,xscale=0,yscale=0):   
  # corresponds to cleave angle of fiber 1
  tilt=np.tan(np.deg2rad(10)); # tilt by 10 deg
  if xscale<>0: tol.ln.zSetSurfaceParameter(0,1,tilt*xscale)   # set Param1: X TANGENT
  if yscale<>0: tol.ln.zSetSurfaceParameter(0,2,tilt*yscale)   # set Param1: X TANGENT 
  tol.ln.zGetUpdate(); 
  return tilt*xscale,tilt*yscale
  
def tilt_img(tol,xscale=0,yscale=0): 
  # corresponds to cleave angle of fiber 2  
  tilt=np.tan(np.deg2rad(10)); # tilt by 10 deg
  if xscale<>0: tol.ln.zSetSurfaceParameter(-1,1,tilt*xscale)   # set Param1: X TANGENT
  if yscale<>0: tol.ln.zSetSurfaceParameter(-1,2,tilt*yscale)   # set Param1: X TANGENT 
  tol.ln.zGetUpdate(); 
  return tilt*xscale,tilt*yscale
  
def decenter_L1(tol,xscale=0,yscale=0): 
  dcntr=0.001;  # [mm]
  tol.tilt_decenter_elements(1,3,xdec=dcntr*xscale,ydec=dcntr*yscale,
                             cbComment1="decenter L1", cbComment2="~decenter L1");
  return dcntr*xscale,dcntr*yscale

def tilt_L1(tol,xscale=0,yscale=0): 
  tilt=np.rad2deg(0.001); # [rad]
  tol.tilt_decenter_elements(1,3,xtilt=tilt*xscale,ytilt=tilt*yscale,
                             cbComment1="tilt L1", cbComment2="~tilt L1");
  return tilt*xscale,tilt*yscale
  
def decenter_L1surf3(tol,xscale=0,yscale=0): 
  dcntr=0.002; # [mm]
  tol.tilt_decenter_surface(3,xdec=dcntr*xscale,ydec=dcntr*yscale);
  return dcntr*xscale,dcntr*yscale

def decenter_F1L1(tol,xscale=0,yscale=0):
  dcntr=0.020;  # [mm]
  tol.insert_coordinate_break(4,xdec=dcntr*xscale,ydec=dcntr*yscale,comment="decenter F1L1");
  return dcntr*xscale,dcntr*yscale

def tilt_F1L1(tol,xscale=0,yscale=0):
  tilt=np.rad2deg(0.001); # [rad]
  tol.insert_coordinate_break(4,xtilt=tilt*xscale,ytilt=tilt*yscale,comment="tilt F1L1");
  return tilt*xscale,tilt*yscale

def decenter_L3(tol,xscale=0,yscale=0): 
  dcntr=0.010;  # [mm]
  tol.tilt_decenter_elements(17,19,xdec=dcntr*xscale,ydec=dcntr*yscale,
                             cbComment1="decenter L3", cbComment2="~decenter L3");
  return dcntr*xscale,dcntr*yscale

def tilt_L3(tol,xscale=0,yscale=0): 
  tilt=np.rad2deg(0.001); # [rad]
  tol.tilt_decenter_elements(17,19,xtilt=tilt*xscale,ytilt=tilt*yscale,
                             cbComment1="tilt L3", cbComment2="~tilt L3");
  return tilt*xscale,tilt*yscale

def decenter_F2L3(tol,xscale=0,yscale=0):
  dcntr=0.010;  # [mm]
  tol.insert_coordinate_break(17,xdec=dcntr*xscale,ydec=dcntr*yscale,comment="decenter F2L3");
  return dcntr*xscale,dcntr*yscale

def tilt_F2L3(tol,xscale=0,yscale=0):
  tilt=np.rad2deg(0.001); # [rad]
  tol.insert_coordinate_break(17,xtilt=tilt*xscale,ytilt=tilt*yscale,comment="tilt F2L3");  
  return tilt*xscale,tilt*yscale

def decenter_L2(tol,xscale=0,yscale=0): 
  dcntr=0.005;  # [mm]
  tol.tilt_decenter_elements(13,15,xdec=dcntr*xscale,ydec=dcntr*yscale,
                             cbComment1="decenter L2", cbComment2="~decenter L2");
  return dcntr*xscale,dcntr*yscale

def tilt_L2(tol,xscale=0,yscale=0): 
  tilt=np.rad2deg(0.001); # [rad]
  tol.tilt_decenter_elements(13,15,xtilt=tilt*xscale,ytilt=tilt*yscale,
                             cbComment1="tilt L2", cbComment2="~tilt L2");
  return tilt*xscale,tilt*yscale

def tilt_single_mirror(tol,nMirror=1,xscale=0,yscale=0,zscale=0):
  tilt=np.rad2deg(0.001); # [rad]
  numSurf = tol.get_orig_surface(6);
  pos = tol.ln.zGetNSCPosition(numSurf,nMirror)._asdict();
  pos['tiltX'] += xscale*tilt;
  pos['tiltY'] += yscale*tilt;
  pos['tiltZ'] += zscale*tilt;
  tol.ln.zSetNSCPositionTuple(numSurf,nMirror,**pos)
  tol.ln.zGetUpdate();
  return tilt*xscale,tilt*yscale

def tilt_M1(tol,**kwargs):
  return tilt_single_mirror(tol,nMirror=1,**kwargs);

def tilt_M2(tol,**kwargs):
  return tilt_single_mirror(tol,nMirror=2,**kwargs);  
  
def tilt_slicer(tol,xscale=0,yscale=0): 
  tilt=np.rad2deg(0.005); # [rad]
  tol.tilt_decenter_elements(6,8,xtilt=tilt*xscale,ytilt=tilt*yscale,
                             cbComment1="tilt slicer", cbComment2="~tilt slicer");
  return tilt*xscale,tilt*yscale  

def tilt_F1(tol,xscale=0,yscale=0): # note: pivot of tilt is in the object plane !
  tilt=np.rad2deg(0.005); # [rad]
  tObj=tol.ln.zGetSurfaceData(0,tol.ln.SDAT_THICK);
  tol.ln.zSetSurfaceData(0,tol.ln.SDAT_THICK,0);   # remove object thickness
  tol.insert_coordinate_break(1,xtilt=tilt*xscale,ytilt=tilt*yscale,comment="tilt F1");  
  tol.ln.zSetSurfaceData(1,tol.ln.SDAT_THICK,tObj);# add object thickness
  tol.ln.zGetUpdate();
  return tilt*xscale,tilt*yscale

def tilt_F2(tol,xscale=0,yscale=0):
  tilt=np.rad2deg(0.005); # [rad]
  tol.insert_coordinate_break(20,xtilt=tilt*xscale,ytilt=tilt*yscale,comment="tilt F1");  
  return tilt*xscale,tilt*yscale      

def decenter_F1(tol,xscale=0,yscale=0):
  dcntr=0.002; # [mm]
  tol.insert_coordinate_break(1,xdec=dcntr*xscale,ydec=dcntr*yscale,comment="decenter F1");  
  return dcntr*xscale,dcntr*yscale       

def decenter_F2(tol,xscale=0,yscale=0):
  dcntr=0.005; # [mm]
  tol.insert_coordinate_break(20,xdec=dcntr*xscale,ydec=dcntr*yscale,comment="decenter F1");  
  return dcntr*xscale,dcntr*yscale
       
def slicer_mirror_separation(tol,dscale=0,xscale=0,yscale=0):
  thick=0.050; # mm
  if dscale==0: dscale=xscale;
  if dscale==0: dscale=yscale;# allow us to use xscale and yscale like in all other functions
  numSurf = tol.get_orig_surface(6);
  pos = tol.ln.zGetNSCPosition(numSurf,1)._asdict(); # get M1
  # shift mirror along global z-axis in order to avoid a change of the referenc point
  # -> also add chang in y-shift of M1  
  pos['y'] -= dscale*thick;
  pos['z'] += dscale*thick;
  tol.ln.zSetNSCPositionTuple(numSurf,1,**pos)
  tol.ln.zGetUpdate();
  return 0,thick*dscale

 
logging.basicConfig(level=logging.ERROR);
outpath = os.path.realpath('../05_monte_carlo_simulation_system11_NA0178');
logfile = os.path.join(outpath,'monte_carlo_simulation.txt');  # file for summary of MC simulation
  
with DDElinkHandler() as hDDE, open(logfile,'w') as OUT:
  ln = hDDE.link;
  # load example file
  #filename = os.path.join(ln.zGetPath()[1], 'Sequential', 'Objectives', 
  #                        'Cooke 40 degree field.zmx')
  #filename= os.path.realpath('../13_catalog_optics_1mm_pupil_inf-inf-relay_point_source_with_slicer_tolerancing.ZMX');
  filename= os.path.realpath('../11_catalog_optics_1mm_pupil_point_source_with_slicer_tolerancing.ZMX');  
  tol=ToleranceSystem(hDDE,filename)

  # define list of tolerances:
  tolerances = [
                decenter_L1, 
                tilt_L1,
                decenter_F1L1,
                tilt_F1L1,
                decenter_L3,
                tilt_L3,
                decenter_F2L3,
                tilt_F2L3,
                decenter_L2,
                tilt_L2,
              ];
  # init random generator
  seed = 22;  
  np.random.seed(seed)  

  # log system data organize logging    
  OUT.write('system: %s\n'%filename)
  OUT.write('seed: %d\n'%seed)
  OUT.write('tolerances included in Monte-Carlo simulation: \n')
  for f in tolerances:
    OUT.write(' - %s\n'%f.func_name );
  y_samples = np.arange(0.033,0.040,0.001);
  y_str = (' I_%dum[W]'*len(y_samples)) % tuple(1000*y_samples);
  OUT.write('\n  run     rotz    Itot[W] %s\n' % (y_str));

  # loop over monte-carlo trials
  for it in xrange(1000):
    pklfile = os.path.join(outpath,'run%04d.pkl'%(it));
    if os.path.exists(pklfile): continue;  # do not overwrite existing data
    #print pklfile;
    save=dict();
    save['tolerances']=[f.func_name for f in tolerances];    
    randn = np.random.normal(size=(len(tolerances),2)); 
    save['randn']=randn;
    for key in ('rotz','img','params','Itot'):  save[key]=[];
    
    # allow for compensators, here rotation about surface normal of slicer
    # fig1,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True);   
    for rotz in (0,0.5,1):
      save['rotz'].append(rotz);

      # restore undisturbed system
      tol.reset();
      tol.change_thickness(4,11,value=2);     # shift of pupil slicer
      #tilt_obj(tol,yscale=1)

      # disturb system
      for n,distrub_func in enumerate(tolerances):
        xs,ys = distrub_func(tol,xscale=randn[n,0],yscale=randn[n,1]);
        #print "tolerance '%20s': dx=%8.5f, dy=%8.5f, dr=%8.5f"%(distrub_func.func_name,xs,ys,np.linalg.norm((xs,ys)))
         
      # update changes
      tol.ln.zPushLens(1);    
      #if rotz==0: tol.print_current_geometric_changes();
    
      # compensator: rotate slicer around surface normal
      if rotz<>0: compensator_rotz(tol,rotz);
  
      # geometric image analysis
      img,params = GeometricImageAnalysis(hDDE);
      save['img'].append(img);
      save['params'].append(params);
      
      # analyze img detector (enboxed energy along x and y):
      x,intx = img.x_projection(fMask=lambda x,y: x>0.07);
      y,inty = img.y_projection(fMask=lambda x,y: x>0.07);
      dx=x[1]-x[0]; dy=y[1]-y[0];
      
      # total intensity in image
      Itot = np.sum(inty)*dy; # [W]
      save['Itot'].append(Itot);
      
      # cumulative sum for getting percentage of loss
      cumx = np.cumsum(intx)*dx;
      cumy = np.cumsum(inty)*dy;
      
      # find intensity in box of given width along y    
      x_boxx,I_boxx = intensity_in_box(x,cumx,min_width=0.12,max_width=0.14)
      y_boxy,I_boxy = intensity_in_box(y,cumy,min_width=0.03,max_width=0.04);
      I_boxy_samples = np.interp(y_samples,y_boxy,I_boxy);
      # log results
      print "\n run #%04d: rotz=%8.5f, Itot=%8.5fW"%(it,rotz,Itot),
      for i in xrange(3): 
        print ", I%d=%5.3fW"%(1000*y_samples[i],I_boxy_samples[i]),
      OUT.write('\n  %04d  %8.5f  %8.5f'%(it,rotz,Itot) +\
                ('  %8.5f'*len(y_samples)) % tuple(I_boxy_samples));
    # end: for rotz       
    
    # write gzipped pkl-file with computed data
    with gzip.open(pklfile,'wb') as f:
      protocol=-1;
      pickle.dump(save,f,protocol);
      
  # end: for it