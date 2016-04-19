# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:55:26 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt
import os
import glob
import pickle
  
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

path    = os.path.realpath('../03_test_sensitivity_system_14_NA0178');
pattern = os.path.join(path,'run*.pkl');

for pklfile in sorted(glob.glob(pattern)):
  # read data from pkl-file    
  f = open(pklfile,'rb');
  save = pickle.load(f);
  f.close();
  
  disturb_name=save['disturb_name'];
  print('\n-- %10s -----------------------------------------------'%disturb_name);
  print('  pklfile: %s'%pklfile);    
  print('      dx        dy        |(dx,dy)|       rotz       Itot[W]');
   
  for i in xrange(len(save['img'])):
    # unpack data from save dictionary
    xscale,yscale = save['scalexy'][i];
    xs,ys = save['dxy'][i];
    rotz  = save['rotz'][i];
    img   = save['img'][i];
    params= save['params'][i];
    Itot  = save['Itot'][i];
    
    # plot footprint
    if rotz==0: 
      fig2 = img.show();
      fig2.axes[0].set_ylim(-0.025,0.025);
      fig2.axes[1].set_ylim(0,40);
    
    # analyze img detector (enboxed energy along x and y):
    x,intx = img.x_projection(fMask=lambda x,y: x>0.07);
    y,inty = img.y_projection(fMask=lambda x,y: x>0.07);
    dx=x[1]-x[0]; dy=y[1]-y[0];
    
    # total intensity in image
    Itot = np.sum(inty)*dy; # [W]
        
    # cumulative sum for getting percentage of loss
    cumx = np.cumsum(intx)*dx;
    cumy = np.cumsum(inty)*dy;
    
    # find intensity in box of given width along y    
    x_boxx,I_boxx = intensity_in_box(x,cumx,min_width=0.12,max_width=0.14)
    y_boxy,I_boxy = intensity_in_box(y,cumy,min_width=0.03,max_width=0.04);
  
    # write results
    print('  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f'%(xs,ys,np.linalg.norm((xs,ys)),rotz,Itot));
        
  
    # plot enboxed energy:
    if rotz==0: #open new figure 
      fig1,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharey=True);   
      fig1.suptitle("tolerance '%s', dx=%8.5f, dy=%8.5f, Itot=%5.3fW" \
                  % (disturb_name,xs,ys,Itot));  
    ax1.plot(x_boxx*1000,I_boxx,label='rotz=%3.1f'%rotz);
    ax2.plot(y_boxy*1000,I_boxy,label='rotz=%3.1f'%rotz);
    ax1.set_xlabel('x-width [um]'); ax1.set_ylabel('Intensity [W]');
    ax2.set_xlabel('y-width [um]'); ax2.set_ylabel('Intensity [W]');
    ax1.set_ylim(0.95,1);  
    ax1.legend(loc=0);  
    ax2.legend(loc=0);
 