# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:55:26 2016

@author: Hambach
"""

import numpy as np
import matplotlib.pylab as plt
import os
import glob
import cPickle as pickle
import gzip

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
 

path    = os.path.realpath('../05_monte_carlo_simulation_system11_NA0178');
pattern = os.path.join(path,'run*.pkz');

# plot for enboxed energy:
fig1,_ = plt.subplots(nrows=1,ncols=3,sharey=True,sharex=True);   
fig1.suptitle("Loss of transmission for different x-width of fiber");
fig2,_ = plt.subplots(nrows=1,ncols=3,sharey=True,sharex=True);   
fig2.suptitle("Loss of transmission for different y-width of fiber");
for ax in fig1.axes: ax.set_xlabel("fiber x-width [um]");
for ax in fig2.axes: ax.set_xlabel("fiber y-width [um]");
for f in (fig1,fig2): f.axes[0].set_ylabel("transmission loss [W]");

footprint=0; Itot0_list=[];
for pklfile in sorted(glob.glob(pattern)):
  # read data from gzipped pkl-file
  print('read pklfile: %s'%pklfile); 
  f = gzip.open(pklfile,'rb');    
  save = pickle.load(f);
  f.close();     
  
  # plot footprint (superimposed for all runs)
  footprint += save['img'][0].get_footprint()[2];    # superimpose rotz=0 footprint for all trials        
  Itot0_list.append(save['Itot'][0]); 
  
  for i in xrange(len(save['img'])):  # corresponds to different rotz values
    # unpack data from save dictionary
    rotz  = save['rotz'][i];
    img   = save['img'][i];
    params= save['params'][i];
    Itot  = save['Itot'][i];
    Itot0 = save['Itot'][0];   # total intensity for rotz=0 as reference
    assert abs(rotz-0.5*i)<1e-6, "we expect rotz=0,0.5,1 (in this order)"
    
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
    x_boxx,I_boxx = intensity_in_box(x,cumx,min_width=0.10,max_width=0.14)
    y_boxy,I_boxy = intensity_in_box(y,cumy,min_width=0.03,max_width=0.04);
  
    # calculate loss and add to plot
    fig1.axes[i].plot(x_boxx*1000, Itot0-I_boxx,'k-',alpha=0.2);
    fig2.axes[i].plot(y_boxy*1000, Itot0-I_boxy,'k-',alpha=0.2);
     
 # end: for i

# end: for pklfile

# final scaling
fig1.axes[0].set_xlim(115,130);
fig1.axes[0].set_xticks(np.arange(115,131,5))
fig1.axes[0].set_ylim(0,0.02);
for i in range(3): fig1.axes[i].axvline(125,color='r',linestyle='--');
  
fig2.axes[0].set_xlim(32,36);
fig2.axes[0].set_xticks(np.arange(32,36.5,1))
fig2.axes[0].set_ylim(0,0.02);
for i in range(3): fig2.axes[i].axvline(33.5,color='r',linestyle='--');

img.intensity = footprint;  # overwrite last existing detector
fig3 = img.show();
fig3.axes[0].images[0].set_clim(vmax=10);  # increase contrast to show extension of footprint
#fig3.axes[0].set_ylim(-0.025,0.025);
#fig3.axes[1].set_ylim(0,40);
plt.draw();
    
# plot variation of Itot0 over system runs
fig4 = plt.figure();
plt.plot(Itot0_list);
fig4.axes[0].set_xlabel("Monte-Carlo run");
fig4.axes[0].set_ylabel("total intensity (without rotz) [W]");
fig4.axes[0].set_title(os.path.split(path)[-1])