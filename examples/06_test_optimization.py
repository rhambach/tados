# -*- coding: utf-8 -*-
"""
test minimization using scipy.optimize, see
http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

@author: Hambach
"""

import logging
import numpy as np
from scipy.optimize import minimize,least_squares
import matplotlib.pylab as plt

import _set_pkgdir
import PyOptics.optimization as opt
from PyOptics.zemax import dde_link


def minimize_scalar_FOM(filename,**kwargs):
  """
    example for minimization of scalar figure-of-merit from Zemax using
    scipy.optimize.minimze(...,**kwargs) 
  """    
  print("#"*100);
  print("# minimize_scalar_FOM() for file '%s'"%filename);
  print("#"*100);  
  zmx_opt = opt.External_Zemax_Optimizer(hDDE,filename);
  # run optimization using specified method
  x0=zmx_opt.getSystemState();
  zmx_opt.print_status(); zmx_opt.print_status(x0);
  res=minimize(zmx_opt.evaluate,x0,callback=zmx_opt.print_status,
               options={'disp':True},**kwargs);  
  # print results
  print(res.message);
  zmx_opt.showSystem(res.x);    # update Zemax LDE
  #hDDE.link.ipzCaptureWindow('L3d',flag=1)
  #hDDE.link.ipzCaptureWindow('Spt',flag=1) 
  

def minimize_vector_FOM(filename,**kwargs):
  """
    example for using a least-squares fit to the individual rows of the
    merit function editor to achieve better convergence, uses 
    scipy.optimize.least_squares(...,**kwargs) from Scipy v>1.7.1
  """
  print("#"*100);
  print("# minimize_scalar_FOM() for file '%s'"%filename);
  print("#"*100);  
  zmx_opt = opt.External_Zemax_Optimizer(hDDE,filename);
  # run optimization using specified method
  x0=zmx_opt.getSystemState();
  zmx_opt.print_status(); zmx_opt.print_status(x0);
  res=least_squares(zmx_opt.evaluate_operands,x0,x_scale=np.abs(x0),
                    verbose=2,**kwargs)    
  # print results
  print(res.message);
  zmx_opt.showSystem(res.x);    # update Zemax LDE
  #hDDE.link.ipzCaptureWindow('L3d') 
  #hDDE.link.ipzCaptureWindow('Spt')   
  

if __name__ == '__main__':
  import os as os
  logging.basicConfig(level=logging.INFO);   # use logging.DEBUG for more info
  
  with dde_link.DDElinkHandler() as hDDE:

    # Optimization of an Asphere (simple)
    filename= os.path.realpath('../tests/zemax/asphere_optimization.ZMX');
    minimize_scalar_FOM(filename,method="BFGS");
    print("\n>>> See Zemax Window for optimized system")    
    raw_input("Press Enter to continue...")
    
    # Optimization of gracing incidence mirror (hard problem, as curvature and tilt-angle
    # are very much interdependent). A simple optimization of the scalar FOM does not work.
    # (Nelder-Mead seams to work best, L-BFGS-B does nothing)
    filename = os.path.realpath('../tests/zemax/gracing_incidence_mirror_optimization.ZMX');
    # restrict tilt angle to 0-360 degree    
    bounds = np.tile((-np.inf,np.inf),(4,1));  bounds[1]=(0,360);
    minimize_scalar_FOM(filename,method='Nelder-Mead',bounds=bounds);
    print("\n>>> See Zemax Window for optimized system")    
    raw_input("Press Enter to continue...")
    
    # more advanced optimization using least-squares fit of individual rows of the MFE
    minimize_vector_FOM(filename,method='dogbox',bounds=bounds.T,jac='3-point');
    print("\n>>> See Zemax Window for optimized system")    
    
    
    