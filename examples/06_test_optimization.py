# -*- coding: utf-8 -*-
"""
test minimization using scipy.optimize, see
http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

@author: Hambach
"""

import logging
import numpy as np
from scipy.optimize import minimize
import matplotlib.pylab as plt

import _set_pkgdir
import PyOptics.optimization as opt
from PyOptics.zemax import dde_link

if __name__ == '__main__':
  import os as os
  logging.basicConfig(level=logging.INFO);
  
  with dde_link.DDElinkHandler() as hDDE:
    # initialize system
    filename= os.path.realpath('../tests/zemax/asphere_optimization.ZMX');
    zmx_opt = opt.External_Zemax_Optimizer(hDDE,filename);
    # run optimization using specified method
    method=None;    
    x0=zmx_opt.getSystemState();
    zmx_opt.print_status();
    res=minimize(zmx_opt.evaluate,x0,method=method,callback=zmx_opt.print_status)    
    # analyze results
    zmx_opt.showSystem(res.x);
    print(res.message)

    
    
    
    