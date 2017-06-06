# -*- coding: utf-8 -*-
 
import numpy as np
  
# --------------------------------------------------------------------
# Helper functions
#
def init_list1d(var,length,dtype,name):
  "repeat variable if it is scalar to become a 1d list of size length"
  var=np.atleast_1d(var);  
  if var.size==1: 
    return np.full(length,var[0],dtype=dtype);
  else:     
    assert var.size==length, 'incompatible length of list \'%s\': should be %d instead of %d'%(name,length,var.size)
    return var.flatten();    
    
