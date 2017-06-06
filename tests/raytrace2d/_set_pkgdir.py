# -*- coding: utf-8 -*-
"""
  Unique place for all runscripts to set the module path.
  (i.e. if you have a copy of TEMareels in some place on your
   hard disk but not in PYTHONPATH). 

  Copyright (c) 2016, rhambach. 
    This file is part of the PyOptics package and released
    under the MIT-Licence. See LICENCE file for details.
"""

# location of the PyOptics package on the hard disk
# (if not specified in PYTHONPATH)
pkgdir = '../../'; 
import sys
from   os.path import abspath;
sys.path.insert(0,abspath(pkgdir));