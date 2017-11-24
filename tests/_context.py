# -*- coding: utf-8 -*-
"""
  AIM: run tests always on local version independent of installed packages
  see http://docs.python-guide.org/en/latest/writing/structure/
  see https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder/33532002#33532002
  
  Copyright (c) 2016, rhambach. 
    This file is part of the tados package and released
    under the MIT-Licence. See LICENCE file for details.
"""

from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
testdir = os.path.dirname(current_path)
moduledir = testdir[:testdir.rfind(os.path.sep)]
if moduledir not in sys.path:
  sys.path.insert(1,moduledir)

import tados