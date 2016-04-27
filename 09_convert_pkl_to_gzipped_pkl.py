# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:48:35 2016

Aim: standard use of pickle produces huge files ~100Mb
     the reason is that the standard protokol produces text files, 
     which are not zipped
     -> we use higher protocol (reduces file size to 30Mb)
     -> and pass output through gzip (reduces files size to 1Mb)!
     -> bz2 leads to smaller file size (0.6Mb), but takes much longer


@author: Hambach
"""


import os
import glob
import cPickle as pickle
import gzip


path = os.path.realpath('../');
pattern = os.path.join(path,'*','run*.pkl');
protocol=-1;
for pklfile in sorted(glob.glob(pattern)):
  
  print('read pklfile: %s'%pklfile);      
  f = open(pklfile,'rb');    
  save = pickle.load(f);
  f.close();

  root,ext = os.path.splitext(pklfile);
  outfile  = root + '.pkz';
  print(' -> zip+save: %s'%outfile);
  with gzip.open(outfile,'wb') as OUT:
    pickle.dump(save,OUT,protocol);
    