# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:11:41 2016

@author: Hambach
"""

# field sampling (octagonal fiber)
# sampling should be rational approx. of tan(pi/8), using continued fractions:
# approx tan(pi/2) = [0;2,2,2,2,....] ~ 1/2, 2/5, 5/12, 12/29, 29/70, 70/169, 169/408
# results in samplings: (alwoys denominator-1): 4,11,28,69,168,407
for N in (11,28,69,168,407):
  v = np.linspace(-1,1,N);
  x,y = np.meshgrid(v,v,indexing='ij')
  ind = (np.abs(x)<=1) & (np.abs(y)<=1) & \
              (np.abs(x+y)<=np.sqrt(2)) & (np.abs(x-y)<=np.sqrt(2));
  out = open("octagon_%d.ima"%N,'w');
  out.write("%d\n"%N)
  for i in xrange(N):
    for j in xrange(N):
      out.write('%d'%ind[i,j]);
    out.write('\n');
  out.close();
  #np.savetxt("octagon_%d.ima"%N, ind, fmt='%d', header='%d'%N, comments='');
  