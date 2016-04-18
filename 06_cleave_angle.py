# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 18:30:25 2016

@author: Hambach
"""

import numpy as np;
import matplotlib.pylab as plt;

# numerical aperture of beam in air
ncore = 1.4475;         # refractive index of SiO2
NA = np.linspace(-0.2,0.2,100);
gamma = 60./180*np.pi;   # cleave angle of fiber [rad]


alpha = np.arcsin(NA/ncore);  # ray angle to axis inside fiber
NAp= ncore*np.sin(alpha+gamma); # outside fiber

plt.figure();
plt.plot(NA,NA,':')
plt.plot(NA,NAp); 
plt.title('change of angle distribution after tilted fiber end');


# same for different cleave angles
gamma = np.linspace(-100,130,100)/180*np.pi;
alpha = np.arcsin(0.2/ncore);
NAp1 = ncore*np.sin(alpha+gamma);
NAp2 = ncore*np.sin(-alpha+gamma);
plt.figure()
plt.plot(gamma,NAp1);
plt.plot(gamma,NAp2);
plt.title('change of output angles for different cleave angles');