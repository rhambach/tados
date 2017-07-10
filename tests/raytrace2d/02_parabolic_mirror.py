# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

import _set_pkgdir
import PyOptics.raytrace2d as rt

# source
source = rt.CollimatedBeam(20,-10,0,n=1);
#source = sources.PointSource((-10,0),amin=-10,amax=10,n=1)

# parabolic mirror
N=10000; #number of segments for parabolic mirror
f=-11;   #focal length of parabolic mirror
y=np.linspace(-12,8,N);
z=0.25/f*y**2

# optical system (sequential list of surfaces, position in global coordinates)
system = [];
system.append( rt.SegmentedSurface(y,z,n=-1) );  # parabolic mirror
system.append( rt.PlaneSurface((-1,f),(1,f),n=-1))

# plot system layout
tracer = rt.Raytracer(source,system);
tracer.print_system(verbosity=1);
tracer.trace(nRays=11);
rt.SimpleLayout(tracer).plot();

# plot spot
tracer.trace(nRays=1001);       # number should not be commensurable with number of segments N
rt.Footprint(tracer,surf=-1).plot(bins=50);
plt.show();
  