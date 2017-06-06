# -*- coding: utf-8 -*-

import matplotlib.pylab as plt

import _set_pkgdir
import PyOptics.raytrace2d as rt

# source
source = rt.CollimatedBeam(10,0,10,n=1);

# optical system (sequential list of surfaces, position in global coordinates)
system = [];
system.append( rt.PropagateDistance(35) );   # virtual raypath beyond first mirror
system.append( rt.PlaneSurface([-10,10],[20,12],n=-1) );     # mirror;
system.append( rt.PlaneSurface([10,-22],[30,-20],n=-1.5) ); # interface to glass
system.append( rt.SegmentedSurface([-10,0,15,30],[-35,-30,-35,-30],n=-1)); # interface to air
system.append( rt.PropagateDistance(-35) ); # negative distance after mirror (propagate against ray direction)


# raytrace
tracer = rt.Raytracer(source,system);
tracer.print_system(verbosity=1);
tracer.trace(nRays=7);
rt.SimpleLayout(tracer).plot();
plt.show();
  