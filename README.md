# Tados
Tools for the Analysis and Design of Optical Systems 

## Features

* advanced interaction between Python and FRED / Zemax
* adaptive pupil sampling
* fast computation of intensity maps (footprint plots) using triangulation
* advanced tolerancing via Python script
* interface for external optimization
* minimal 2D sequential raytracer for freeform surfaces

## Dependencies

* numpy, matplotlib
* [pyzdde](https://github.com/indranilsinharoy/PyZDDE)

## Examples

<img src="examples/results/05_fraunhofer_skinny_triangles_split.png" width="55%" align="middle"> <img src="examples/results/07_pentagon_raytrace.png" width="44%" align="middle"> 
<img src="tests/results/01_image_slicer_Zemax_GIA_with_10mio_rays.png" alt="intensity map using triangulation" width="35%" align="middle"> <img src="examples/results/03_monte_carlo_simulation_system13_NA0178_ywidth.png" alt="monte carlo simulation for enboxed energy" width="60%" align="middle">
