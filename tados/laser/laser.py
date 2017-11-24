# -*- coding: utf-8 -*-
"""
Module providing a functions for calculations with Gaussian beams

@author: Lippmann
"""


from __future__ import print_function, division
import numpy as np


def w2fwhm(w, n=2.0):
    return 2 * w * ((-0.5*np.log(0.5))**(1.0/n))

    
def fwhm2w(fwhm, n=2.0):
    return 0.5 * fwhm / ((-0.5*np.log(0.5))**(1.0/n))
    
    
def divergence(waist, wavelength):
    return np.arctan(wavelength / (np.pi * waist))

    
def waist(divergence, wavelength):
    return wavelength / (np.pi * np.tan(divergence))


def rayleigh(waist, wavelength):
    return np.pi * waist**2 / wavelength

    
def image_waist_position(a0, focal_length, waist, wavelength):
    return -focal_length**2 / ((a0 + focal_length)**2 + rayleigh(waist, wavelength)**2) * (a0 + focal_length) + focal_length

    
def max_waist_distance(focal_length, waist, wavelength):
    return focal_length + 0.5 * focal_length**2 / rayleigh(waist, wavelength)

    
def image_waist_size(obj_waist_distance, focal_length, waist, wavelength):
    return np.sqrt(focal_length**2 / ((obj_waist_distance + focal_length)**2 + rayleigh(waist, wavelength)**2) * waist**2)

    
def image_rayleigh(obj_waist_distance, focal_length, obj_rayleigh):
    return focal_length**2 / ((obj_waist_distance + focal_length)**2 + obj_rayleigh**2) * obj_rayleigh

    
def beam_size(position, waist, wavelength):
    return waist * (1 + (position / rayleigh(waist, wavelength))**2)

    
def beam_radius(position, waist, wavelength):
    rayl = rayleigh(waist, wavelength)
    return -position * (1 + (rayl / position)**2)

def intensity_profile(x, w, n=2.0):
    return np.exp(-2*(x/w)**n)

    
if __name__ == "__main__":
    import unittest
    
    
    class LaserTests(unittest.TestCase):
        def test_w2fwhm(self):
            self.assertAlmostEqual(w2fwhm(1.0), 2.0/1.6986, places=4)
            self.assertAlmostEqual(w2fwhm(fwhm2w(3.0, 5.0), 5.0), 3.0)
        
        def test_fwhm2w(self):
            self.assertAlmostEqual(fwhm2w(2.0), 1.6986, places=4)
            
        def test_intensity_profile(self):
            self.assertAlmostEqual(intensity_profile(0.0, 1.0), 1.0)
            self.assertAlmostEqual(intensity_profile(23.5, 23.5), np.exp(-2))
            self.assertAlmostEqual(intensity_profile(-23.5, 23.5), np.exp(-2))
            self.assertAlmostEqual(intensity_profile(23.5, 23.5, n=4.0), np.exp(-2))
            self.assertAlmostEqual(intensity_profile(-23.5, 23.5, n=4.0), np.exp(-2))
            self.assertAlmostEqual(intensity_profile(0.5 * w2fwhm(23.5), 23.5), 0.5)
            self.assertAlmostEqual(intensity_profile(-0.5 * w2fwhm(23.5), 23.5), 0.5)
            self.assertAlmostEqual(intensity_profile(0.5 * w2fwhm(23.5, n=4.0), 23.5, n=4.0), 0.5)
            self.assertAlmostEqual(intensity_profile(-0.5 * w2fwhm(23.5, n=4.0), 23.5, n=4.0), 0.5)
            self.assertTupleEqual(tuple(np.round(intensity_profile(np.array([0.0, 0.5 * w2fwhm(23.5, n=4.0), 23.5]), 23.5, n=4.0), decimals=10)), (1.0, 0.5, np.round(np.exp(-2), decimals=10)))
            self.assertTupleEqual(tuple(np.round(intensity_profile(np.array([0.0, -0.5 * w2fwhm(23.5, n=4.0), -23.5]), 23.5, n=4.0), decimals=10)), (1.0, 0.5, np.round(np.exp(-2), decimals=10)))

            
    unittest.main()
    