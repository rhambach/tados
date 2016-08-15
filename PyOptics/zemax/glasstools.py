# -*- coding: utf-8 -*-
'''
Created on Fri Aug 12 13:37:00 2016

@author: Uwe Lippmann

This script reads glass data from a Zemax AGF file, and plots a modified Abbe
diagram for an arbitrary set of wavelengths.
'''

import codecs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor


# Common wavelengths in microns
wave = {'F': 0.4861327,
        'd': 0.5875618,
        'C': 0.6562725}


def extended(wavelength, param):
    return schott(wavelength, param)
    
    
def extended2(wavelength, param):
    index = param[0]
    index += param[1] * wavelength**2
    index += param[2] / wavelength**2
    index += param[3] / wavelength**4
    index += param[4] / wavelength**6
    index += param[5] / wavelength**8
    index += param[6] * wavelength**4
    index += param[7] * wavelength**6
    return np.sqrt(index)


def extended3(wavelength, param):
    index = param[0]
    index += param[1] * wavelength**2
    index += param[2] * wavelength**4
    index += param[3] / wavelength**2
    index += param[4] / wavelength**4
    index += param[5] / wavelength**6
    index += param[6] / wavelength**8
    index += param[7] / wavelength**10
    index += param[8] / wavelength**12
    return np.sqrt(index)


def handbook1(wavelength, param):
    index = param[0] + param[1] / (wavelength**2 - param[2]) - param[3] * wavelength**2
    return np.sqrt(index)
    

def handbook2(wavelength, param):
    index = param[0] + param[1] * wavelength**2 / (wavelength**2 - param[2]) - param[3] * wavelength**2
    return np.sqrt(index)
    
    
def herzberger(wavelength, param):
    L = 1 / (wavelength**2 - 0.028)
    index = param[0] + param[1] * L + param[2] * L**2
    for i in xrange(3, len(param)):
        index += param[i] * wavelength**(2*(i-2))


def schott(wavelength, param):
    index = param[0]
    index += param[1] * wavelength**2
    for i in xrange(2, len(param)):
        index += param[i] / wavelength**(2*(i-1))
    return np.sqrt(index)


def sellmeier1(wavelength, param):
    index = 0.0
    for i in xrange(0, len(param) // 2):
        index += param[2*i] * wavelength**2 / (wavelength**2 - param[2*i+1])
    return np.sqrt(index + 1)


def sellmeier2(wavelength, param):
    index = param[0] + param[1] * wavelength**2 / (wavelength**2 - param[2]**2) + param[3] / (wavelength**2 - param[4]**2)
    return np.sqrt(index + 1)

def sellmeier3(wavelength, param):
    return sellmeier1(wavelength, param)


def sellmeier4(wavelength, param):
    index = param[0] + param[1] * wavelength**2 / (wavelength**2 - param[2]) + param[3] * wavelength**2 / (wavelength**2 - param[4])
    return np.sqrt(index)

def sellmeier5(wavelength, param):
    return sellmeier1(wavelength, param)


def conrady(wavelength, param):
    return param[0] + param[1] / wavelength + param[2] / wavelength**(3.5)


glass_formula = (None, schott, sellmeier1, herzberger, sellmeier2, conrady, sellmeier3, handbook1, handbook2,
                 sellmeier4, extended, sellmeier5, extended2, extended3)
glass_status = ('Standard', 'Preferred', 'Obsolete', 'Special', 'Melt')


def index(glass, wavelength=0.5875618):
    """
    Calculate the refractive index for a given wavelength.

    Parameters
    ----------
        glass : dict
            Glass data as returned by ``read_agf_file()``.
        wavelength : float
            Wavelength for the index calculation in microns. Default: 0.5875618 (helium d line)
            
    Returns
    -------
        float
            Index of refraction.
    """
    return glass['formula'](wavelength, glass['dispersion_data'])


def abbe(glass, wave1=0.4861327, wave2=0.5875618, wave3=0.6562725):
    """
    Calculate the Abbe number for a set of wavelengths
    
    Parameters
    ----------
        glass : dict
            Glass data as returned by ``read_agf_file()``.
        wave1 : float
            Lower boundary of the wavelength range in microns. Default: 0.4861327 (hydrogen F line)
        wave2 : float
            Center of the wavelength range in microns. Default: 0.5875618 (helium d line)
        wave3 : float
            Upper boundary of the wavelength range in microns. Default: 0.6562725 (ydrogen C line)

    Returns
    -------
        float
            Abbe number.
    """
    n1 = glass['formula'](wave1, glass['dispersion_data'])
    n2 = glass['formula'](wave2, glass['dispersion_data'])
    n3 = glass['formula'](wave3, glass['dispersion_data'])
    return (n2 - 1) / (n1 - n3)


def read_agf_file(filename, encoding='ascii'):
    """
    Read a Zemax AGF glass catalog file and return glass data as dictionary.
    
    Parameters
    ----------
        filename : str
            Name of the glass catalog file to be read.
        encoding : str
            Encoding of the text file (``'ascii'``, ``'utf-8'``, ``'utf-16'``). Default is ``'ascii'``.
    
    Returns
    -------
        dict
            Dictionary containing the glass data.
            
    Note
    ----
        Current Zemax AGF files come with a lot of exceptions (omitted values, incomplete lines, etc.) and hardly comply
        with their own specification. Non-complying lines are simply skipped at the moment.
            
    Example
    -------
        >>> cat = read_agf_file("schott.agf")
        >>> cat
        {'F2': {  'ar': 2.3,
                  'cr': 1.0,
                  'density': 3.599,
                  'dispersion_data': (1.34533359, 0.00997743871, 0.209073176, 0.0470450767, 0.937357162, 111.886764),
                  'dpgf': 0.0002,
                  'exclude_sub': True,
                  'formula': <function __main__.sellmeier1>,
                  'fr': 0.0,
                  'ignore_expansion': True,
                  'max_lambda': 2.5,
                  'melt_freq': 0,
                  'mil': '620364.360',
                  'min_lambda': 0.32,
                  'nd': 1.62004,
                  'pr': 1.3,
                  'rel_cost': 1.6,
                  'sr': 1.0,
                  'status': 'Preferred',
                  'tce_-30_70': 8.2,
                  'tce_100_300': 9.2,
                  'thermal_data': (1.51e-06, 1.56e-08, -2.78e-11, 9.34e-07, 1.04e-09, 0.25, 20.0),
                  'transmission': [0.0, 0.211, 0.78, 0.921, 0.94, 0.963, 0.977, 0.985, 0.987, 0.991, ... ],
                  'transmission_lambda': [0.32, 0.334, 0.35, 0.365, 0.37, 0.38, 0.39, 0.4, 0.405, 0.42, ... ],
                  'transmission_thickness': [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, ... ],
                  'vd': 36.37}}
    """
    catalog = {}
    # First read the catalog comment
    with codecs.open(filename, "r", encoding=encoding) as glassfile:
        name = ''
        for line in glassfile:
            try:
                line = line.strip().split(" ", 1)
                field = line[0].upper()
                if field == 'CC':
                    comment = line[-1]
                elif field == 'NM':
                    line = line[-1].split(" ")
                    name = line[0]
                    catalog[name] = {'formula': glass_formula[int(line[1])],
                                     'mil': line[2],
                                     'nd': float(line[3]),
                                     'vd': float(line[4]),
                                     'exclude_sub': bool(line[5]),
                                     'status': glass_status[int(line[6])],
                                     'transmission_lambda': [],
                                     'transmission': [],
                                     'transmission_thickness': []}
                    try:
                        catalog[name]['melt_freq'] = int(line[7])
                    except:
                        catalog[name]['melt_freq'] = 0
                elif field == 'GC':
                    catalog[name]['comment'] = line[-1]
                elif field == 'ED':
                    line = line[-1].split(" ")
                    catalog[name]['tce_-30_70'] = float(line[0])
                    catalog[name]['tce_100_300'] = float(line[1])
                    catalog[name]['density'] = float(line[2])
                    catalog[name]['dpgf'] = float(line[3])
                    catalog[name]['ignore_expansion'] = bool(line[4])
                elif field == 'CD':
                    param = []
                    for p in line[-1].split(" "):
                        param.append(float(p))
                    # remove last parameters that are zero
                    for i in xrange(1, len(param)):
                        if param[-1] != 0.0:
                            break
                        param.pop()
                    catalog[name]['dispersion_data'] = tuple(param)
                elif field == 'TD':
                    param = []
                    for p in line[-1].split(" "):
                        param.append(float(p))
                    catalog[name]['thermal_data'] = tuple(param)
                elif field == 'OD':
                    line = line[-1].split(" ")
                    try:
                        catalog[name]['rel_cost'] = float(line[0])
                    except ValueError:
                        catalog[name]['rel_cost'] = 0.0
                    catalog[name]['cr'] = float(line[1])
                    catalog[name]['fr'] = float(line[2])
                    catalog[name]['sr'] = float(line[3])
                    catalog[name]['ar'] = float(line[4])
                    catalog[name]['pr'] = float(line[5])
                elif field == 'LD':
                    line = line[-1].split(" ")
                    catalog[name]['min_lambda'] = float(line[0])
                    catalog[name]['max_lambda'] = float(line[1])
                elif field == 'IT':
                    line = line[-1].split(" ")
                    catalog[name]['transmission_lambda'].append(float(line[0]))
                    catalog[name]['transmission'].append(float(line[1]))
                    catalog[name]['transmission_thickness'].append(float(line[2]))
            except:
                continue
    return catalog


def abbe_plot(catalog, wave1=0.4861327, wave2=0.5875618, wave3=0.6562725):
    """
    Generate an Abbe diagram from a Zemax AGF glass catalog file for an arbitrary set of wavelengths.
    
    Parameters
    ----------
        catalog : dict
            Glass catalog to be displayed (as returned by read_agf_file()).
        wave1 : float
            Lower boundary of the wavelength range in microns. Default: 0.4861327 (hydrogen F line)
        wave2 : float
            Center of the wavelength range in microns. Default: 0.5875618 (helium d line)
        wave3 : float
            Upper boundary of the wavelength range in microns. Default: 0.6562725 (ydrogen C line)
    
    Example
    -------
        >>> # Read glass catalog file
        >>> catalog = read_agf_file("schott.agf")
        >>> # "Abbe" diagram for near-infrared
        >>> abbe_plot(catalog, 0.8, 0.85, 0.9)
    """
    fig = plt.figure(figsize=(8, 6)) # facecolor='white'
    ax = plt.axes()
    # adds grid
    plt.grid(True)
    plt.locator_params(nbins=20) # grid size
    
    points_with_annotation = list() # data point with annotation

    # Facecolor indicate the Status as same as Zemax indicated:Status is 0 for Standard,
    # 1 for Preferred, 2 for Obsolete, 3 for Special, and 4 for Melt.
    fc = {'Standard': 'black',
          'Preferred': 'green',
          'Obsolete': 'red',
          'Special': 'blue',
          'Melt': 'yellow'}

    for glass in catalog:
        n = index(catalog[glass], wave2)
        v = abbe(catalog[glass], wave1, wave2, wave3)
        point, = plt.plot(v, n, 'o', markersize=10, markerfacecolor=fc[catalog[glass]['status']])
        annotation = ax.annotate("%s $nd$=%f $vd$=%f" % (glass, n, v),
                                 xy=(v, n), xycoords='data',xytext=(v, n), textcoords='data', horizontalalignment="left",
                                 bbox=dict(boxstyle="round", facecolor="w", edgecolor="0.5", alpha=0.9))
                                 
        # by default, disable the annotation visibility
        annotation.set_visible(False)
        points_with_annotation.append([point, annotation])
    
    ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])  # extend of abbe number(Vd)
    ax.set_ylim(1.4, 2.1) # extend of index(Nd)
    ax.set_title('Modified Abbe diagram')
    s1 = "n({}) - 1".format(wave2)
    s2 = "n({}) - n({})".format(wave1, wave3)
    ax.set_xlabel(r'Equivalent Abbe number $\nu = \frac{' + s1 + '}{' + s2 + '}$', fontsize=16)
    ax.set_ylabel('Refractive Index $n({:1})$'.format(wave2), fontsize=16)

    # Thanks to pelson (https://stackoverflow.com/users/741316/pelson) for providing the method at stackoverflow.com
    # https://stackoverflow.com/questions/11537374/matplotlib-basemap-popup-box#new-answer
    def on_move(event):
        visibility_changed = False
        for point, annotation in points_with_annotation:
            should_be_visible = (point.contains(event)[0] == True)
    
            if should_be_visible != annotation.get_visible():
                visibility_changed = True
                annotation.set_visible(should_be_visible)
    
        if visibility_changed:
            plt.draw()
    
    on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    from os.path import expanduser
    
    home = expanduser("~")
    
    wave1 = 0.8
    wave2 = 0.85
    wave3 = 0.9
    
    filename = home + r"\Documents\Zemax\Glasscat\schott.agf"
    print filename
    
    cat = read_agf_file(filename, encoding='ascii')
    abbe_plot(cat, wave1, wave2, wave3)
    