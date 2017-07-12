# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:51:05 2016

@author: ulip
"""

import time
import glob
import numpy as np
import scipy.misc


def fgd2array(f, header_info=False):
    """
    Construct an array from data in a FRED FGD file.
    
    Parameters
    ----------
        f : file or str
            Open the file object or filename.
        header_info : bool
            If ``True`` returns a dict with the contents of the header of the file
            alongside with the actual data array.
    
    Example
    -------
        >>> from tados.fred.fgdfile import fgd2array
        >>> data, header = fgd2array("datafile.fgd", header_info=True)
        >>> header
        {'ORIGIN_POSITION': array([ -34.86855408,  -98.8746984 ,  171.25600122]),
        'TITLE': '', 'B_AXIS_MIN': -2.7135, 'A_AXIS_MAX': 3.392, 'B_AXIS_TYPE': 'Spatial',
        'A_AXIS_TYPE': 'Spatial', 'DATATYPE': 'Double', 'B_AXIS_DIM': 1024, 'DATAUNITS': '',
        'A_AXIS_DIR': array([ 0.98480775, -0.08682409,  0.15038373]), 'A_AXIS_DIM': 1280,
        'B_AXIS_DIR': array([ 0.       ,  0.8660254,  0.5      ]), 'B_AXIS_LABEL': 'Cam Height',
        'PHYSICAL_MEANING': 'Unknown', 'FILETYPE': 'Grid2D',
        'DATETIME': 'Friday, July 15, 2016 14:15:38',
        'VERSION': '2', 'BINARY': True, 'HOLE_VALUE': 1e+308, 'B_AXIS_MAX': 2.7135,
        'FRED_FILENAME': 'optisches_modell.frd', 'A_AXIS_MIN': -3.392, 'A_AXIS_UNITS': 'mm',
        'A_AXIS_LABEL': 'Cam Width', 'B_AXIS_UNITS': 'mm'}
    """

    if type(f) == str:
        f = open(f, "r")
    
    # read header
    line = f.readline()
    if line.rstrip().upper() != "FRED_DATA_FILE":
        f.close()
        raise IOError
    
    info = {}
    
    while line:
        line = f.readline()
        
        if "BeginData" in line:
            break
        
        try:
            key, value = line.rstrip().split("=", 1)
        except:
            continue
        
        key = key.upper()
        if key == "FILETYPE":
            info[key] = value.strip()
        if key == "DATATYPE":
            info[key] = value.strip() 
        if key == "PHYSICAL_MEANING":
            info[key] = value.strip()
        if key == "TITLE":
            info[key] = value.strip().strip('"')
        if key == "DATAUNITS":
            info[key] = value.strip().strip('"')    
        if key == "FRED_FILENAME":
            info[key] = value.strip().strip('"')
        if key == "DATETIME":
            info[key] = value.strip().strip('"')
        if key == "ORIGIN_POSITION":
            value = value.strip().split()
            info[key] = np.array([float(value[0]),
                                  float(value[1]),
                                  float(value[2])])
        if key == "BINARY":
            info[key] = (value.strip().upper() == "TRUE")
        if key == "HOLE_VALUE":
            info[key] = float(value)
        if key == "VERSION":
            info[key] = value.strip()
    
        if key == "A_AXIS_MIN":
            info[key] = float(value)
        if key == "A_AXIS_MAX":
            info[key] = float(value)
        if key == "A_AXIS_DIM":
            info[key] = int(value)
        if key == "A_AXIS_LABEL":
            info[key] = value.strip().strip('"')
        if key == "A_AXIS_TYPE":
            info[key] = value.strip()
        if key == "A_AXIS_UNITS":
            info[key] = value.strip()
        if key == "A_AXIS_DIR":
            value = value.strip().split()
            info[key] = np.array([float(value[0]),
                                  float(value[1]),
                                  float(value[2])])       
            
        if key == "B_AXIS_MIN":
            info[key] = float(value)
        if key == "B_AXIS_MAX":
            info[key] = float(value)
        if key == "B_AXIS_DIM":
            info[key] = int(value)
        if key == "B_AXIS_LABEL":
            info[key] = value.strip().strip('"')
        if key == "B_AXIS_TYPE":
            info[key] = value.strip()
        if key == "B_AXIS_UNITS":
            info[key] = value.strip()
        if key == "B_AXIS_DIR":
            value = value.strip().split()
            info[key] = np.array([float(value[0]),
                                  float(value[1]),
                                  float(value[2])])
    
    if info["BINARY"]:
        f.seek(f.tell() + 16)  # Skip another 16 bytes in case of binary
        data = np.fromfile(f, dtype=np.float64)
    else:
        data = np.fromfile(f, dtype=np.float64, sep=" ")    
    
    data = np.reshape(data, (info["B_AXIS_DIM"], info["A_AXIS_DIM"]))
    
    f.close()
    
    if header_info:
        return data, info
    else:
        return data


def array2fgd(filename, a, header):
    """
    Write data from an array to a FRED FGD file.
    
    Parameters
    ----------
        filename : str
            Open the file object or filename.
        
        a : array_like
            Array containing the data.
            
        header : dict
            Header information preceding the actual data array in the FGD file.
            The following items are available:
                * PHYSICAL_MEANING: str, ``Unknown`` / ``Irradiance`` / etc.
                * TITLE: str, title of the dependent axis (perpendicular to A and B)
                * DATAUNITS: str, units of the dependent axis (arbitrary text)
                * FRED_FILENAME: str, name of the FRED file the data originates from
                * ORIGIN_POSITION: 3-element tuple/array with origin position
                * HOLE_VALUE: ``1e+308``
                * A_AXIS_MIN: float, negative limit of A axis
                * A_AXIS_MAX: float, positive limit of A axis
                * A_AXIS_DIM: int, number of data points in A direction
                * A_AXIS_LABEL: str, title for the A axis
                * A_AXIS_TYPE: str, ``Unknown`` / ``Spatial`` / etc.
                * A_AXIS_UNITS: str, ``Unknown`` / ``mm`` / etc.
                * A_AXIS_DIR: 3-element tuple/array with unit direction vector
                * B_AXIS_MIN: float, negative limit of B axis
                * B_AXIS_MAX: float, positive limit of B axis
                * B_AXIS_DIM: int, number of data points in B direction
                * B_AXIS_LABEL: str, title for the B axis
                * B_AXIS_TYPE: str, ``Unknown`` / ``Spatial`` / etc.
                * B_AXIS_UNITS: str, ``Unknown`` / ``mm`` / etc.
                * B_AXIS_DIR: 3-element tuple/array with unit direction vector
    
    Note
    ----
        Binary mode does not work yet. File export is always in text mode.
    """

    header["BINARY"] = False
    header["DATETIME"] = time.strftime("%A, %B %d, %Y %H:%M:%S")
    
    f = open(filename, "w")
    header_str = "FRED_DATA_FILE\n"
    for key in header:
        header_str += key + "= " + str(header[key]).strip(" []") + "\n"
    header_str += "BeginData\n"
    f.write(header_str)
    a.tofile(f, sep=" ")
    f.close()


def fgd2image(pattern, extension="png", ref_radiance=None, ref_gray=255):
    """
    Convert FRED FGD file into image file.
    
    Parameters
    ----------
        pattern : str
            File name pattern for file(s) to convert. This may include
            wildcards to match multiple files.
        
        extension : str
            Extension of the target image file. This determines the image format
            of the target file.
        
        ref_radiance : float
            Radiance value that is the common normalization reference for all
            converted files. If ``None`` the reference radiance is the maximum
            of each converted file individually (i.e. there is no common
            reference).
        
        ref_gray : int
            Gray value corresponding to the reference radiance.
    
    Returns
    -------
        count : int
            Number of the converted files.
    
    Example
    -------
        >>> from tados.fred.fgdfile import fgd2image  
        >>> count = fgd2image("Camera_image_1.fgd", ref_radiance=0.001)
        >>> print(count, "file(s) converted.")
        1 file(s) converted.
        >>> count = fgd2image("Camera_image_*.fgd", ref_radiance=0.001)
        >>> print(count, "file(s) converted.")
        5 file(s) converted.
    """
    count = 0
    common_norm = False
    if ref_radiance:
        common_norm = True
        
    for filename in glob.glob(pattern):
        data, header = fgd2array(filename, header_info=True)
        data[data < 0.0] = 0.0
        
        if not common_norm:
            ref_radiance = data.max()
        
        # normalize array to maximum gray value and clip values above 255
        data = data / ref_radiance * ref_gray
        data[data > 255] = 255
        
        # quantization to integer gray values
        data = np.round(data).astype(np.int)
        # scipy.misc.imsave() does rescales the gray value range to 0-255 so
        # we use scipy.misc.toimage().save() with cmin=0 and cmax=255 instead
        scipy.misc.toimage(data, cmin=0, cmax=255).save(filename[:-4] + "." + extension)
        
        count += 1
        
    return count
