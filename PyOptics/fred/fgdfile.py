# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:51:05 2016

@author: ulip
"""

import numpy as np


def from_fgd(f, header_info=False):
    """
    Construct an array from data in a FRED FGD file.
    
    Parameters
    ----------
        f : file or str
            Open the file object or filename.
        header_info : bool
            If ``True`` returns a dict with the contents of the header of the file
            along with the actual data array.
    
    Example
    -------
        >>> import fred.fgdfile
        >>> data, header = fred.fgdfile.from_fgd("datafile.fgd", header_info=True)
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
