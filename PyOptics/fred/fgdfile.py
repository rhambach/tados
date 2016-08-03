# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:51:05 2016

@author: ulip
"""

from __future__ import print_function, division
import numpy as np


def from_fgd(f, header_info=False):
    
    if type(f) == str:
        f = open(f, "r")
    
    # read header
    line = f.readline()
    if line.rstrip().upper() != "FRED_DATA_FILE":
        f.close()
        print("File is not a FRED FGD file.")
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


if __name__ == "__main__":
    
    f = "Camera 1_integral.fgd"
    data, info = from_fgd(f, header_info=True)
    print(info)
    