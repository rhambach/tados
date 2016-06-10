"""
Module providing a class for optimizing optical systems in Zemax

@author: Lippmann
"""

import numpy as np

class External_Zemax_Optimizer(object):
  
    def __init__(self,hDDE,filename):
        self.hDDE=hDDE;
        self.zlink = hDDE.link;
        self.filename=filename;
        self.reset();
        # variable parameters 
        self.variables = self.getVariables()
      
    def reset(self):
        " reset system to original state"
        self.hDDE.load(self.filename);
        self.zlink.zPushLens(1);

    def getVariables(self):
        # TODO: include extra data editor
        surfaces = self.zlink.zGetNumSurf()
        variables = []
        for surf in xrange(surfaces + 1):
            for param in xrange(18):
                if self.zlink.zGetSolve(surf, param)[0] == 1:
                    variables.append((surf, param))
        return variables

    def evaluate(self, x):
        if x.ndim == 1:
            self.setSystemState(x)
            val = np.array(self.zlink.zOptimize(-1))
        else:
            val = np.zeros(x.shape[1])
            for i in xrange(x.shape[1]):
                self.setSystemState(x[:, i])
                val[i] = self.zlink.zOptimize(-1)
        return val

    def getSystemState(self):
        x = np.zeros(len(self.variables))
        for i, param in enumerate(self.variables):
            # Curvature, thickness and conic are read by zGetSurfaceData()
            if param[1] < 5:
                x[i] = self.zlink.zGetSurfaceData(param[0], param[1] + 2)
            # Parameter values must be read by zGetSurfaceParameter()
            elif param[1] < 17:
                x[i] = self.zlink.zGetSurfaceParameter(param[0], param[1] - 4)
            # Parameter 0 is addressed by solve parameter number 17
            elif param[1] == 17:
                x[i] = self.zlink.zGetSurfaceParameter(param[0], 0)
        return x

    def setSystemState(self, x):
        for i, param in enumerate(self.variables):
            # Curvature, thickness and conic are set by zSetSurfaceData()
            if param[1] < 5:
                self.zlink.zSetSurfaceData(param[0], param[1] + 2, x[i])
            # Parameter values must be set by zSetSurfaceParameter()
            elif param[1] < 17:
                self.zlink.zSetSurfaceParameter(param[0], param[1] - 4, x[i])
            # Parameter 0 is addressed by solve parameter number 17
            elif param[1] == 17:
                self.zlink.zSetSurfaceParameter(param[0], 0, x[i])

    def showSystem(self, x):
        self.setSystemState(x)
        self.zlink.zPushLens(1)
