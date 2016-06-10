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
    
    def getVariableNames(self):
        " return array of strings, describing the meaning of each varable"
        
        names = ['Curv','Thick','Glass','SemiDia','Conic'];
        names.extend(['Par%d'%i for i in xrange(1,13)]);
        names.extend(['Par0']);
        return ["S%d.%s"%(surf,names[param]) 
                  for (surf,param) in self.variables];
        
    def evaluate(self, x):
        """
        evaluate the Zemax mertit function for the system state x
        
        Parameters
        ----------
           x : ndarray of shape ``(nParam,)`` or ``(nParam,nStates)``
             single or multiple system states, given as ndarray of double values
            
        Returns
        -------
           val : scalar or vector of length ``nStates``
             merit-function value or list of values of shape ``(nStates,)``
        """
        if x.ndim == 1:
            self.setSystemState(x)
            val = np.array(self.zlink.zOptimize(-1))
        else:
            nParam,nStates = x.shape;
            val = np.zeros(nStates)
            for i in xrange(nStates):
                self.setSystemState(x[:, i])
                val[i] = self.zlink.zOptimize(-1)
        return val

    def getSystemState(self):
        x = np.zeros(len(self.variables))
        for i, (surf,param) in enumerate(self.variables):
            # Curvature, thickness and conic are read by zGetSurfaceData()
            if param < 5:
                x[i] = self.zlink.zGetSurfaceData(surf, param + 2)
            # Parameter values must be read by zGetSurfaceParameter()
            elif param < 17:
                x[i] = self.zlink.zGetSurfaceParameter(surf, param - 4)
            # Parameter 0 is addressed by solve parameter number 17
            elif param == 17:
                x[i] = self.zlink.zGetSurfaceParameter(surf, 0)
        return x

    def setSystemState(self, x):
        for i, (surf,param) in enumerate(self.variables):
            # Curvature, thickness and conic are set by zSetSurfaceData()
            if param < 5:
                self.zlink.zSetSurfaceData(surf, param + 2, x[i])
            # Parameter values must be set by zSetSurfaceParameter()
            elif param < 17:
                self.zlink.zSetSurfaceParameter(surf, param - 4, x[i])
            # Parameter 0 is addressed by solve parameter number 17
            elif param == 17:
                self.zlink.zSetSurfaceParameter(surf, 0, x[i])

    def showSystem(self, x):
        self.setSystemState(x)
        self.zlink.zPushLens(1)
        
    def print_status(self, x=None):  
        if x is None:
          # print header
          print("-"*80);
          print(" ".join(["%12s"%descr for descr in self.getVariableNames()]))
          print("-"*80);
        else:
          # print variable values
          print(" ".join(["%12.5g"%_x for _x in x]));
