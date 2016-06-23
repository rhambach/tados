"""
Module providing a class for optimizing optical systems in Zemax

@author: Lippmann
"""

import numpy as np
from sys import stdout
import logging

class External_Zemax_Optimizer(object):

    def __init__(self,hDDE,filename):
        """
        interface for optimizing optical systems in Zemax
    
        Parameters
        ----------
          hDDE : instance of pyOptics.zemax.dde_link.DDElinkHandler
            DDE-link handler
          filename : string
            name of the ZMX system file
        
        Note
        ----
        to enable debugging, set the log level to DEBUG 
        
        >>> import logging 
        >>> logging.basicConfig(level=logging.DEBUG);
        """
        self.hDDE=hDDE;
        self.zlink = hDDE.link;
        self.filename=filename;
        self.reset();
        # variable parameters 
        self.variables = self.getVariables()
        # merit function operands
        self.MFE_weights = self.get_MFE_weights()
        self.MFE_targets = self.get_MFE_targets()
      
    def reset(self):
        " reset system to original state"
        self.hDDE.load(self.filename);
        self.zlink.zPushLens(1);

    def getVariables(self):
        # TODO: include extra data editor
        surfaces = self.zlink.zGetNumSurf()
        variables = []
        for surf in xrange(surfaces + 1):
            for param in range(0,3)+range(4,18):     # exclude SDIA=column #3
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
    
    def get_MFE_weights(self):
        " return weight of each row in the merit function editor"
        # workaround to get number of operands in MFE: 
        # insert and remove operand at first postion        
        self.zlink.zInsertMFO(1);
        nRows=self.zlink.zDeleteMFO(1);
        # get weight of each operand
        weights = [];      
        for row in xrange(1,nRows+1):
          # set weight of comment rows to 0
          typ=self.zlink.zGetOperand(row, 1);
          if typ=='BLNK': weights.append(0); 
          else:           weights.append( self.zlink.zGetOperand(row, 9) );
        return np.asarray(weights);
    
    def get_MFE_targets(self):
        " return target for each row in the merit function editor"
        targets=[];
        for row in xrange(1,self.MFE_weights.size+1):
          targets.append( self.zlink.zGetOperand(row,8) );
        return np.asarray(targets);
        
    def get_MFE_values(self,rows=None):
        """
        return current value of each row in the merit function editor, 
        use zOptimize(-1) before to update all values of the merit function editor
                
        Parameters
        ---------
          rows : vector of integers, optional
            list of row indices within the merit-function editor (starting from 1!),
            by default, all operand values are returned
        
        Returns
        -------
          values : ndArray of floats
            list of floats
            
        See
        ---
          zlink.zOptimize(), evaluate_operands();
        """
        # interesting rows of the MFE
        if rows is None: rows = np.arange(1,self.MFE_weights.size+1);
        # get all values via the DDE-link  
        values = [];        
        for row in rows:
          values.append( self.zlink.zGetOperand(row,10) );
        # print ["%10.8f "%v for v in values];
        return np.asarray(values);
        
      
    def evaluate(self, x):
        """
        evaluate the Zemax merit function for the system state x
        
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
            self.setSystemState(x);
            #self.showSystem(x);
            val = np.array(self.zlink.zOptimize(-1))
        else:
            nParam,nStates = x.shape;
            val = np.zeros(nStates)
            for i in xrange(nStates):
                self.setSystemState(x[:, i])
                val[i] = self.zlink.zOptimize(-1)
        logging.debug("FOM: %8.5f"%val + "".join([" %12.9f,"%f for f in x]));
        return val

    def evaluate_operands(self, x):
        """
        evaluate the Zemax merit function for a given system state and return
        vector of all (weighted) operands in the merit function
        
        Parameters
        ----------
           x : ndarray of shape ``(nParam,)``
             single system state, given as ndarray of double vlues
             
        Returns
        -------
           val : vector of floats
             value of each operand in the merit function, weighted by weights
        """
        # evaluate merit function for new system state x
        MFtot = self.evaluate(x);
        # calculate difference of each operand value from target values
        bNonzero = np.nonzero(self.MFE_weights);        
        vals = self.get_MFE_values(rows=bNonzero[0]+1);
        diff = self.MFE_targets[bNonzero]-vals;
        wgth = self.MFE_weights[bNonzero];
        wgth/= np.sum(wgth);
        diff*= np.sqrt(wgth);
        assert np.allclose(np.sum(diff**2),MFtot**2,rtol=1,atol=1e-8), \
                    'total merit-function value differs from Zemax value';
        return diff;


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
        # update pupil positions, solves, and index data and return error flag
        return self.zlink.zGetUpdate();

    def showSystem(self, x):
        self.setSystemState(x)
        self.zlink.zPushLens(1)
        
    def print_status(self, x=None, out=stdout):  
        if x is None:
          # print header
          out.write("-"*80+"\n");
          out.write(" ".join(["%12s"%descr for descr in self.getVariableNames()])+"\n")
          out.write("-"*80+"\n");
        else:
          # print variable values
          out.write(" ".join(["%12.5g"%_x for _x in x])+"\n");
