import numpy as np

class multimesh:
    """
    Stores multiple meshgrid objects with different properties.
    """
    def __init__(self, obj1, obj2):
        """
        __init__ : initialises a multimesh object containing multiple meshgrids with different materials
        
        Inputs:
            obj1, obj2 : meshgrid objects to be combined into a multimesh
            REQUIRE same step size.
        
        Data attributes:
            self.step : step size of the grid
            self.xpts : the coordinates along the x-axis of the points on the grid
            self.ypts : the coordinates along the y-axis of the points on the grid
            self.meshval : a zero matrix of the size of xpts*ypts, used for storing values
            solved in the differential equation
            self.data : a matrix of the same size as self.meshval to classify whether
            the point is an internal point (value = data), a fictitious point (value = -1) or an ambient point (value = -2).
            self.values : an array that stores values of k, q, and Tguess for the meshgrids with row number = 'datanum'
        """
        if obj1.xpts[0] > obj2.xpts[0]: # Select which starting point is smaller
            xstartnew = round(obj2.xpts[0], 1)
        else:
            xstartnew = round(obj1.xpts[0], 1)
        if obj1.ypts[0] > obj2.ypts[0]:
            ystartnew = round(obj2.ypts[0], 1)
        else:
            ystartnew = round(obj1.ypts[0], 1)
        if obj1.xpts[-1] >= obj2.xpts[-1]:
            xstopnew = round(obj1.xpts[-1] + obj1.step, 1)
        else:
            xstopnew = round(obj2.xpts[-1] + obj1.step, 1)
        if obj1.ypts[-1] >= obj2.ypts[-1]:
            ystopnew = round(obj1.ypts[-1] + obj1.step, 1)
        else:
            ystopnew = round(obj2.ypts[-1] + obj1.step, 1)
        if obj1.step == obj2.step:
            self.step = obj1.step
        else:
            raise ValueError("Require step size of both meshgrids to be the same.")
        self.xpts = np.arange(xstartnew, xstopnew, self.step)
        self.ypts = np.arange(ystartnew, ystopnew, self.step)
        self.data = np.zeros([ystopnew/self.step + 2, xstopnew/self.step + 2])
        # first set up every point to be ambient points, then modify using cooedinates of objects
        self.data[:, :] = -2
        xselfstart = round(obj1.xpts[0]/obj1.step)
        xselfend = round(obj1.xpts[-1]/obj1.step)
        yselfstart = round(obj1.ypts[0]/obj1.step)
        yselfend = round(obj1.ypts[-1]/obj1.step)
        xotherstart = round(obj2.xpts[0]/obj2.step)
        xotherend = round(obj2.xpts[-1]/obj2.step)
        yotherstart = round(obj2.ypts[0]/obj2.step)
        yotherend = round(obj2.ypts[-1]/obj2.step)
        # set "data" of fictitious points to -1
        self.data[yselfstart : yselfend + 3, xselfstart] = -1
        self.data[yselfstart, xselfstart : xselfend + 3] = -1
        self.data[yselfstart : yselfend + 3, xselfend + 2] = -1
        self.data[yselfend + 2, xselfstart : xselfend + 2] = -1
        self.data[yotherstart : yotherend + 3, xotherstart] = -1
        self.data[yotherstart, xotherstart : xotherend + 3] = -1
        self.data[yotherstart : yotherend + 3, xotherend + 2] = -1
        self.data[yotherend + 2, xotherstart : xotherend + 3] = -1        
        # now calculate the correct values for "data" for internal points - this will overwrite any fictitious points at the interface
        self.data[yselfstart + 1 : yselfend + 2, xselfstart + 1 : xselfend + 2] = obj1.datanum
        self.data[yotherstart + 1 : yotherend + 2, xotherstart + 1 : xotherend + 2] = obj2.datanum
        # now create an array to store k, q, and Tguess for different materials
        self.values = np.zeros([2, 7])
        self.values[obj1.datanum, 0] = obj1.k
        self.values[obj1.datanum, 1] = obj1.q
        self.values[obj1.datanum, 2] = obj1.Tguess
        self.values[obj1.datanum, 3] = round(obj1.xpts[0]/self.step)
        self.values[obj1.datanum, 4] = round(obj1.xpts[-1]/self.step)
        self.values[obj1.datanum, 5] = round(obj1.ypts[0]/self.step)
        self.values[obj1.datanum, 6] = round(obj1.ypts[-1]/self.step)
        self.values[obj2.datanum, 0] = obj2.k
        self.values[obj2.datanum, 1] = obj2.q
        self.values[obj2.datanum, 2] = obj2.Tguess
        self.values[obj2.datanum, 3] = round(obj2.xpts[0]/self.step)
        self.values[obj2.datanum, 4] = round(obj2.xpts[-1]/self.step)
        self.values[obj2.datanum, 5] = round(obj2.ypts[0]/self.step)
        self.values[obj2.datanum, 6] = round(obj2.ypts[-1]/self.step)
        
        # now set up meshgrid
        self.meshval = np.zeros_like(self.data)
        T1 = np.where(self.data == -2)
        T2 = np.where(self.data == obj1.datanum)
        T3 = np.where(self.data == obj2.datanum)
        
        self.meshval(T1) = 20. + 273.
        self.meshval(T2) = obj1.Tguess
        self.meshval(T3) = obj2.Tguess
        
        
        for i in np.arange(self.meshval.shape[0]):
            for j in np.arange(self.meshval.shape[1]):
                if self.data[i, j] == -2:
                    self.meshval[i, j] = 20. + 273. # set temperature of ambient points to Ta = 20 deg C
                elif self.data[i, j] == obj1.datanum: # internal point
                    self.meshval[i, j] = obj1.Tguess
                elif self.data[i, j] == obj2.datanum:
                    self.meshval[i, j] = obj2.Tguess
                    
    def combine(self, other):
        """
        combine - Combines a multimesh object to a third meshgrid object along the y-direction
        
        Input:
            other : meshgrid object to be combined into a multimesh
        Output:
            Updates attributes of current multimesh to include the object 'other'
        """
        if self.xpts[0] > other.xpts[0]: # Select which starting point is smaller
            xstartnew = round(other.xpts[0], 1)
        else:
            xstartnew = round(self.xpts[0], 1)
        if self.ypts[0] > other.ypts[0]:
            ystartnew = round(other.ypts[0], 1)
        else:
            ystartnew = round(self.ypts[0], 1)
        if self.xpts[-1] >= other.xpts[-1]:
            xstopnew = round(self.xpts[-1] + self.step, 1)
        else:
            xstopnew = round(other.xpts[-1] + self.step, 1)
        if self.ypts[-1] >= other.ypts[-1]:
            ystopnew = round(self.ypts[-1] + self.step, 1)
        else:
            ystopnew = round(other.ypts[-1] + self.step, 1)
        data = np.zeros([ystopnew/self.step + 2, xstopnew/self.step + 2])
        data[:, :] = -2
        meshval = np.zeros_like(data)
        # copying original data and meshval into new multimesh
        data[:self.data.shape[0], :self.data.shape[1]] = self.data[:, :].copy()
        meshval[:self.meshval.shape[0], :self.meshval.shape[1]] = self.meshval[:, :].copy()
        #meshval[:self.meshval.shape[0], round(self.xpts[0] / self.step):round(self.xpts[-1] / self.step)] = self.meshval[:, round(self.xpts[0] / self.step):round(self.xpts[-1] / self.step)].copy()
        # set the fictitious point values for the other meshgrid            
        xotherstart = round(other.xpts[0]/other.step)
        xotherend = round(other.xpts[-1]/other.step)
        yotherstart = round(other.ypts[0]/other.step)
        yotherend = round(other.ypts[-1]/other.step)
        data[yotherstart : yotherend + 3, xotherstart] = -1
        data[yotherstart, xotherstart : xotherend + 3] = -1
        data[yotherstart : yotherend + 3, xotherend + 2] = -1
        data[yotherend + 2, xotherstart : xotherend + 3] = -1
        # now input internal point values for other meshgrid - to ensure that fictitious points from self is covered
        data[yotherstart + 1 : yotherend + 2, xotherstart + 1 : xotherend + 2] = other.datanum
        # some of the interface values were covered by the new addition - need to copy again the last row
        for i in np.arange(self.values.shape[0]):
            data[self.values[i, 6]+1, self.values[i, 3] + 1:self.values[i, 4] + 2] = self.data[self.values[i, 6]+1, self.values[i, 3] + 1:self.values[i, 4] + 2].copy()
        # finally set the values for ambient temperatures    
        for i in np.arange(meshval.shape[0]):
            for j in np.arange(meshval.shape[1]):
                if data[i, j] == -1:
                    meshval[i, j] = 0.
                elif data[i, j] == -2:
                    meshval[i, j] = 20. + 273. # set temperature of ambient points to Ta = 20 deg C
                elif data[i, j] == other.datanum: # internal point
                    meshval[i, j] = other.Tguess
        self.data = data.copy()
        self.meshval = meshval.copy()
        self.xpts = np.arange(xstartnew, xstopnew, self.step)
        self.ypts = np.arange(ystartnew, ystopnew, self.step)
        # now input k, q, Tguess for the new meshgrid
        values = np.zeros([self.values.shape[0] + 1, 7])
        values[:-1, :] = self.values.copy()
        values[-1, 0] = other.k
        values[-1, 1] = other.q
        values[-1, 2] = other.Tguess
        values[-1, 3] = round(other.xpts[0]/self.step)
        values[-1, 4] = round(other.xpts[-1]/self.step)
        values[-1, 5] = round(other.ypts[0]/self.step)
        values[-1, 6] = round(other.ypts[-1]/self.step)
        self.values = values.copy()
        return self
        
    def Jacobiroll(self):
        """
        Jacobiroll - Jacobi method with pictorial operator represented by 'rolling'
        the values in the meshgrid.
        """
        deltax = 1.
        meshval = self.meshval.copy()
        count = 1
        while deltax >= 5e-6:
            rolldown = np.roll(meshval, 1, axis = 0)
            rollup = np.roll(meshval, -1, axis = 0)
            rollright = np.roll(meshval, 1, axis = 1)
            rollleft = np.roll(meshval, -1, axis = 1)
            meshvalnew = (rolldown + rollup + rollright + rollleft)/4
            meshvalnew[self.values[0, 3] + 1:self.values[0, 4] + 2, self.values[0, 5] + 1:self.values[0, 6] + 2] += self.step**2*(0.15/0.5)/4 # for microprocessor
#            for i in np.arange(self.meshval.shape[0]):
#                for j in np.arange(self.meshval.shape[1]):
#                    if self.data[i, j] == -1 or self.data[i, j] == -2:
#                        meshvalnew[i, j] = self.meshval[i, j].copy()
#            meshvalnew[:, 0] = self.meshval[:, 0]
#            meshvalnew[:, -1] = self.meshval[:, -1]
#            meshvalnew[0, :] = self.meshval[0, :]
#            meshvalnew[-1, :] = self.meshval[-1, :]
#            meshvalnew[self.values[0, 3] + 1:self.values[0, 4] + 2, self.values[0, 5] + 1:self.values[0, 6] + 2] = self.meshval[self.values[0, 3] + 1:self.values[0, 4] + 2, self.values[0, 5] + 1:self.values[0, 6] + 2]
#            meshvalnew[1:6, 86:101] = self.meshval[1:6, 86:101]
            #intpoints = np.array([])
            #intpointsnew = np.array([])
            #for i in np.arange(self.meshval.shape[0]):
            #    for j in np.arange(self.meshval.shape[1]):
            #        if self.data[i, j] >= 0:
            #            intpoints = np.append(intpoints, meshval[i, j])
            #            meshvalnew += self.step**2*(self.values[self.data[i, j], 1]/self.values[self.data[i, j], 0])/4
            #            intpointsnew = np.append(intpointsnew, meshvalnew[i, j])
            deltax = np.abs(np.linalg.norm(meshvalnew) - np.linalg.norm(meshval))/np.linalg.norm(meshval) # only take internal point temperatures and compare
            #print("Jacobi deltax =", deltax, "count =", count)            
            meshval = meshvalnew.copy()
            count += 1
        self.meshval = meshvalnew.copy() # only change the values in the meshgrid object AFTER finish iterating
        return self
        
    def updatebc(self, mode = "natural"):
        """
        updatebc - updates the meshgrid fictitious point values using values calculated
        from one successful iteration of the Jacobi method.
        
        Input:
            mode : "natural" or "forced" convection
            
        Output:
            meshvalnew : values on meshgrid with updated boundary values
        """
        meshvalnew = self.meshval.copy()
        # set values of fictitious points
        if mode == "natural":
            h1 = lambda i, j: 1.31e-6*np.cbrt(self.meshval[i, j + 1] - (20+273))
        else:
            h1 = lambda i, j: -(11.4 + 5.7*20)*1e-3
        
        for i in np.arange(self.meshval.shape[0]):
            for j in np.arange(self.meshval.shape[1]):
                if self.data[i, j] == -1:
                    meshvalarr = np.array([])
                    if i + 1 < self.meshval.shape[0] and j + 1 < self.meshval.shape[1]:
                        if self.data[i, j + 1] >= 0: # Locating internal point direction

                            meshvalarr = np.append(meshvalarr, 2*h1(i, j)*self.step*(self.meshval[i, j + 1]-(20.+273.))/self.values[self.data[i, j + 1], 0] + self.meshval[i, j + 2])
                        if self.data[i + 1, j] >= 0:
                            if mode == "natural":
                                h3 = 1.31e-6*np.cbrt(self.meshval[i + 1, j] - (20+273))
                            if mode == "forced":
                                h3 = -(11.4 + 5.7*20)*1e-3
                            meshvalarr = np.append(meshvalarr, 2*h3*self.step*(self.meshval[i + 1, j]-(20.+273.))/self.values[self.data[i + 1, j], 0] + self.meshval[i + 2, j])
                    if self.data[i, j - 1] >= 0:
                        if mode == "natural":
                            h2 = 1.31e-6*np.cbrt(self.meshval[i, j - 1] - (20+273))
                        if mode == "forced":
                            h2 = -(11.4 + 5.7*20)*1e-3
                        meshvalarr = np.append(meshvalarr, 2*h2*self.step*(self.meshval[i, j - 1]-(20.+273.))/self.values[self.data[i, j - 1], 0] + self.meshval[i, j - 2])
                    if self.data[i - 1, j] >= 0:
                        if mode == "natural":
                            h4 = 1.31e-6*np.cbrt(self.meshval[i - 1, j] - (20+273))
                        if mode == "forced":
                            h4 = -(11.4 + 5.7*20)*1e-3
                        meshvalarr = np.append(meshvalarr, 2*h4*self.step*(self.meshval[i - 1, j]-(20.+273.))/self.values[self.data[i - 1, j], 0] + self.meshval[i - 2, j])
                    if meshvalarr.size > 0:
                        meshvalnew[i, j] = np.average(meshvalarr)
                    else:
                        meshvalnew[i, j] = 20. + 273. # points at the corner, will not affect any calculations of internal point temperature
                    # set values of ambient points
                if self.data[i, j] == -2:
                    meshvalnew[i, j] = 20. + 273.
        #meshvalnew[:5, :15] = self.meshval[:5, :15]
        #meshvalnew[:5, 87:] = self.meshval[:5, 87:]
        self.meshval = meshvalnew.copy()
        return meshvalnew
        
    def iterateJacobi(self, mode = "natural"):
        """
        iterateJacobi - Iterative method to determine temperature in the meshgrid until stabilise.
        This method uses the Jacobi method to repeatedly obtain updated values
        of surface temperatures through solving the Poisson's equation until the surface
        temperatures are stabilised.
    
        Input:
            mode : "natural" or "forced" convection        
        
        Output:
            meshval : meshgrid with temperature values at each point (after stabilising)
        """
        meshval = self.updatebc(mode)
        deltaave = 1
        count = 1
        while deltaave >= 1e-5:
            self.Jacobiroll()
            meshvalnew = self.updatebc(mode)
            deltaave = np.abs(np.average(meshvalnew[self.values[0, 3] + 1:self.values[0, 4] + 2, self.values[0, 5] + 1:self.values[0, 6] + 2]) - np.average(meshval[self.values[0, 3] + 1:self.values[0, 4] + 2, self.values[0, 5] + 1:self.values[0, 6] + 2]))/np.average(meshval[self.values[0, 3] + 1:self.values[0, 4] + 2, self.values[0, 5] + 1:self.values[0, 6] + 2])
            meshval = meshvalnew.copy()
            print("iterate deltaave =", deltaave, "count =", count)
            count += 1
        self.meshval = meshvalnew.copy() # only change the values in the meshgrid object AFTER finish iterating
        return self
        