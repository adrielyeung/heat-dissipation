import numpy as np

class meshgrid:
    """
    Generates a 2D meshgrid with equally spaced points with given dimensions
    with or without an outer layer of fictitious points.
    """
    def __init__(self, xstart, xstop, ystart, ystop, step, k, q, Tguess, data, wfict = True):
        """
        __init__ - initialises the meshgrid object
        
        Inputs:
            xstart, xstop : coordinates on the x-axis of the object
            ystart, ystop : cooridnates on the y-axis of the object
            (xstop - xstart)*(ystop -ystart) represents the 2D dimensions of the object.
            REQUIRE xstop >= xstart and ystop >= ystart.
            step : spacing between two mesh points in each dimension (same for both)
            REQUIRE step is divisible by both x and y so that the boundary points
            are added.
            k : Thermal conductivity of material (assumed constant over surface)
            q : Power density (assumed constant over surface)
            Tguess : an initial estimate of the surface temperature (assume uniform over
            surface)
            data : an integer >= 0 to denote internal points of a material.
            Should have a different integer for a different material with different q and k.
            wfict : boolean to represent whether meshval contains fictitious points
                        
        Data attributes:
            self.xpts : the coordinates along the x-axis of the points on the grid
            self.ypts : the coordinates along the y-axis of the points on the grid
            self.meshval : a zero matrix of the size of xpts*ypts, used for storing values
            solved in the differential equation
            self.data : a matrix of the same size as self.meshval to classify whether
            the point is an internal point (value = data), a fictitious point (value = 1) or an ambient point (value = 0).
            self.datanum : input 'data'
            self.step, self.k, self.q, self.Tguess, self.wfict : same as inputs
        """
        if data >= 0:
            self.xpts = np.arange(xstart, xstop, step)
            self.ypts = np.arange(ystart, ystop, step)
            if wfict == True:
                self.meshval = np.zeros([self.ypts.size + 2, self.xpts.size + 2])
                self.data = np.zeros([self.ypts.size + 2, self.xpts.size + 2])
                self.data[1:-1, 1:-1] = data
                self.data[0, :] = -1 # fictitious point
                self.data[-1, :] = -1
                self.data[:, 0] = -1
                self.data[:, -1] = -1
            else:
                self.meshval = np.zeros([self.ypts.size, self.xpts.size])
                self.data = np.zeros([self.ypts.size, self.xpts.size])
                self.data[:, :] = data
            self.datanum = data
            self.step = step
            self.k = k
            self.q = q
            self.Tguess = Tguess
            self.wfict = wfict
        else:
            raise ValueError("Input 'data' must be an integer >= 2.")
            
    def Jacobi(self):
        """
        Jacobi - Jacobi method with pictoral operator operation over all grid points
        This takes all points (excluding fictitious) and operate on them using the pictorial
        operator (form depends on the order).
        
        Output:
            meshvalnew : updated meshgrid after iterating until convergence
        """
        meshvalnew = np.zeros_like(self.meshval)
        deltax = 1.
        while deltax >= 1e-14:
            for i in np.arange(1, self.meshval.shape[0] - 1):
                for j in np.arange(1, self.meshval.shape[1] - 1):
                    meshvalnew[i, j] = 1/4*(self.meshval[i - 1, j] + self.meshval[i + 1, j]
                    + self.meshval[i, j - 1] + self.meshval[i, j + 1])
                    - self.step**2 * (-self.q/self.k)/4
            deltax = np.abs(np.linalg.norm(meshvalnew[1:-1, 1:-1]) - np.linalg.norm(self.meshval[1:-1, 1:-1]))/np.linalg.norm(self.meshval[1:-1, 1:-1]) # only take internal point temperatures and compare
            self.meshval = meshvalnew.copy()
        #meshvalnew = meshvalnew[::-1] # Flip around the vertical direction
        return meshvalnew
    
    def Jacobiroll(self):
        """
        Jacobiroll - Jacobi method with pictorial operator represented by 'rolling'
        the values in the meshgrid.
        
        Output:
            meshvalnew : updated meshgrid after iterating until convergence
        """
        deltax = 1.
        meshval = self.meshval.copy()
        while deltax >= 1e-14:
            rolldown = np.roll(meshval, 1, axis = 0)
            rolldown = rolldown[1:-1, 1:-1] # extract only the internal points
            rollup = np.roll(meshval, -1, axis = 0)
            rollup = rollup[1:-1, 1:-1]
            rollright = np.roll(meshval, 1, axis = 1)
            rollright = rollright[1:-1, 1:-1]
            rollleft = np.roll(meshval, -1, axis = 1)
            rollleft = rollleft[1:-1, 1:-1]
            meshvalnew = meshval.copy()
            meshvalnew[1:-1, 1:-1] = (rolldown + rollup + rollright + rollleft)/4
            meshvalnew[1:-1, 1:-1] += self.step**2*(self.q/self.k)/4
            deltax = np.abs(np.linalg.norm(meshvalnew[1:-1, 1:-1]) - np.linalg.norm(meshval[1:-1, 1:-1]))/np.linalg.norm(meshval[1:-1, 1:-1]) # only take internal point temperatures and compare
            meshval = meshvalnew.copy()
        self.meshval = meshvalnew.copy() # only change the values in the meshgrid object AFTER finish iterating
        return self
    
    def updatebc(self, mode = "natural"):
        """
        updatebc - updates the meshgrid fictitious point values using values calculated
        from one successful iteration of the Jacobi method.
        
        Inputs:
            mode : "natural" or "forced" convection
            
        Output:
            meshvalnew : values on meshgrid with updated boundary values
        """
        meshvalnew = self.meshval.copy()
        for i in np.arange(1, self.meshval.shape[0] - 1):
            if mode == "natural":
                h1 = 1.31e-6*np.cbrt(self.meshval[i, 1] - (20+273))
                h2 = 1.31e-6*np.cbrt(self.meshval[i, -2] - (20+273))
            if mode == "forced":
                h1 = (11.4 + 5.7*20)*1e-6
                h2 = (11.4 + 5.7*20)*1e-6
            meshvalnew[i, 0] = 2*h1*self.step*(self.meshval[i, 1]-(20.+273.))/self.k + self.meshval[i, 2]
            meshvalnew[i, -1] = 2*h2*self.step*(self.meshval[i, -2]-(20.+273.))/self.k + self.meshval[i, -3]
        for j in np.arange(1, self.meshval.shape[1] - 1):
            if mode == "natural":
                h3 = 1.31e-6*np.cbrt(self.meshval[1, j] - (20+273))
                h4 = 1.31e-6*np.cbrt(self.meshval[-2, j] - (20+273))
            if mode == "forced":
                h3 = (11.4 + 5.7*20)*1e-6
                h4 = (11.4 + 5.7*20)*1e-6
            meshvalnew[0, j] = 2*h3*self.step*(self.meshval[1, j]-(20.+273.))/self.k + self.meshval[2, j]
            meshvalnew[-1, j] = 2*h4*self.step*(self.meshval[-2, j]-(20.+273.))/self.k + self.meshval[-3, j]
        self.meshval = meshvalnew.copy()
        return meshvalnew

    def iterateJacobi(self):
        """
        iterateJacobi - Iterative method to determine temperature in the meshgrid until stabilise.
        This method uses the Jacobi method to repeatedly obtain updated values
        of surface temperatures through solving the Poisson's equation until the surface
        temperatures are stabilised.
            
        Output:
            meshval : meshgrid with temperature values at each point (after stabilising)
        """
        # Initialise the meshgrid with Tguess on all points on the microprocessor 
        # and values on boundaries calculated through temperatures of the internal points
        for i in np.arange(1, self.meshval.shape[0] - 1):
            for j in np.arange(1, self.meshval.shape[1] - 1):
                self.meshval[i, j] = self.Tguess
        meshval = self.updatebc()
        deltax = 1
        deltadeltax = 0
        count = 1
        while deltadeltax <= 0 or count == 2 or count == 3:
            self.Jacobiroll()
            meshvalnew = self.updatebc()
            deltaxnew = np.abs(np.linalg.norm(meshvalnew[1:-1, 1:-1]) - np.linalg.norm(meshval[1:-1, 1:-1]))/np.linalg.norm(meshval[1:-1, 1:-1]) # only take internal point temperatures and compare
            deltadeltax = deltaxnew - deltax
            meshval = meshvalnew.copy()
            print("deltax =", deltaxnew, "deltadeltax =", deltadeltax, "count =", count)
            count += 1
            deltax = deltaxnew.copy()
        self.meshval = meshvalnew.copy() # only change the values in the meshgrid object AFTER finish iterating
        return self
