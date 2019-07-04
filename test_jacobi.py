import numpy as np
import meshclass as msc
import multimesh as mm
import pylab as pl

# Microprocessor with dimensions 14mm x 1mm, with one point on the meshgrid for
# every 0.2 mm.

xstart_micro = 3.
xstop_micro = xstart_micro + 14.
ystart_micro = 0.
ystop_micro = ystart_micro + 1.
step = 0.2
k_micro = 0.15 # For microprocessor
q_micro = 0.5 # Heat production within microprocessor
Tguess_micro = 70. + 273. # initial guess = 70 deg C

micro = msc.meshgrid(xstart_micro, xstop_micro, ystart_micro, ystop_micro, step, k_micro, q_micro, Tguess_micro, 0)

# Ceramic case - this sits on top of the microprocessor, with dimensions 20mm x
# 2 mm.
xstart_ceram = 0.
xstop_ceram = xstart_ceram + 20.
ystart_ceram = ystop_micro
ystop_ceram = ystart_ceram + 2.
k_ceram = 0.23
q_ceram = 0.
Tguess_ceram = 27. + 273.

ceramic = msc.meshgrid(xstart_ceram, xstop_ceram, ystart_ceram, ystop_ceram, step, k_ceram, q_ceram, Tguess_ceram, 1)

# No heat sink - only microprocessor and ceramic case considered
# Combine these together into a multimesh
system = mm.multimesh(micro, ceramic)
system.iterateJacobi()
meshvalsystem = system.meshval

# for plotting purposes
meshvalsystem[:, 0] = np.nan
meshvalsystem[:, -1] = np.nan
meshvalsystem[0, :] = np.nan
meshvalsystem[-1, :] = np.nan
meshvalsystem[1:6, 1:16] = np.nan
meshvalsystem[1:6, 86:101] = np.nan

# Plot the values on a mesh plot
X, Y = np.meshgrid(system.xpts, system.ypts)
pl.figure()
nosinkplot = pl.contourf(X, Y, meshvalsystem[1:-1, 1:-1]) # take the internal points only
pl.title("Mesh plot of temperature without a heat sink")
pl.xlabel("x / mm")
pl.ylabel("y / mm")
pl.colorbar(nosinkplot)

# Heat sink with variable number of fins
numfins = 7 # Number of fins
a = 30 # Height of a fin
b = 5 # Separation between fins
# Thickness set to 1 mm

# Consider only the case when the length of the heat sink >= length of ceramic. So x = 0 at the start of the heat sink always.
# The base of the sink
xstart_sinkbase = 0.
xstop_sinkbase = (b + 1)*(numfins - 1) + 1 # excluding last fin spacing
ystart_sinkbase = ystop_ceram
ystop_sinkbase = ystart_sinkbase + 4.
k_sink = 0.248
q_sink = 0.
Tguess_sink = 27. + 273.

sinkbase = msc.meshgrid(xstart_sinkbase, xstop_sinkbase, ystart_sinkbase, ystop_sinkbase, step, k_sink, q_sink, Tguess_sink, 2)
    
shift = (xstop_sinkbase - (xstop_ceram - xstart_ceram))/2
xstart_micro += shift
xstop_micro += shift
xstart_ceram += shift
xstop_ceram += shift
micro = msc.meshgrid(xstart_micro, xstop_micro, ystart_micro, ystop_micro, step, k_micro, q_micro, Tguess_micro, 0)
ceramic = msc.meshgrid(xstart_ceram, xstop_ceram, ystart_ceram, ystop_ceram, step, k_ceram, q_ceram, Tguess_ceram, 1)
system = mm.multimesh(micro, ceramic) # first insert objects furthest from the centre of the whole system, this way the extent of x- and y-coordinates can be defined
system.combine(sinkbase)

# The fins on top of the base
for i in np.arange(numfins):
    xstart_fin = (b + 1)*i
    xstop_fin = xstart_fin + 1
    ystart_fin = ystop_sinkbase
    ystop_fin = ystart_fin + a
    fin = msc.meshgrid(xstart_fin, xstop_fin, ystart_fin, ystop_fin, step, k_sink, q_sink, Tguess_sink, 2)
    system.combine(fin)

system.iterateJacobi()
meshvalsystem = system.meshval

# for plotting purposes
for i in np.arange(meshvalsystem.shape[0]):
    for j in np.arange(meshvalsystem.shape[1]):
        if system.data[i, j] == -1 or system.data[i, j] == -2:
            meshvalsystem[i, j] = np.nan

# Plot the values on a mesh plot
X, Y = np.meshgrid(system.xpts, system.ypts)
pl.figure()
sinkplot = pl.contourf(X, Y, meshvalsystem[1:-1, 1:-1]) # take the internal points only
pl.title("Mesh plot of temperature with a heat sink, with %.0f fins, a = %.0f and b = %.0f (natural)" %(numfins, a, b))
pl.xlabel("x / mm")
pl.ylabel("y / mm")
pl.colorbar(sinkplot)

# Forced convection
system = mm.multimesh(micro, ceramic) # first insert objects furthest from the centre of the whole system, this way the extent of x- and y-coordinates can be defined
system.combine(sinkbase)

# The fins on top of the base
for i in np.arange(numfins):
    xstart_fin = (b + 1)*i
    xstop_fin = xstart_fin + 1
    ystart_fin = ystop_sinkbase
    ystop_fin = ystart_fin + a
    fin = msc.meshgrid(xstart_fin, xstop_fin, ystart_fin, ystop_fin, step, k_sink, q_sink, Tguess_sink, 2)
    system.combine(fin)

system.iterateJacobi("forced")
meshvalsystem = system.meshval

# for plotting purposes
for i in np.arange(meshvalsystem.shape[0]):
    for j in np.arange(meshvalsystem.shape[1]):
        if system.data[i, j] == -1 or system.data[i, j] == -2:
            meshvalsystem[i, j] = np.nan

# Plot the values on a mesh plot
X, Y = np.meshgrid(system.xpts, system.ypts)
pl.figure()
windplot = pl.contourf(X, Y, meshvalsystem[1:-1, 1:-1]) # take the internal points only
pl.title("Mesh plot of temperature with a heat sink, with %.0f fins, a = %.0f and b = %.0f (forced)" %(numfins, a, b))
pl.xlabel("x / mm")
pl.ylabel("y / mm")
pl.colorbar(windplot)