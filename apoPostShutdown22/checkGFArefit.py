from coordio.defaults import calibration
import pandas
import numpy
import matplotlib.pyplot as plt

gc = calibration.gfaCoords.reset_index()
gc_new = pandas.read_csv("gfaCoordsAPO.csv")

gcb = gc.merge(gc_new, on="id", suffixes=(None, "_new"))
gcb["dx"] = gcb.xWok - gcb.xWok_new
gcb["dy"] = gcb.yWok - gcb.yWok_new

dAng = []
for ii, row in gcb.iterrows():
    x1 = numpy.array([row.ix, row.iy, row.iz])
    x2 = numpy.array([row.ix_new, row.iy_new, row.iz_new])
    x1 = x1/numpy.linalg.norm(x1)
    dAng.append(numpy.degrees(numpy.arccos(x1@x2)))

    # print(x1, x2, numpy.linalg.norm(x1), numpy.linalg.norm(x2))
gcb["dAng"] = dAng


dr = numpy.sqrt(gcb.dx**2+gcb.dy**2)
print(dr)
plt.quiver(gcb.xWok, gcb.yWok, gcb.dx, gcb.dy)
plt.show()
import pdb; pdb.set_trace()