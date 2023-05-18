import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from multiprocessing import Pool
import time
from functools import partial

df = pandas.read_csv("dxyPixels.csv")


def gauss(x, mu, sigma=75):
    return 1/(sigma*numpy.sqrt(2*numpy.pi))*numpy.exp(-0.5*((x-mu)/sigma)**2)


fit, ax = plt.subplots(1,1, figsize=(10,10))
q = ax.quiver(df.x, df.y, df.dx, df.dy, angles="xy", units="xy", width=2, scale=0.005)
ax.quiverkey(q, 0.9, 0.9, 0.5, "0.5 pix")
ax.set_xlabel("x CCD (pix)")
ax.set_ylabel("y CCD (pix)")
plt.axis("equal")

plt.figure(figsize=(10,10))
sns.scatterplot(x="x", y="y", hue="dx", s=20, data=df)
plt.axis("equal")
ax.set_xlabel("x CCD (pix)")
ax.set_ylabel("y CCD (pix)")

plt.figure(figsize=(10,10))
sns.scatterplot(x="x", y="y", hue="dy", s=20, data=df)
plt.axis("equal")
ax.set_xlabel("x CCD (pix)")
ax.set_ylabel("y CCD (pix)")


xMin = 0
yMin = 0
xMax = 8176
yMax = 6132






def doOneColumn(ix, sigma):
    _df = df[(df.x > (ix-sigma*3)) & (df.x < (ix+sigma*3))]
    dxColumn = numpy.zeros(yMax)
    dyColumn = numpy.zeros(yMax)
    for jj in range(yMin, yMax):
        _ddf = _df[(_df.y > (jj-sigma*3)) & (_df.y < (jj+sigma*3))]
        xweights = gauss(_ddf.x, ix, sigma=sigma)
        yweights = gauss(_ddf.y, jj, sigma=sigma)
        weights = xweights*yweights
        dxCorr = weights*_ddf.dx
        dyCorr = weights*_ddf.dy
        dxColumn[jj] = numpy.sum(dxCorr)
        dyColumn[jj] = numpy.sum(dyCorr)

    return dxColumn, dyColumn


# xRows = range(10)

for sigma in [10,25,50,75,100]:
    print("processing sigma", sigma)
    xRows = range(xMin,xMax)
    xCorr = numpy.zeros((yMax, xMax))
    yCorr = numpy.zeros((yMax, xMax))
    _doOne = partial(doOneColumn, sigma=sigma)
    p = Pool(10)
    resList = p.map(_doOne, xRows)
    p.close()

    for ix, (_xCorr, _yCorr) in enumerate(resList):
        xCorr[:, ix] = _xCorr
        yCorr[:, ix] = _yCorr

    with open("yCorr_%i.npy"%sigma, "wb") as f:
        numpy.save(f, yCorr)

    with open("xCorr_%i.npy"%sigma, "wb") as f:
        numpy.save(f, xCorr)

plt.figure()
plt.imshow(yCorr, origin="lower")
plt.axis("equal")

plt.figure()
plt.imshow(xCorr, origin="lower")
plt.axis("equal")

plt.show()


# print("ten columns took", (time.time()-t1)/60)

# import pdb; pdb.set_trace()


# for ii in range(2000,2100):
#     print("x", ii)
#     _df = df[(df.x > (ii-300)) & (df.x < (ii+300))]
#     for jj in range(2200, 3600):
#         _ddf = _df[(_df.y > (jj-300)) & (_df.y < (jj+300))]
#         xweights = gauss(_ddf.x, ii, sigma=50)
#         yweights = gauss(_ddf.y, jj, sigma=50)
#         weights = xweights*yweights
#         dxCorr = weights*_ddf.dx
#         dyCorr = weights*_ddf.dy
#         xCorr[jj,ii] = numpy.sum(dxCorr)
#         yCorr[jj,ii] = numpy.sum(dyCorr)

# with open("yCorr.npy", "wb") as f:
#     numpy.save(f, yCorr)

# with open("xCorr.npy", "wb") as f:
#     numpy.save(f, xCorr)

# import pdb; pdb.set_trace()





# with open("yCorr.npy", "rb") as f:
#     yCorr = numpy.load(f)

# plt.figure()
# plt.imshow(yCorr, origin="lower")

# xbins = numpy.arange(0,xMax,50)
# ybins = numpy.arange(0,yMax,50)

# plt.figure(figsize=(13,13))
# sns.histplot(df,x="x", y="y", binwidth=75, cbar=True, pmax=0.01)
# plt.axis("equal")

# plt.show()