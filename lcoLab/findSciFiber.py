from coordio.defaults import calibration
from coordio.utils import fitsTableToPandas
from astropy.io import fits
from coordio.transforms import FVCTransformLCO
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import numpy
from skimage.transform import EuclideanTransform
import pandas
from coordio.transforms import positionerToWok
import seaborn as sns
from scipy.optimize import minimize
import time
from coordio import defaults

mjd = 59741


# ap
imgNum = 5
imgNumStr = ("%i"%imgNum).zfill(4)
imgFile = "/Volumes/futa/osu/lco/%i/proc-fimg-fvclab-%s.fits"%(mjd,imgNumStr)
ff = fits.open(imgFile)
dfAp = fitsTableToPandas(ff["CENTROIDS"].data)

# boss
imgNum = 8
imgNumStr = ("%i"%imgNum).zfill(4)
imgFile = "/Volumes/futa/osu/lco/%i/proc-fimg-fvclab-%s.fits"%(mjd,imgNumStr)
ff = fits.open(imgFile)
dfBoss = fitsTableToPandas(ff["CENTROIDS"].data)

# met
imgNum = 2
imgNumStr = ("%i"%imgNum).zfill(4)
imgFile = "/Volumes/futa/osu/lco/%i/proc-fimg-fvclab-%s.fits"%(mjd,imgNumStr)
ff = fits.open(imgFile)
dfMet = fitsTableToPandas(ff["CENTROIDS"].data)

plt.figure()

xm = dfMet.x.to_numpy() - numpy.mean(dfMet.x)
ym = dfMet.y.to_numpy() - numpy.mean(dfMet.y)
r = numpy.sqrt(xm**2+ym**2)
plt.plot(r, dfMet.peak, '.', label="met")

xm = dfBoss.x.to_numpy() - numpy.mean(dfBoss.x)
ym = dfBoss.y.to_numpy() - numpy.mean(dfBoss.y)
r = numpy.sqrt(xm**2+ym**2)
plt.plot(r, dfBoss.peak, '.', label="boss")

xm = dfAp.x.to_numpy() - numpy.mean(dfAp.x)
ym = dfAp.y.to_numpy() - numpy.mean(dfAp.y)
r = numpy.sqrt(xm**2+ym**2)
plt.plot(r, dfAp.peak, '.', label="ap")

plt.legend()


plt.show()

import pdb; pdb.set_trace()

