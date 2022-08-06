# this script finds the gfa fiducial locations
# this uses data from 59736 (before 2? GFA Fifs were broken)
# one fiducial R-2C14 was bumped at that point so it's ignored

from coordio.transforms import FVCTransformAPO
from coordio.defaults import calibration
from astropy.io import fits
from coordio.utils import fitsTableToPandas
import matplotlib.pyplot as plt
import numpy
import pandas
import glob
import seaborn as sns

fc = calibration.fiducialCoords.reset_index()


# GFAIDs = ["F9", "F8", "F6", "F5", "F3", "F2", "F18", "F17", "F15", "F14", "F12", "F11"]
GFAIDs = ["F3", "F2", "F18", "F17"]
badFIFs = GFAIDs
imgNums1 = [9,10,11,12]
centType = "nudge"
mjd1 = 59784
baseDir1 = "/Volumes/futa/apo/data/fcam/%i"%mjd1
mjd2 = 59763
imgNums2 = [100,101,102,103]
baseDir2 = "/Volumes/futa/apo/data/fcam/%i"%mjd2


def plotFIFErr(fvct, clip=1, title="fif err"):
    fcm = fvct.fiducialCoordsMeas.copy()
    # fcm = fcm[fcm.wokErr < clip]

    print("fiducialRMS", numpy.sqrt(numpy.mean(fcm.wokErr**2)))
    xExpect = fcm.xWok.to_numpy()
    yExpect = fcm.yWok.to_numpy()
    xMeas = fcm.xWokMeas.to_numpy()
    yMeas = fcm.yWokMeas.to_numpy()
    dx = xExpect - xMeas
    dy = yExpect - yMeas

    plt.figure()
    plt.quiver(xExpect, yExpect, dx, dy, angles="xy", units="xy")
    plt.axis("equal")
    plt.title(title)

    plt.figure()
    plt.hist(numpy.sqrt(dx**2+dy**2))
    plt.title(title)


def compile(baseDir, imgNums, title):

    gfaMeasList = []


    idx = 0
    for imgNum in imgNums:
        imgStr = ("%i"%imgNum).zfill(4)
        fileName = baseDir + "/proc-fimg-fvc1n-%s.fits"%(imgStr)
        print("processing file", fileName)

        ff = fits.open(fileName)
        rot = ff[1].header["IPA"]
        print("rot angle", rot)
        imgData = ff[1].data
        posAngles = fitsTableToPandas(ff["POSANGLES"].data)

        fvct = FVCTransformAPO(
            imgData,
            posAngles,
            rot,
            # fiducialCoords=fc[~fc.holeID.isin(badFIFs)]
        )
        try:
            fvct.extractCentroids()
            fvct.fit(centType, maxFinalDist=1.5)
        except:
            print("file", fileName, "failed")
            continue
        clip = fvct.fiducialCoordsMeas.copy()
        print("ft rms's", fvct.fullTransform.unbiasedRMS)


        plotFIFErr(fvct, title=title)

compile(baseDir2,imgNums2[:2], "pre shutdown")
compile(baseDir1,imgNums1[:2], "post shuddown")
plt.show()
