from coordio.defaults import calibration
from coordio.utils import fitsTableToPandas
from astropy.io import fits
from coordio.transforms import FVCTransformAPO
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
import os

fcCols = ["site", "holeID", "id", "xWok", "yWok", "zWok", "col", "row"]
ptCols = ["site", "holeID", "positionerID", "wokID", "alphaArmLen", "metX", "metY", "apX",
           "apY", "bossX", "bossY", "alphaOffset", "betaOffset", "dx", "dy"]

pt = calibration.positionerTable.reset_index()
wc = calibration.wokCoords.reset_index()
fc = calibration.fiducialCoords.reset_index()

jt = pt.merge(wc, on="holeID")

mjd = 59785
imgStart = 4
imgEnd = 30
baseDir = "/Volumes/futa/apo/data/fcam/%i"%mjd

# 54 first robot to get to run by disabling, maybe its flux?

disabledRobots = [608, 612, 1136, 182, 54, 1300, 565, 719]


def processImgs(
        pt=pt, wc=wc, fc=fc, mjd=mjd, imgStart=imgStart, imgEnd=imgEnd, maxFinalDist=1.5
    ):
    """
    inputs
    --------
    name : for saving files and plots
    pt : positioner table
    wc : wok coordinates
    fc : fiducial coordinates
    mjd : mjd
    imgStart : first image number
    imgEnd : last image number
    maxFinalDist : max euclidean distance for centroid matching
    """

    pDF = []


    for centType in [None, "zbplus", "zbplus2"]:
        for imgNum in range(imgStart, imgEnd + 1):
            strNum = ("%i"%imgNum).zfill(4)
            img1 = baseDir + "/proc-fimg-fvc1n-%s.fits"%(strNum)
            if not os.path.exists(img1):
                print(img1, "doesn't exist!")
                continue
            ff = fits.open(img1)

            imgData = ff[1].data
            posAngles = fitsTableToPandas(ff["POSANGLES"].data)
            IPA = ff[1].header["ROTPOS"]

            fvct = FVCTransformAPO(
                imgData,
                posAngles,
                IPA,
                plotPathPrefix=None,
                positionerTable=pt,
                wokCoords=wc,
                fiducialCoords=fc
            )

            fvct.extractCentroids()
            # increase max final distance to get matches easier
            # robots are a bit off (mostly trans/rot/scale)
            fvct.fit(centType=centType)
            ptm = fvct.positionerTableMeas.copy()
            ptm["imgNum"] = imgNum
            ptm["centType"] = centType


            pDF.append(ptm)

            # import pdb; pdb.set_trace()

            # x = ptm.xWokMeasMetrology.to_numpy()
            # y = ptm.yWokMeasMetrology.to_numpy()

            # dx = x - ptm.xWokReportMetrology.to_numpy()
            # dy = y - ptm.yWokReportMetrology.to_numpy()

            # xs = numpy.hstack((xs, x))
            # ys = numpy.hstack((ys, y))
            # dxs = numpy.hstack((dxs, dx))
            # dys = numpy.hstack((dys, dy))

            ff.close()

    pDF = pandas.concat(pDF)

    pDF.to_csv("zbplus2_pt.csv")


def plotQuiver(df):
    plt.figure(figsize=(8,8))
    x = df.xWokMeasMetrology.to_numpy()
    y = df.yWokMeasMetrology.to_numpy()
    dx = df.dx.to_numpy()
    dy = df.dy.to_numpy()

    plt.quiver(x,y,dx,dy,angles="xy",units="xy", scale=0.001)
    plt.axis("equal")


if __name__ == "__main__":

    processImgs(pt=pt, imgStart=imgStart, imgEnd=imgEnd, maxFinalDist=1.5)

    df = pandas.read_csv("zbplus2_pt.csv")
    df=df[["imgNum", "positionerID", "xWokMeasMetrology", "yWokMeasMetrology", "centType"]]

    # import pdb; pdb.set_trace()

    ctNone = df[df.centType!=df.centType]
    ctzbplus = df[df.centType=="zbplus"]
    ctzbplus2 = df[df.centType=="zbplus2"]

    d1 = ctNone.merge(ctzbplus, on=["imgNum", "positionerID"], suffixes=(None, "_x"))
    d2 = ctzbplus2.merge(ctzbplus, on=["imgNum", "positionerID"], suffixes=(None, "_x"))
    d3 = ctNone.merge(ctzbplus2, on=["imgNum", "positionerID"], suffixes=(None, "_x"))

    d1["dx"] = d1.xWokMeasMetrology - d1.xWokMeasMetrology_x
    d1["dy"] = d1.yWokMeasMetrology - d1.yWokMeasMetrology_x

    d2["dx"] = d2.xWokMeasMetrology - d2.xWokMeasMetrology_x
    d2["dy"] = d2.yWokMeasMetrology - d2.yWokMeasMetrology_x

    d3["dx"] = d3.xWokMeasMetrology - d3.xWokMeasMetrology_x
    d3["dy"] = d3.yWokMeasMetrology - d3.yWokMeasMetrology_x

    # import pdb; pdb.set_trace()

    plotQuiver(d1)
    plotQuiver(d2)
    plotQuiver(d3)
    plt.show()

    # import pdb; pdb.set_trace()
