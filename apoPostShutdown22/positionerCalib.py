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
imgEnd = 62
centType="nudge"
baseDir = "/Volumes/futa/apo/data/fcam/%i"%mjd

# 54 first robot to get to run by disabling, maybe its flux?

disabledRobots = [608, 612, 1136, 182, 54, 1300, 565, 719]


def processImgs(
        name, pt=pt, wc=wc, fc=fc, mjd=mjd, imgStart=imgStart, imgEnd=imgEnd, maxFinalDist=1.5
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
    fDF = []


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

        fvct = FVCTransformLCO(
            imgData,
            posAngles,
            IPA,
            plotPathPrefix="pdf/%s.imgNum%s"%(name, ("%i"%imgNum).zfill(3)),
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
        fcm = fvct.fiducialCoordsMeas.copy()
        fcm["rot"] = IPA

        pDF.append(ptm)
        fDF.append(fcm)

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
    fDF = pandas.concat(fDF)

    pDF.to_csv("pt_%s.csv"%name)
    fDF.to_csv("fc_%s.csv"%name)


def forwardModel(x, positionerID, alphaDeg, betaDeg):
    xBeta, la, alphaOffDeg, betaOffDeg, dx, dy = x
    yBeta = 0 # by definition for metrology fiber

    _jt = jt[jt.positionerID==positionerID]
    b = numpy.array([_jt[x] for x in ["xWok", "yWok", "zWok"]]).flatten()
    iHat = numpy.array([_jt[x] for x in ["ix", "iy", "iz"]]).flatten()
    jHat = numpy.array([_jt[x] for x in ["jx", "jy", "jz"]]).flatten()
    kHat = numpy.array([_jt[x] for x in ["kx", "ky", "kz"]]).flatten()

    xw, yw, zw = positionerToWok(
        alphaDeg, betaDeg,
        xBeta, yBeta, la,
        alphaOffDeg, betaOffDeg,
        dx, dy, b, iHat, jHat, kHat
    )

    return xw, yw


def minimizeMe(x, positionerID, alphaDeg, betaDeg, xWok, yWok):
    xw, yw = forwardModel(x, positionerID, alphaDeg, betaDeg)
    return numpy.sum((xw-xWok)**2 + (yw-yWok)**2)


def fitCalibs(positionerTableIn, positionerTableMeas):
    x0 = numpy.array([
        defaults.MET_BETA_XY[0], defaults.ALPHA_LEN,
        0, 0, 0, 0
    ])
    positionerIDs = positionerTableIn.positionerID.to_numpy()
    _xBeta = []
    _la = []
    _alphaOffDeg = []
    _betaOffDeg = []
    _dx = []
    _dy = []
    for positionerID in positionerIDs:
        if positionerID in disabledRobots:
            print("hacking disabled positioner %i, keep old values"%positionerID)
            _row = positionerTableIn[positionerTableIn.positionerID==positionerID]
            _xBeta.append(float(_row["metX"]))
            _la.append(float(_row["alphaArmLen"]))
            _alphaOffDeg.append(float(_row["alphaOffset"]))
            _betaOffDeg.append(float(_row["betaOffset"]))
            _dx.append(float(_row["dx"]))
            _dy.append(float(_row["dy"]))
            continue

        print("calibrating positioner", positionerID)
        _df = positionerTableMeas[positionerTableMeas.positionerID==positionerID]
        args = (
            positionerID,
            _df.alphaReport.to_numpy(),
            _df.betaReport.to_numpy(),
            _df.xWokMeasMetrology.to_numpy(),
            _df.yWokMeasMetrology.to_numpy()
        )
        tstart = time.time()
        out = minimize(minimizeMe, x0, args, method="Powell")
        xBeta, la, alphaOffDeg, betaOffDeg, dx, dy = out.x
        _xBeta.append(xBeta)
        _la.append(la)
        _alphaOffDeg.append(alphaOffDeg)
        _betaOffDeg.append(betaOffDeg)
        _dx.append(dx)
        _dy.append(dy)
        tend = time.time()
        print("took %.2f"%(tend-tstart))
    _xBeta = numpy.array(_xBeta)
    _la = numpy.array(_la)
    _alphaOffDeg = numpy.array(_alphaOffDeg)
    _betaOffDeg = numpy.array(_betaOffDeg)
    _dx = numpy.array(_dx)
    _dy = numpy.array(_dy)

    positionerTableOut = positionerTableIn.copy()
    positionerTableOut["metX"] = _xBeta
    positionerTableOut["alphaArmLen"] = _la
    positionerTableOut["alphaOffset"] = _alphaOffDeg
    positionerTableOut["betaOffset"] = _betaOffDeg
    positionerTableOut["dx"] = _dx
    positionerTableOut["dy"] = _dy

    return positionerTableOut

def plotDistances(pt_meas, title=""):
    pt_meas = pt_meas.copy()
    xReport = pt_meas.xWokReportMetrology
    yReport = pt_meas.yWokReportMetrology
    xMeas = pt_meas.xWokMeasMetrology
    yMeas = pt_meas.yWokMeasMetrology
    dx = xReport - xMeas
    dy = yReport - yMeas
    dr = numpy.sqrt(dx**2+dy**2)

    rms = numpy.sqrt(numpy.mean(dr**2))*1000
    rmsStr = " RMS: %.2f um"%rms
    title = title + rmsStr

    plt.figure(figsize=(13,8))
    sns.boxplot(x=pt_meas.positionerID, y=dr)


    plt.title(title)

    plt.figure(figsize=(8,8))
    plt.quiver(xReport, yReport, dx, dy ,angles="xy", units="xy", scale=.01, width=1)
    plt.title(title)



if __name__ == "__main__":


    ################## after alpha/beta home, safe calib ####################
    processImgs("stage2.2", pt=pt, imgStart=imgStart, imgEnd=imgEnd, maxFinalDist=1.5)
    df = pandas.read_csv("pt_stage2.2.csv")
    df = df[~df.wokErrWarn] # throw out likely mismatches
    plotDistances(df, title="2.2")

    positionerTableOut = fitCalibs(pt, df)
    positionerTableOut.to_csv("positionerTable.stage2.2.csv")

    processImgs("stage2.3", pt=positionerTableOut, imgStart=imgStart, imgEnd=imgEnd, maxFinalDist=0.5)

    positionerTableOut = pandas.read_csv("positionerTable.stage2.2.csv")

    df = pandas.read_csv("pt_stage2.3.csv")
    df = df[~df.wokErrWarn] # throw out likely mismatches
    plotDistances(df, title="2.3")

    ############# danger move calibration #########################




    plt.show()

