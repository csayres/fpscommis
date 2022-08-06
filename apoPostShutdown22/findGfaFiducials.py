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
imgNums = [9,10,11,12]
centType = "nudge"
mjd = 59784
baseDir = "/Volumes/futa/apo/data/fcam/%i"%mjd



def plotFIFErr(fvct, clip=1):
    fcm = fvct.fiducialCoordsMeas.copy()
    fcm = fcm[fcm.wokErr < clip]

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

    plt.figure()
    plt.hist(numpy.sqrt(dx**2+dy**2))


def compile():

    gfaMeasList = []

    fileNames = glob.glob(baseDir + "/proc*.fits")
    idx = 0
    # for imgNum in imgNums:
    for fileName in fileNames[:10]:
        # imgStr = ("%i"%imgNum).zfill(4)
        # fileName = baseDir + "/proc-fimg-fvc1n-%s.fits"%(imgStr)
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
        )
        try:
            fvct.extractCentroids()
            fvct.fit(centType, maxFinalDist=1.5)
        except:
            print("file", fileName, "failed")
            continue
        clip = fvct.fiducialCoordsMeas.copy()
        print("ft rms's", fvct.fullTransform.unbiasedRMS)

        gfaFound = clip[clip.holeID.isin(GFAIDs)]

        assert len(gfaFound) == 4

        # plotFIFErr(fvct)
        # plt.show()

        # run again this time removing fiducials with known errors

        fvct = FVCTransformAPO(
            imgData,
            posAngles,
            rot,
            fiducialCoords=fc[~fc.holeID.isin(badFIFs)]
        )
        try:
            fvct.extractCentroids()
            fvct.fit(centType)
        except:
            print("failed", fileName)
            continue
        clip = fvct.fiducialCoordsMeas.copy()
        print("good ft rms's", fvct.fullTransform.unbiasedRMS)
        if fvct.fullTransform.unbiasedRMS > 0.11:
            print("skipping due to rms value")
            continue

        # now transform GFA pixels into xyWok mms

        xyWok = fvct.fullTransform.apply(gfaFound[["xNudge", "yNudge"]].to_numpy())
        gfaFound["xWokMeas"] = xyWok[:,0]
        gfaFound["yWokMeas"] = xyWok[:,1]
        gfaFound["idx"] = idx
        gfaMeasList.append(gfaFound)
        idx += 1

    gfasFIFs = pandas.concat(gfaMeasList)
    gfasFIFs.to_csv("gfaAll.csv")
    gfaFIFs_m = gfasFIFs.groupby("holeID").mean().reset_index()
    gfaFIFs_s = gfasFIFs.groupby("holeID").std().reset_index()

    gfaFIFs_m.to_csv("meanGFAFIF.csv")
    gfaFIFs_s.to_csv("stdGFAFIF.csv")


def update():
    df = pandas.read_csv("gfaAll.csv")
    print("set gfas", set(df.holeID))
    df_mean = df.groupby("holeID").mean().reset_index()
    df = df.merge(df_mean, on="holeID", suffixes=(None, "_m")).reset_index()

    df["dx"] = df.xWokMeas - df.xWokMeas_m
    df["dy"] = df.yWokMeas - df.yWokMeas_m

    plt.figure()
    sns.scatterplot(x="dx", y="dy", hue="holeID", data=df)
    plt.axis("equal")
    plt.show()

    xWok = fc.xWok.to_numpy()
    yWok = fc.yWok.to_numpy()
    holeIDs = fc.holeID.to_numpy()

    for ii, holeID in enumerate(holeIDs):
        _df = df_mean[df_mean.holeID==holeID]
        if len(_df) == 0:
            print("skipping holeID", holeID)
            continue
        xw = float(_df.xWokMeas)
        yw = float(_df.yWokMeas)
        moveDist = numpy.sqrt((xw-xWok[ii])**2+(yw-yWok[ii])**2)
        print("move dist", moveDist)
        xWok[ii] = xw
        yWok[ii] = yw
    fc["xWok"] = xWok
    fc["yWok"] = yWok

    fc.to_csv("fiducialCoords.phase1.csv")


def test():
    fc1 = pandas.read_csv("fiducialCoords.phase1.csv")

    for imgNum in imgNums:
        imgStr = ("%i"%imgNum).zfill(4)
        fileName = baseDir + "/proc-fimg-fvc1n-%s.fits"%(imgStr)
        ff = fits.open(fileName)
        rot = ff[1].header["IPA"]
        imgData = ff[1].data
        posAngles = fitsTableToPandas(ff["POSANGLES"].data)


        fvct = FVCTransformAPO(
            imgData,
            posAngles,
            rot,
            fiducialCoords=fc1
        )
        fvct.extractCentroids()
        fvct.fit(centType)
        print("rms", fvct.fullTransform.unbiasedRMS)
        plotFIFErr(fvct)
    plt.show()

    # import pdb; pdb.set_trace()

# compile()
# update()
test()
