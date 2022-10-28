import numpy
from astropy.io import fits
from coordio.utils import fitsTableToPandas
from coordio.transforms import FVCTransformAPO
import matplotlib.pyplot as plt
import pandas
from coordio.defaults import calibration
from coordio.utils import refinexy, simplexy
import sep
# from photutils.detection import DAOStarFinder
from coordio.transforms import arg_nearest_neighbor
import seaborn as sns
from skimage.transform import SimilarityTransform
from coordio.zhaoburge import getZhaoBurgeXY
from scipy.optimize import minimize
import os

# FVC_SCALE = 0.1108 # mm per pix
# FVC_ROT = 0.5694 # degrees
xCent = "xNudge"
yCent = "yNudge"
MJD = 59864
firstIMG = 9
lastIMG = 168

# image sequence was 5x Met, 5x Ap
# then reconofigure.  Exptime 5 sec

apExclude = []
# bossExclude = [50, 1015, 224, 231, 873, 958]  # broken met or boss fiber
# apExclude = [50, 151, 773]  # broken met or ap fiber
newRobots = [1231, 710, 920, 776, 440, 983, 1144, 56, 877, 1370, 515, 1017,
                898, 1111, 1092, 792, 1256, 204, 751, 615, 916, 478, 117, 423, 535]

pt = calibration.positionerTable.reset_index()
wc = calibration.wokCoords.reset_index()
pt = pt.merge(wc, on="holeID")
hasAp = numpy.array(["Apogee" in x for x in pt.holeType.to_numpy()])
positionerIDs = pt.positionerID.to_numpy()
positionersApogee = numpy.array(list(set(positionerIDs[hasAp])-set(apExclude)))


def getImgFile(imgNum):
    imgStr = ("%i"%imgNum).zfill(4)
    imgFile = "/data/fcam/%i/proc-fimg-fvc1n-%s.fits"%(MJD, imgStr)
    if os.path.exists(imgFile):
        isProc = True
    else:
        imgFile = "/data/fcam/%i/fimg-fvc1n-%s.fits"%(MJD, imgStr)
        isProc = False

    return imgFile, isProc


def processImg(ff, centroidOnly=False, positionerCoords=None):
    # imgNum = int(fitsFileName.split("-")[-1].split(".")[0])
    if positionerCoords is None:
        positionerCoords = fitsTableToPandas(ff["POSANGLES"].data)
    imgData = ff[1].data

    fvcT = FVCTransformAPO(
        imgData,
        positionerCoords,
        ff[1].header["IPA"],
        polids=range(33)
    )

    fvcT.extractCentroids()
    if centroidOnly:
        return fvcT.centroids.copy()

    fvcT.fit(centType="nudge")

    df = fvcT.positionerTableMeas.copy()
    # zero out all coeffs for dataframe to work
    df["rot"] = fvcT.fullTransform.simTrans.rotation
    df["scale"] = fvcT.fullTransform.simTrans.scale
    df["transx"] = fvcT.fullTransform.simTrans.translation[0]
    df["transy"] = fvcT.fullTransform.simTrans.translation[1]
    for polid in range(33):
        zpad = ("%i"%polid).zfill(2)
        df["ZB_%s"%zpad] = 0.0

    for polid, coeff in zip(fvcT.polids, fvcT.fullTransform.coeffs):
        # add the coeffs that were fit
        zpad = ("%i"%polid).zfill(2)
        df["ZB_%s"%zpad] = coeff

    return df


class AdHocTransform(object):

    def __init__(self, rotation, scale, transx, transy, polids, coeffs):
        self.simTrans = SimilarityTransform(
            translation=[transx, transy],
            scale=scale,
            rotation=rotation
        )

        self.polids = numpy.array(polids, dtype=numpy.int16)
        self.coeffs = numpy.array(coeffs)

    def apply(self, xyCCD):
        xySimTransFit = self.simTrans(xyCCD)

        dx, dy = getZhaoBurgeXY(
            self.polids,
            self.coeffs,
            xySimTransFit[:, 0],
            xySimTransFit[:, 1],
        )
        xWokFit = xySimTransFit[:, 0] + dx
        yWokFit = xySimTransFit[:, 1] + dy
        xyWokFit = numpy.array([xWokFit, yWokFit]).T
        return xyWokFit


def compileData():

    configID = 0
    imgSeq = -1
    lastLED = None
    _ptm = []
    metCounter = 0

    for imgNum in range(firstIMG, lastIMG+1):
        print("imgnum", imgNum)

        imgFile, isProc = getImgFile(imgNum)
        ff = fits.open(imgFile)
        if isProc == False:
            # non proc imgs need a flip
            ff[1].data = ff[1].data[:,::-1]
        imgSeq += 1

        # fbiMet = bool(ff[1].header["LED1"])
        fbiMet = isProc
        if fbiMet and lastLED != "met":
            print("met fiber", imgNum)
            imgSeq = 0
            configID += 1
            lastLED = "met"
            dfMetList = []
            metCounter += 1
            posAngleTable = fitsTableToPandas(ff["POSANGLES"].data)


        # fbiAp = bool(ff[1].header["LED3"])
        fbiAp = isProc == False
        if fbiAp and lastLED != "ap":
            print("ap fiber", imgNum)
            imgSeq = 0
            lastLED = "ap"
            # complie the previous 5 met
            # images into something reasonable
            dfMet = pandas.concat(dfMetList)
            _dfMetMeanAll = dfMet.median()
            _dfMetMean = dfMet.groupby("positionerID").median().reset_index()
            # generate the FVC transform to use on sci fibers...

            polids = numpy.arange(33, dtype=numpy.int16)
            coeffs = numpy.array([_dfMetMeanAll["ZB_%s"%(("%i"%x).zfill(2))] for x in polids])

            ahc = AdHocTransform(float(_dfMetMeanAll.rot), float(_dfMetMeanAll.scale), float(_dfMetMeanAll.transx), float(_dfMetMeanAll.transy), polids, coeffs)

            xyWokMet = _dfMetMean[["xWokMeasMetrology", "yWokMeasMetrology"]].to_numpy()
            posIDMet = _dfMetMean.positionerID.to_numpy()
            # fvc rot and scale hardcoded above
            # fvcRotMean = numpy.median(_dfMetMean.fvcRot)
            # fvcScaleMean = numpy.median(_dfMetMean.fvcScale)
            alphaMet = _dfMetMean.alphaMeas.to_numpy()
            betaMet = _dfMetMean.betaMeas.to_numpy()
            alphaOffMet = _dfMetMean.alphaOffset.to_numpy()
            betaOffMet = _dfMetMean.betaOffset.to_numpy()
            # print("fvcRot and scale %.4f %.4f"%(fvcRotMean, fvcScaleMean))


        if fbiMet:
            dfMetMeas = processImg(ff)
            dfMetList.append(dfMetMeas)
            ptm = dfMetMeas.copy()

            ptm["dx"] = 0
            ptm["dy"] = 0
            ptm["xWok"] = ptm.xWokMeasMetrology
            ptm["yWok"] = ptm.yWokMeasMetrology
            ptm["peak"] = ptm.peak

            ptm = ptm[["positionerID", "xWok", "yWok", "peak", "alphaMeas", "betaMeas", "alphaOffset", "betaOffset", "dx", "dy"]]

        else:
            # use last ptmMet for associations
            cnts = processImg(ff, centroidOnly=True, positionerCoords=posAngleTable)
            xySciCCD = cnts[[xCent, yCent]].to_numpy()
            xySciWok = ahc.apply(xySciCCD)
            fluxSci = cnts.peak.to_numpy()
            _pSci = []
            _xSci = []
            _ySci = []
            _peakSci = []
            _alphaSci = []
            _betaSci = []
            _alphaOffSci = []
            _betaOffSci = []
            _dxSci = []
            _dySci = []
            for _posID, _xyWokMet, _alphaMet, _betaMet, _alphaOffMet, _betaOffMet in zip(posIDMet, xyWokMet, alphaMet, betaMet, alphaOffMet, betaOffMet):
                if fbiAp and _posID not in positionersApogee:
                    # this robot doesn't have an apogee fiber:
                    continue
                dxySci = xySciWok - _xyWokMet
                amin = numpy.argmin(numpy.linalg.norm(dxySci, axis=1))
                xyFound = xySciWok[amin]
                dxyFound = dxySci[amin]
                fluxFound = fluxSci[amin]
                _pSci.append(_posID)
                _xSci.append(xyFound[0])
                _ySci.append(xyFound[1])
                _dxSci.append(dxyFound[0])
                _dySci.append(dxyFound[1])
                _peakSci.append(fluxFound)
                _alphaSci.append(_alphaMet)
                _betaSci.append(_betaMet)
                _alphaOffSci.append(_alphaOffMet)
                _betaOffSci.append(_betaOffMet)

            ptm = pandas.DataFrame({
                "positionerID": _pSci,
                "xWok": _xSci,
                "yWok": _ySci,
                "peak": _peakSci,
                "alphaMeas": _alphaSci,
                "betaMeas": _betaSci,
                "alphaOffset": _alphaOffSci,
                "betaOffset": _betaOffSci,
                "dx": _dxSci,
                "dy": _dySci,

            })


        ptm["imgNum"] = imgNum
        ptm["imgSeq"] = imgSeq
        ptm["fiberIllum"] = lastLED
        ptm["configID"] = configID
        _ptm.append(ptm)
        imgSeq += 1
        ff.close()

    ptm = pandas.concat(_ptm)
    ptm.to_csv("ptm2.csv")



def getLinearFit(xWok, yWok, dx, dy, xOff=0, yOff=0):
    _xWok = xWok - xOff
    _yWok = yWok - yOff
    _rWok = numpy.sqrt(_xWok**2 + _yWok**2)
    thetas = numpy.arctan2(_yWok, _xWok)
    dxRot = numpy.cos(thetas)*dx + numpy.sin(thetas)*dy
    # dyRot = -numpy.sin(thetas)*_dfMean.dx + numpy.cos(thetas)*_dfMean.dy
    X = numpy.ones((len(dxRot), 2))
    X[:, 1] = _rWok
    coeffs = numpy.linalg.lstsq(X, dxRot)[0]

    return coeffs



def measureOffsets():
    pt = calibration.positionerTable.reset_index()

    df = pandas.read_csv("ptm2.csv")
    df["dr"] = numpy.sqrt(df.dx**2+df.dy**2)
    df["r"] = numpy.sqrt(df.xWok**2+df.yWok**2)

    df = df[df.dr < 3] # filter out bogus detections

    # only get apogee detections
    df = df[df.fiberIllum=="ap"]

    # note which positioners are new
    df["isNew"] = df.positionerID.isin(newRobots)

    #### derotate dxy's
    totalRot = numpy.radians(df.alphaMeas+df.betaMeas+df.alphaOffset+df.betaOffset-90)
    dxBeta = numpy.cos(totalRot)*df.dx + numpy.sin(totalRot)*df.dy
    dyBeta = -numpy.sin(totalRot)*df.dx + numpy.cos(totalRot)*df.dy
    df["dxBeta"] = dxBeta
    df["dyBeta"] = dyBeta

    # filter
    df = df[df.dxBeta > 0]
    df = df[df.dyBeta < 0.6]



    plt.figure()
    sns.scatterplot(x="dxBeta", y="dyBeta", hue="isNew", s=10, data=df, alpha=0.5)
    plt.axis("equal")

    dfMean = df.groupby("positionerID").mean().reset_index()
    # get mean dxy for apogee fibers from metrology fiber
    dxBetaMed = numpy.median(dfMean.dxBeta)
    dyBetaMed = numpy.median(dfMean.dyBeta)

    plt.figure()
    sns.scatterplot(x="dxBeta", y="dyBeta", hue="isNew", s=10, data=dfMean, alpha=0.5)
    plt.axis("equal")

    ptAp = pt.loc[hasAp]

    ptAp["dxBeta"] = ptAp.apX - ptAp.metX
    ptAp["dyBeta"] = ptAp.apY - ptAp.metY

    ptApMerge = ptAp.merge(dfMean, on="positionerID", suffixes=(None, "_new"))

    ptApMerge["ddx"] = ptApMerge.dxBeta - ptApMerge.dxBeta_new
    ptApMerge["ddy"] = ptApMerge.dyBeta - ptApMerge.dyBeta_new
    ptApMerge["ddr"] = numpy.sqrt(ptApMerge.ddx**2+ptApMerge.ddy**2)

    missingApRobots = set(ptAp.positionerID) - set(dfMean.positionerID)
    print("missing Ap robots", missingApRobots)

    plt.figure()
    bins = numpy.arange(0,0.2,0.002)
    sns.histplot(x="ddr", hue="isNew", bins=bins, data=ptApMerge, element="step")

    # update apogee fiber locations for ONLY new robots
    positionerIDs = pt.positionerID.to_numpy()
    apX = []
    apY = []
    for pid in positionerIDs:
        ptRow = pt[pt.positionerID==pid]
        _apX = float(ptRow.apX)
        _apY = float(ptRow.apY)
        _metX = float(ptRow.metX)
        _metY = float(ptRow.metY)
        if pid in newRobots:
            measRow = dfMean[dfMean.positionerID==pid]
            if len(measRow) > 0:
                # apogee fiber was measured
                dxBeta = float(measRow.dxBeta)
                dyBeta = float(measRow.dyBeta)
                print(dxBeta, dyBeta)
                _apX = _metX + dxBeta
                _apY = _metY + dyBeta
            else:
                # apogee fiber wasn't measured
                # (not hooked up to gang connector)
                # give it the median value as a guess
                _apX = _metX + dxBetaMed
                _apY = _metY + dyBetaMed


        apX.append(_apX)
        apY.append(_apY)

    pt["apX"] = numpy.array(apX)
    pt["apY"] = numpy.array(apY)

    # make a model based on old robots
    # that both apogee and boss fibers
    oldRobots = pt[~pt.positionerID.isin(newRobots)]
    oldJoin = oldRobots.merge(wc, on="holeID", suffixes=(None, "_wc"))
    oldJoin = oldJoin[oldJoin.holeType=="ApogeeBoss"]
    dxAp = oldJoin.apX - oldJoin.metX
    dyAp = oldJoin.apY - oldJoin.metY
    dxBoss = oldJoin.bossX - oldJoin.metX
    dyBoss = oldJoin.bossY - oldJoin.metY
    thetaAp = numpy.arctan2(dyAp,dxAp)
    thetaBoss = numpy.arctan2(dyBoss, dxBoss)

    # predict thetaBoss from thetaAp
    X = numpy.ones((len(thetaBoss), 2))
    X[:,1] = thetaAp
    coeffs, *_ = numpy.linalg.lstsq(X,thetaBoss)

    fit = coeffs[0] + thetaAp*coeffs[1]

    rAp = numpy.sqrt(dxAp**2+dyAp**2)
    rBoss = numpy.sqrt(dxBoss**2+dyBoss**2)
    print("mean r ap", numpy.mean(rAp), numpy.std(rAp))
    print("mean r boss", numpy.mean(rBoss), numpy.std(rBoss))
    plt.figure()
    plt.plot(dxAp,dyAp,'.', color="red")
    plt.plot(dxBoss,dyBoss,'.', color="blue")
    plt.axis("equal")

    plt.figure()
    plt.plot(thetaAp, thetaBoss, '.k')
    plt.plot(thetaAp, fit, '--r')
    plt.xlabel("theta ap")
    plt.ylabel("theta boss")

    plt.figure()
    plt.plot(rAp, rBoss, '.k')
    plt.xlabel("r ap")
    plt.ylabel("r boss")

    # update boss positions for new robots
    # based on model
    bossX = []
    bossY = []
    for pid in positionerIDs:
        ptRow = pt[pt.positionerID==pid]
        _bossX = float(ptRow.bossX)
        _bossY = float(ptRow.bossY)
        _apX = float(ptRow.apX)
        _apY = float(ptRow.apY)
        _metX = float(ptRow.metX)
        _metY = float(ptRow.metY)
        _dxAp = _apX - _metX
        _dyAp = _apY - _metY
        if pid in newRobots:
            apTheta = numpy.arctan2(_dyAp, _dxAp)
            bossTheta = coeffs[0] + apTheta*coeffs[1]
            dxBoss = numpy.mean(rBoss) * numpy.cos(bossTheta)
            dyBoss = numpy.mean(rBoss) * numpy.sin(bossTheta)

            _bossX = _metX + dxBoss
            _bossY = _metY + dyBoss

        bossX.append(_bossX)
        bossY.append(_bossY)


    pt["bossX"] = numpy.array(bossX)
    pt["bossY"] = numpy.array(bossY)

    _pt = pt.copy()
    _pt["isNew"] = _pt.positionerID.isin(newRobots)
    _ptNew = _pt[_pt.isNew]
    _ptOld = _pt[~_pt.isNew]

    plt.figure()
    plt.plot(_ptOld.bossX, _ptOld.bossY, '.k', alpha=0.8)
    plt.plot(_ptOld.apX, _ptOld.apY, '.k', alpha=0.8)

    plt.plot(_ptNew.bossX, _ptNew.bossY, '.r', ms=10, alpha=0.8)
    plt.plot(_ptNew.apX, _ptNew.apY, '.r', ms=10, alpha=0.8)
    plt.axis("equal")

    pt.to_csv("positionerTable_sciUpdate.csv")

    plt.show()




if __name__ == "__main__":
    # compileData()
    measureOffsets()

