import numpy
from astropy.io import fits
from coordio.utils import fitsTableToPandas
from coordio.transforms import FVCTransformLCO
import matplotlib.pyplot as plt
import pandas
from coordio.defaults import calibration
from coordio.utils import refinexy, simplexy
import sep
from photutils.detection import DAOStarFinder
from coordio.transforms import arg_nearest_neighbor
import seaborn as sns
from skimage.transform import SimilarityTransform
from coordio.zhaoburge import getZhaoBurgeXY
from scipy.optimize import minimize

# FVC_SCALE = 0.1108 # mm per pix
# FVC_ROT = 0.5694 # degrees
xCent = "xSimple"
yCent = "ySimple"

# image sequence was 5x Met, 5x Ap, then 5x Boss,
# then reconofigure.  Exptime 1 sec

bossExclude = [50, 1015, 224, 231, 873, 958]  # broken met or boss fiber
apExclude = [50, 151, 773]  # broken met or ap fiber

pt = calibration.positionerTable.reset_index()
wc = calibration.wokCoords.reset_index()
pt = pt.merge(wc, on="holeID")
hasAp = numpy.array(["Apogee" in x for x in pt.holeType.to_numpy()])
positionerIDs = pt.positionerID.to_numpy()
positionersBoss = numpy.array(list(set(positionerIDs)-set(bossExclude)))
positionersApogee = numpy.array(list(set(positionerIDs[hasAp])-set(apExclude)))


def getImgFile(imgNum):
    mjd = 59742
    imgStr = ("%i"%imgNum).zfill(4)
    imgFile = "/Volumes/futa/osu/lco/%i/proc-fimg-fvclab-%s.fits"%(mjd, imgStr)
    return imgFile


def processImg(ff, centroidOnly=False):
    # imgNum = int(fitsFileName.split("-")[-1].split(".")[0])
    positionerCoords = fitsTableToPandas(ff["POSANGLES"].data)
    imgData = ff[1].data

    fvcT = FVCTransformLCO(
        imgData,
        positionerCoords,
        135.4,
        polids=range(33)
    )

    fvcT.extractCentroids()
    if centroidOnly:
        return fvcT.centroids.copy()

    fvcT.fit(centType="simple")

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

    firstImg = 3
    lastImg = 602
    configID = 0
    imgSeq = -1
    lastLED = None
    _ptm = []
    metCounter = 0

    for imgNum in range(firstImg, lastImg+1):
        print("imgnum", imgNum)

        imgFile = getImgFile(imgNum)
        ff = fits.open(imgFile)
        imgSeq += 1

        fbiMet = bool(ff[1].header["LED1"])
        if fbiMet and lastLED != "met":
            imgSeq = 0
            configID += 1
            lastLED = "met"
            dfMetList = []
            metCounter += 1
            # if metCounter > 3:
            #     # ptm = pandas.concat(_ptm)
            #     # ptm.to_csv("ptm2.csv")

            #     break

        fbiAp = bool(ff[1].header["LED3"])
        if fbiAp and lastLED != "ap":
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

        fbiBoss = bool(ff[1].header["LED4"])
        if fbiBoss and lastLED != "boss":
            imgSeq = 0
            lastLED = "boss"

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
            cnts = processImg(ff, centroidOnly=True)
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
                if fbiBoss and _posID not in positionersBoss:
                    # this robot doesn't have a boss fiber
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

    # import pdb; pdb.set_trace()

    ptm = pandas.concat(_ptm)
    ptm.to_csv("ptm2.csv")


def minimizeMe(x, xWok, yWok, dx, dy):
    xOff, yOff = x
    coeffs = getLinearFit(xWok,yWok,dx,dy,xOff,yOff)
    return coeffs[0]**2


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


# def findComaCenter(_df):
#     xInit = numpy.array([0,0])
#     _dfMean = _df.groupby(["positionerID"]).mean().reset_index()
#     xWok = _dfMean.xWok.to_numpy()
#     yWok = _dfMean.yWok.to_numpy()
#     dx = _dfMean.dx.to_numpy()
#     dy = _dfMean.dy.to_numpy()
#     args = (xWok,yWok,dx,dy)
#     out = minimize(minimizeMe, xInit, args, method="Nelder-Mead", options=dict(disp=True))

#     xOff, yOff = out.x
#     xOff = 0
#     yOff = 0

#     coeffs = getLinearFit(xWok,yWok,dx,dy,xOff,yOff)
#     return xOff, yOff, coeffs


    # thetas = numpy.arctan2(_dfMean.yWok, _dfMean.xWok)
    # dxRot = numpy.cos(thetas)*_dfMean.dx + numpy.sin(thetas)*_dfMean.dy
    # dyRot = -numpy.sin(thetas)*_dfMean.dx + numpy.cos(thetas)*_dfMean.dy

    # X = numpy.ones((len(dxRot), 2))
    # X[:,1] = _dfMean.r.to_numpy()
    # coeffs = numpy.linalg.lstsq(X,dxRot)[0]
    # print(fi, coeffs)
    # dxFit = X @ coeffs


def measureOffsets():
    df = pandas.read_csv("ptm2.csv")
    df["dr"] = numpy.sqrt(df.dx**2+df.dy**2)
    df["r"] = numpy.sqrt(df.xWok**2+df.yWok**2)
    df = df[df.dr < 3] # filter out bogus detections

    # dfMean = df.groupby(["positionerID", "fiberIllum"]).mean().reset_index()
    # dfMean = dfMean[["positionerID", "fiberIllum", "dx", "dy"]]
    # df = df.merge(dfMean, on=["positionerID", "fiberIllum"], suffixes=(None, "_m")).reset_index()
    # df["dx"] = df.dx - df.dx_m
    # df["dy"] = df.dy - df.dy_m

    # adjust boss dxys by a linear radial fit
    # bossCoeffs = numpy.array([0.01958768, 0.99925557])

    _dfList = []

    for fi in ["boss", "ap", "met"]:
        _df = df[df.fiberIllum==fi]

        if fi=="boss":

            _dfMean = _df.groupby(["positionerID"]).mean().reset_index()
            _xWok = _dfMean.xWok.to_numpy()
            _yWok = _dfMean.yWok.to_numpy()
            _rWok = numpy.sqrt(_xWok**2 + _yWok**2)
            _dx = _dfMean.dx.to_numpy()
            _dy = _dfMean.dy.to_numpy()

            coeffs = getLinearFit(_xWok, _yWok, _dx, _dy)

            thetas = numpy.arctan2(_df.yWok, _df.xWok)
            dxRot = numpy.cos(thetas)*_df.dx + numpy.sin(thetas)*_df.dy


            X = numpy.ones((len(_df), 2))
            X[:,1] = _df.r.to_numpy()
            dxFit = X @ coeffs
            dxNew = _df.dx - numpy.cos(thetas)*dxFit
            dyNew = _df.dy - numpy.sin(thetas)*dxFit
            _df["dx"] = dxNew
            _df["dy"] = dyNew


        _dfList.append(_df)

    # plt.show()

    df = pandas.concat(_dfList)
    df["dr"] = numpy.sqrt(df.dx**2+df.dy**2)
    df["r"] = numpy.sqrt(df.xWok**2+df.yWok**2)


    dfSci = df[df.fiberIllum != "met"]

    plt.figure()
    plt.hist(dfSci[dfSci.fiberIllum=="ap"]["dr"], histtype="step", density=True, label="ap") #bins=numpy.linspace(0, 1.7, 100), label="ap")
    plt.hist(dfSci[dfSci.fiberIllum=="boss"]["dr"], histtype="step", density=True, label="boss") #bins=numpy.linspace(0, 1.7, 100), label="boss")
    plt.legend()

    plt.xlabel("dr (mm)")

    dfSciMean = dfSci.groupby(["positionerID", "configID", "fiberIllum"]).mean().reset_index()
    dfSciMean["dr"] = numpy.sqrt(dfSciMean.dx**2+dfSciMean.dy**2)

    plt.figure()
    plt.hist(dfSciMean[dfSciMean.fiberIllum=="ap"]["dr"], histtype="step", density=True, bins=numpy.linspace(0, 1.7, 100), label="apMean")
    plt.hist(dfSciMean[dfSciMean.fiberIllum=="boss"]["dr"], histtype="step", density=True, bins=numpy.linspace(0, 1.7, 100), label="boss")
    plt.legend()

    plt.xlabel("mean dr (mm)")

    dfSciMean["ddr"] = dfSciMean.dr - numpy.mean(dfSciMean.dr)
    plt.figure(figsize=(8,8))
    sns.scatterplot(x="xWok", y="yWok", hue="ddr", s=7, alpha=1, palette="vlag", data=dfSciMean[dfSciMean.fiberIllum=="boss"])
    plt.axis("equal")


    # for positionerID in [1199, 541]:
    #     _df = dfSciMean[dfSciMean.positionerID==positionerID]
    #     plt.figure(figsize=(8,8))
    #     for fi, color in zip(["boss", "ap"], ["blue", "red"]):
    #         _ddf = _df[_df.fiberIllum==fi]
    #         x,y,dx,dy = _ddf[["xWok", "yWok", "dx", "dy"]].to_numpy().T
    #         dx = dx - numpy.mean(dx)
    #         dy = dy - numpy.mean(dy)
    #         plt.quiver(x,y,dx,dy,angles="xy",color=color)
    #     plt.title("%i"%positionerID)

    _dfB = dfSciMean[dfSciMean.fiberIllum=="boss"]
    plt.figure(figsize=(8,8))
    xb,yb,dxb,dyb = _dfB[["xWok", "yWok", "dx", "dy"]].to_numpy().T
    plt.quiver(xb,yb,dxb,dyb,angles="xy", units="xy", width=0.5, scale=0.1)
    plt.axis("equal")
    plt.title("boss")


    _dfA = dfSciMean[dfSciMean.fiberIllum=="ap"]
    plt.figure(figsize=(8,8))
    xa,ya,dxa,dya = _dfA[["xWok", "yWok", "dx", "dy"]].to_numpy().T
    plt.quiver(xa,ya,dxa,dya,angles="xy", units="xy", width=0.5, scale=0.1)
    plt.axis("equal")
    plt.title("apogee")

    plt.figure(figsize=(8,8))
    plt.quiver(xa,ya,dxa,dya,angles="xy", units="xy", width=0.2, scale=0.5, color="red")
    plt.quiver(xb,yb,dxb,dyb,angles="xy", units="xy", width=0.2, scale=0.5, color="blue")
    plt.axis("equal")

    #### derotate dxy's
    totalRot = numpy.radians(df.alphaMeas+df.betaMeas+df.alphaOffset+df.betaOffset-90)
    dxBeta = numpy.cos(totalRot)*df.dx + numpy.sin(totalRot)*df.dy
    dyBeta = -numpy.sin(totalRot)*df.dx + numpy.cos(totalRot)*df.dy
    df["dxBeta"] = dxBeta
    df["dyBeta"] = dyBeta

    plt.figure()
    sns.scatterplot(x="dxBeta", y="dyBeta", hue="fiberIllum", s=10, data=df, alpha=0.5)
    plt.axis("equal")

    dfMean = df.groupby(["positionerID", "fiberIllum"]).mean().reset_index()
    plt.figure()
    sns.scatterplot(x="dxBeta", y="dyBeta", hue="fiberIllum", s=10, data=dfMean, alpha=0.5)
    plt.axis("equal")

    pt = calibration.positionerTable.reset_index()
    robotIDs = pt.positionerID.to_numpy()
    metXs = pt.metX.to_numpy()
    metYs = pt.metY.to_numpy()
    apX = pt.apX.to_numpy()
    apY = pt.apY.to_numpy()
    bossX = pt.bossX.to_numpy()
    bossY = pt.bossY.to_numpy()
    for ii, robotID in enumerate(robotIDs):
        _dfMean = dfMean[dfMean.positionerID==robotID]
        _dfBoss = _dfMean[_dfMean.fiberIllum=="boss"]
        _dfAp = _dfMean[_dfMean.fiberIllum=="ap"]

        metX = metXs[ii]
        metY = metYs[ii]

        if len(_dfBoss) != 0:
            # update boss fiber position
            bossX[ii] = metX + float(_dfBoss.dxBeta)
            bossY[ii] = metY + float(_dfBoss.dyBeta)

        if len(_dfAp) != 0:
            # update apogee fiber position
            apX[ii] = metX + float(_dfAp.dxBeta)
            apY[ii] = metY + float(_dfAp.dyBeta)

    pt["apX"] = apX
    pt["apY"] = apY
    pt["bossX"] = bossX
    pt["bossY"] = bossY

    # some apFibers (assoc w/boss or broken were not measured)
    # set them to the average location of the others

    goodMeas = pt[pt.apX > 4]

    # calcuate bossThetas
    dxBoss = goodMeas.bossX - goodMeas.metX
    dyBoss = goodMeas.bossY
    thetaBoss = numpy.arctan2(dyBoss, dxBoss)

    # calcuate apThetas
    dxAp = goodMeas.apX - goodMeas.metX
    dyAp = goodMeas.apY

    X = numpy.ones((len(dxAp), 3))
    X[:,1] = dxAp
    X[:,2] = dxAp**2

    coeffs = numpy.linalg.lstsq(X,dyAp)[0]

    dyHat = X@coeffs

    # import pdb; pdb.set_trace()

    # plt.figure()
    # sns.scatterplot(goodMeas.metX, goodMeas.apX, hue=goodMeas.apY)
    # plt.figure()
    # sns.scatterplot(goodMeas.metX, goodMeas.apY, hue=goodMeas.apX)
    # plt.figure()
    # sns.scatterplot(goodMeas.apX, goodMeas.apY, hue=goodMeas.metX)



    plt.figure()
    sns.scatterplot(dxAp, dyAp, hue=thetaBoss)
    plt.plot(dxAp,dyHat,'.r', ms=5, alpha=0.7)
    plt.axis("equal")

    plt.figure()
    plt.plot(thetaBoss, dxAp, '.k')

    X2 = numpy.ones((len(dyHat), 2))
    X2[:,1] = thetaBoss

    coeffs2 = numpy.linalg.lstsq(X2, dxAp)[0]

    dxHat = X2@coeffs2
    plt.plot(thetaBoss, dxHat, '.r')

    # next estimate full model based on theta
    X[:,1] = dxHat
    X[:,2] = dxHat**2
    dyHat = X@coeffs

    plt.figure()
    plt.plot(dxAp, dyAp, '.k')
    plt.plot(dxHat, dyHat, '.r')


    # now estimate apXY positions based on boss theta, metX
    badMeas = pt[pt.apX <= 4]
    dxBoss = badMeas.bossX - badMeas.metX
    dyBoss = badMeas.bossY
    thetaBoss = numpy.arctan2(dyBoss, dxBoss)

    # estimateX
    X = numpy.ones((len(thetaBoss),3))
    X2 = numpy.ones((len(thetaBoss),2))
    X2[:,1] = thetaBoss
    dxEst = X2@coeffs2
    # estimateY
    X[:,1] = dxEst
    X[:,2] = dxEst**2
    dyEst = X@coeffs

    plt.figure()
    plt.plot(dxEst,dyEst,'.k')

    xApEst = numpy.array(badMeas.metX + dxEst)
    yApEst = dyEst

    plt.figure()
    plt.plot(badMeas.metX, badMeas.metY, '.k')
    plt.plot(xApEst, yApEst, '.r')
    plt.plot(badMeas.bossX, badMeas.bossY, '.b')

    # overwrite ap fiber measurements for those
    # without ccd measurements (dark apogee fibers)
    # based on fits above
    badPosIDs = badMeas.positionerID.to_numpy()
    xAp = pt.apX.to_numpy()
    yAp = pt.apY.to_numpy()
    allPosIDs = pt.positionerID.to_numpy()
    for ii, pid in enumerate(allPosIDs):
        if pid in badPosIDs:
            jj = numpy.argwhere(badPosIDs==pid)[0][0]
            print(jj)
            xAp[ii] = xApEst[jj]
            yAp[ii] = yApEst[jj]




    # overwrite here
    pt["apX"] = xAp
    pt["apY"] = yAp



    pt.to_csv("positionerTable.sciFiberMeas.csv")


    # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        # metX = float(dfMean)

    # theta1 = numpy.arctan2(y,x)
    # theta2 = numpy.arctan2(y+dy, x+dx)
    # r1 = numpy.sqrt(x**2+y**2)
    # r2 = numpy.sqrt((x+dx)**2+(y+dy)**2)

    # plt.figure()
    # plt.plot(r1,r2, '.k')
    # plt.title("boss")
    # plt.axis("equal")
    # X = numpy.ones((len(r1), 2))
    # X[:,1] = r1
    # coeffs = numpy.linalg.lstsq(X,r2)[0]
    # print("boss coeffs", coeffs)
    # rFit = X @ coeffs
    # resid = r2-rFit
    # plt.figure()
    # plt.plot(r1,resid, '.k')
    # plt.title("boss")


    # # import pdb; pdb.set_trace()




    # theta1 = numpy.arctan2(y,x)
    # theta2 = numpy.arctan2(y+dy, x+dx)
    # r1 = numpy.sqrt(x**2+y**2)
    # r2 = numpy.sqrt((x+dx)**2+(y+dy)**2)

    # plt.figure()
    # plt.plot(r1,r2, '.k')
    # plt.axis("equal")
    # plt.title("apogee")
    # X = numpy.ones((len(r1), 2))
    # X[:,1] = r1
    # coeffs = numpy.linalg.lstsq(X,r2)[0]
    # print("ap coeffs", coeffs)
    # rFit = X @ coeffs
    # resid = r2-rFit
    # plt.figure()
    # plt.plot(r1,resid, '.k')
    # plt.title("apogee")

    #     # sns.scatterplot(x="xWok", y="yWok", hue="ddr", s=35, alpha=1, palette="vlag", data=dfSciMean[(dfSciMean.fiberIllum=="boss") & (dfSciMean.positionerID==positionerID)])
    #     # plt.axis("equal")


    # _df = df[df.rWok>300]
    # print(set(_df.positionerID))

    # plt.show()
    # plt.figure(figsize=(8,8))
    # sns.scatterplot(x="dx", y="dy", hue="fiberIllum", data=dfSci)
    # plt.axis("equal")
    # plt.show()
    # # look at dx,dy within sets of 5 images for met, ap, and boss
    # # first just look at the scatter in repeated centroid measruement
    # imgSetMean = df.groupby(["positionerID", "configID", "fiberIllum"]).mean().reset_index()
    # imSetStd = df.groupby(["positionerID", "configID", "fiberIllum"]).std().reset_index()



    # import pdb; pdb.set_trace()
    plt.show()



if __name__ == "__main__":
    # compileData()
    measureOffsets()
    # updatePositionerTables()

