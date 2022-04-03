import pandas
import glob
from multiprocessing import Pool
import numpy
from dateutil import parser
from coordio.utils import fitsTableToPandas
from coordio.transforms import FVCTransformAPO
from astropy.io import fits
import time
import os
from functools import partial

# nuisance columns
badColSet = set(['Unnamed: 0', 'index.1', 'index', 'wokID_x', 'holeType', 'holeID', 'robotailID', 'wokID_y'])

def dropColsDF(df):
    cols = set(df.columns)
    dropMe = list(cols.intersection(badColSet))
    return df.drop(columns=dropMe)

# intermediate csv directory
CSV_DIR = "/uufs/chpc.utah.edu/common/home/u0449727/fpscommis/fvc/rot2"

polidSet = {
    "nom": [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],
    "desi": [0, 1, 2, 3, 4, 5, 6, 9, 20, 27, 28, 29, 30],
    "all": list(range(33)),
}

DO_REDUCE = True
DO_COMPILE = False
DO_RESAMP = False


def getRawFile(mjd, imgNum):
    baseDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/apo/%i"%mjd
    zpad = ("%i"%imgNum).zfill(4)
    return baseDir + "/proc-fimg-fvc1n-%s.fits"%zpad


def solveImage(ff, mjd=None, imgNum=None, sigma=0.7, boxSize=3, writecsv=False):
    # print("processing", fitsFileName)
    #
    # mjd = int(sptPath[-2])
    if hasattr(ff, "lower"):
        sptPath = ff.split("/")
        mjd = int(sptPath[-2])
        imgNum = int(ff.split("-")[-1].split(".")[0])
        ff = fits.open(ff)
    IPA = ff[1].header["IPA"]
    LED = ff[1].header["LED1"]
    ROTPOS = ff[1].header["ROTPOS"]
    ALT = ff[1].header["ALT"]
    TEMP = ff[1].header["TEMPRTD2"]
    CONFIGID = ff[1].header["CONFIGID"]
    # CAPPLIED = ff[1].header["CAPPLIED"]
    dateObs = parser.parse(ff[1].header["DATE-OBS"])
    # imgNum = int(fitsFileName.split("-")[-1].split(".")[0])
    if ff[6].name == "POSANGLES":
        positionerCoords = fitsTableToPandas(ff[6].data)
    elif ff[7].name == "POSANGLES":
        positionerCoords = fitsTableToPandas(ff[7].data)
    else:
        raise RuntimeError("couldn't find POSANGLES table!!!!", mjd, imgNum)
    imgData = ff[1].data

    useWinpos = True
    dfList = []
    for polidName, polids in polidSet.items():

        fvcT = FVCTransformAPO(
            imgData,
            positionerCoords,
            IPA,
            polids=polids
        )

        fvcT.extractCentroids(
            winposSigma=sigma,
            winposBoxSize=boxSize
        )

        fvcT.fit(
            useWinpos=useWinpos
        )

        df = fvcT.positionerTableMeas.copy()
        df["polidName"] = polidName
        df["rotpos"] = ROTPOS
        df["ipa"] = IPA
        df["alt"] = ALT
        df["imgNum"] = imgNum
        df["useWinpos"] = useWinpos
        df["wpSig"] = sigma
        df["boxSize"] = boxSize
        df["scale"] = fvcT.fullTransform.simTrans.scale
        df["xtrans"] = fvcT.fullTransform.simTrans.translation[0]
        df["ytrans"] = fvcT.fullTransform.simTrans.translation[1]
        df["fvcRot"] = numpy.degrees(fvcT.fullTransform.simTrans.rotation)
        df["fiducialRMS"] = fvcT.fiducialRMS
        df["positionerRMS"] = fvcT.positionerRMS
        df["positionerRMS_clipped"] = fvcT.positionerRMS_clipped
        df["nPositionerWarn"] = fvcT.nPositionerWarn
        df["date"] = dateObs
        df["temp"] = TEMP
        df["configid"] = CONFIGID
        df["mjd"] = mjd

        # zero out all coeffs for dataframe to work
        for polid in range(33):
            zpad = ("%i"%polid).zfill(2)
            df["ZB_%s"%zpad] = 0.0

        for polid, coeff in zip(fvcT.polids, fvcT.fullTransform.coeffs):
            # add the coeffs that were fit
            zpad = ("%i"%polid).zfill(2)
            df["ZB_%s"%zpad] = coeff

        dfList.append(df)
    df = pandas.concat(dfList)
    if writecsv:
        strImgNum = zpad = ("%i"%imgNum).zfill(4)
        df.to_csv("%s/fvc-%i-%s.csv"%(CSV_DIR, mjd, strImgNum))
    return df


if DO_REDUCE:
    tstart = time.time()
    dataDict = {
        59661: [26, 619],
        59667: [2, 793],
        59668: [1, 792],
        59669: [5, 611]
    }

    # generate the file list
    fileList = []
    for mjd, imgRange in dataDict.items():
        for imgNum in range(imgRange[0], imgRange[1]+1):
            rawFile = getRawFile(mjd, imgNum)
            if not os.path.exists(rawFile):
                print("%s doesn't exist"%rawFile)
                continue
            fileList.append(rawFile)

    _solveImage = partial(solveImage, writecsv=True)

    p = Pool(28)
    p.map(_solveImage, fileList)
    p.close()

    print("DO_REDUCE took", (time.time()-tstart)/60, "minutes")

if DO_COMPILE:
    tstart = time.time()
    files = sorted(glob.glob("%s/*.csv"%CSV_DIR))
    dfList = []
    slewNum = 0
    lastRotPos = None
    lastAlt = None
    lastMJD = None
    lastConfig = None
    for f in files:
        df = pandas.read_csv(f)
        rotpos = list(set(df.rotpos))[0]
        alt = numpy.round(list(set(df.alt))[0], 4) # round altitude to 4 decimal places
        mjd = list(set(df.mjd))[0]
        configid = list(set(df.configid))[0]

        if rotpos != lastRotPos or alt != lastAlt or mjd != lastMJD or configid != lastConfig:
            slewNum += 1
        df["slewNum"] = slewNum
        df["navg"] = 1
        df["sampNum"] = 0
        df["pixelCombine"] = 0
        df["avgType"] = "none"
        df["alt"] = alt
        df["date"] = pandas.to_datetime(df["date"])
        dfList.append(df)
        lastRotPos = rotpos
        lastAlt = alt
        lastMJD = mjd
        lastConfig = configid

    df = pandas.concat(dfList)
    df = dropColsDF(df)
    df.to_csv("raw.csv")
    print("DO_COMPILE took", (time.time()-tstart)/60, "minutes")
# df = df[df.slewNum.isin([1,2,3,4,5])]

#### boot strap averages over multiple exposures #####
### append this to the data set ####
# only average over images in the same slewNum (telescope not moved)
navgList = [2, 3, 5]
nresample = 11
replace = True  # tecnhically for bootstrap we should replace


def resampleAndAverage(slewNum):
    sn = df[df.slewNum == slewNum].copy()
    mjd = int(sn.mjd.to_numpy()[0])
    imgNums = list(set(sn.imgNum))
    dfList = []
    for navg in navgList:
        for ii in range(nresample):
            imgSamp = numpy.random.choice(imgNums, navg, replace=replace)
            for polidName in list(set(sn.polidName)):
                _df = sn[(sn.imgNum.isin(imgSamp)) & (sn.polidName==polidName)]
                # save a random date to keep the stacking happy
                date = _df.date.iloc[0]
                for avgType in ["mean", "median"]:
                    if avgType == "mean":
                        _ddf = _df.groupby(["positionerID"]).mean().reset_index()
                    else:
                        _ddf = _df.groupby(["positionerID"]).median().reset_index()
                    _ddf["avgType"] = avgType
                    _ddf["navg"] = navg
                    _ddf["sampNum"] = ii + 1
                    _ddf["polidName"] = polidName
                    _ddf["slewNum"] = slewNum
                    _ddf["date"] = date
                    _ddf["imgNum"] = -999
                    _ddf = dropColsDF(_ddf)
                    dfList.append(_ddf)

            # next average over raw images instead of fits/models
            rawFiles = [getRawFile(mjd, int(imgNum)) for imgNum in imgSamp]
            f = fits.open(rawFiles[0])
            imgShape = f[1].data.shape
            flatStack = numpy.zeros((navg, imgShape[0]*imgShape[1]), dtype=numpy.float32)

            for ii, rawFile in enumerate(rawFiles):
                _f = fits.open(rawFile)
                flatStack[ii,:] = _f[1].data.flatten()

            for avgType in ["mean", "median"]:
                if avgType == "mean":
                    newdata = numpy.mean(flatStack, axis=0).reshape(imgShape)
                else:
                    newdata = numpy.median(flatStack, axis=0).reshape(imgShape)
                f[1].data = newdata

                # note a better thing could be
                # averaging over all headers too
                # here were just using headers from
                # first randomly picked image and averaging the pixels

                _df = solveImage(f, mjd, imgNum=-999)
                _df["navg"] = navg
                _df["avgType"] = avgType
                _df["sampNum"] = ii + 1
                _df["pixelCombine"] = 1
                _df["slewNum"] = slewNum
                _df = dropColsDF(_df)
                dfList.append(_df)

    return pandas.concat(dfList)


if DO_RESAMP:
    tstart = time.time()
    slewNums = list(set(df.slewNum))
    # slewNums = list(range(1,25))

    p = Pool(25)
    dfList = p.map(resampleAndAverage, slewNums)
    p.close()

    df = pandas.concat(dfList)

    df.to_csv("resampled.csv")

    print("DO_RESAMP took", (time.time()-tstart)/60, "minutes")












