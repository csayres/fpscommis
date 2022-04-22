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
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.transform import SimilarityTransform

# same as centroidRotReduce2, but hack in blanton's xys.
# only works on coordio blantonxy branch, for mjd 69661
# for fvc images he processed

# nuisance columns
badColSet = set(['Unnamed: 0', 'index.1', 'index', 'wokID_x', 'holeType', 'holeID', 'robotailID', 'wokID_y'])

def dropColsDF(df):
    cols = set(df.columns)
    dropMe = list(cols.intersection(badColSet))
    return df.drop(columns=dropMe)

# intermediate csv directory
WORK_DIR = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u0449727"
CSV_DIR = WORK_DIR + "/rot5"

polidSet = {
    "nom": [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],
    # "desi": [0, 1, 2, 3, 4, 5, 6, 9, 20, 27, 28, 29, 30],
    "all": list(range(33)),
}

# rawDF = None
# resampDF = None

DO_REDUCE = True
DO_COMPILE = True
DO_FILTER = True
DO_RESAMP = True


DISABLED_ROBOTS = [54, 463, 608, 1136, 1182, 184, 1042]

# centtypeList = ["simple"]
# centSigmaList = [0.7, 1, 1.25, 1.5, 2]

centSigmaList = [("sep", 1), ("simple", 0.005), ("simple", 0.2), ("simple", 0.5), ("simple", 1)]


def getRawFile(mjd, imgNum):
    baseDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/apo/%i"%mjd
    zpad = ("%i"%imgNum).zfill(4)
    return baseDir + "/proc-fimg-fvc1n-%s.fits"%zpad


def solveImage(ff, mjd=None, imgNum=None, writecsv=False, clobber=False):
    # print("processing", fitsFileName)
    #
    # mjd = int(sptPath[-2])

    try:
        if hasattr(ff, "lower"):
            sptPath = ff.split("/")
            mjd = int(sptPath[-2])
            imgNum = int(ff.split("-")[-1].split(".")[0])
            ff = fits.open(ff)

        print("processing image", mjd, imgNum)
        strImgNum = zpad = ("%i"%imgNum).zfill(4)
        csvName = "%s/fvc-%i-%s.csv"%(CSV_DIR, mjd, strImgNum)
        if writecsv and os.path.exists(csvName) and clobber==False:
            # dont overwrite!
            print("skipping", mjd, imgNum)
            return

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

        dfList = []
        for polidName, polids in polidSet.items():
            for centtype, sigma in centSigmaList:
                fvcT = FVCTransformAPO(
                    imgData,
                    positionerCoords,
                    IPA,
                    polids=polids
                )

                fvcT.extractCentroids(
                    simpleSigma=sigma
                )

                fvcT.fit(
                    centType=centtype
                )

                df = fvcT.positionerTableMeas.copy()
                df["polidName"] = polidName
                df["rotpos"] = ROTPOS
                df["ipa"] = IPA
                df["alt"] = ALT
                df["imgNum"] = imgNum
                df["simpleSigma"] = sigma
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
                df["centtype"] = centtype

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
            df.to_csv(csvName)
        return df
    except:
        print(mjd, imgNum, "failed! skipping")


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

    # for fileName in fileList[:5]:
    #     print("processing file", fileName)
    #     _solveImage(fileName)

    # import pdb; pdb.set_trace()

    p = Pool(25)
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
        df["alt"] = alt
        df["date"] = pandas.to_datetime(df["date"])
        dfList.append(df)
        lastRotPos = rotpos
        lastAlt = alt
        lastMJD = mjd
        lastConfig = configid

    rawDF = pandas.concat(dfList)
    rawDF = dropColsDF(rawDF)
    rawDF.to_csv(CSV_DIR + "/raw.csv")
    print("DO_COMPILE took", (time.time()-tstart)/60, "minutes")


# begin filtering data
if DO_FILTER:

    t1 = time.time()
    rawDF = pandas.read_csv(CSV_DIR + "/raw.csv")
    print("loading raw file took", (time.time()-t1)/60)


    rawDF = rawDF[rawDF.rotpos > -20]

    keepCols = [
        "positionerID", "configid", "rotpos", "alt", "mjd", "date", "temp", "slewNum", "navg", "fvcRot",
        "x", "x2", "y", "y2", "xWinpos", "yWinpos", "xSimple", "ySimple", "flux", "peak", "scale", "xtrans", "ytrans",
        "xWokMeasMetrology", "yWokMeasMetrology", "fiducialRMS", "polidName", "centtype", "wokErr", "imgNum", "simpleSigma"
    ] + ["ZB_%s"%("%i"%polid).zfill(2) for polid in range(33)]

    rawDF = rawDF[keepCols]
    rawDF.to_csv(CSV_DIR + "/raw_filtered.csv")
    print("DO_FILTER took", (time.time()-t1)/60)


if DO_RESAMP:
    tstart = time.time()
    rawDF = pandas.read_csv(CSV_DIR + "/raw_filtered.csv")

    slewNums = list(set(rawDF.slewNum))

    navgList = [2, 3]


    def resampleAndAverage(slewNum):
        sn = rawDF[rawDF.slewNum == slewNum].copy()
        mjd = int(sn.mjd.to_numpy()[0])
        imgNums = sorted(list(set(sn.imgNum)))
        if len(imgNums) < 3:
            return None
        second2last = imgNums[-2]
        dfList = []
        for ii, imgNum in enumerate(imgNums[:-1]):
            img2 = imgNum + 1
            img3 = imgNum + 2
            for navg in [2,3]:
                if navg == 2:
                    imgList = [imgNum, img2]
                elif imgNum==second2last:
                    # skip this one (cant average 3!)
                    continue
                else:
                    imgList = [imgNum, img2, img3]

                _df = sn[sn.imgNum.isin(imgList)].copy()

                _ddf =  _df.groupby(["positionerID", "polidName", "slewNum", "centtype", "simpleSigma"]).mean().reset_index()
                _ddf["navg"] = navg
                _ddf["imgNum"] = imgList[0]
                _ddf = dropColsDF(_ddf)
                dfList.append(_ddf)


        return pandas.concat(dfList)

    p = Pool(25)
    dfList = p.map(resampleAndAverage, slewNums)
    dfList = [df for df in dfList if df is not None]
    p.close()

    resampDF = pandas.concat(dfList)

    resampDF.to_csv(CSV_DIR + "/averaged_filtered.csv")

    print("DO_RESAMP took", (time.time()-tstart)/60, "minutes")





















