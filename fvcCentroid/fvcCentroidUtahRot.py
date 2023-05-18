import glob
import os
from astropy.io import fits
import numpy
import pandas
from coordio.transforms import FVCTransformAPO
from coordio.utils import fitsTableToPandas
from coordio.defaults import calibration
from multiprocessing import Pool
from functools import partial
from dateutil import parser

polidSet = {
    "nom": [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],
    "desi": [0,1,2,3,4,5,6,9,20,27,28,29,30],
    "all": list(range(33)),
    # "z1": [0, 1, 2, 3, 4, 5, 6, 28, 29],
    # "z2": [0, 1, 2, 3, 4, 5, 6],
    # "z3": [0, 1, 2, 3, 4],
    # "z4": [0, 1, 2, 3, 4, 28, 29],
}


def dataGen(fitsFileName, sigma=0.7, boxSize=3, outPath=None):
    print("processing", fitsFileName)
    sptPath = fitsFileName.split("/")
    mjd = int(sptPath[-2])
    ff = fits.open(fitsFileName)
    IPA = ff[1].header["IPA"]
    LED = ff[1].header["LED1"]
    ROTPOS = ff[1].header["ROTPOS"]
    ALT = ff[1].header["ALT"]
    TEMP = ff[1].header["TEMPRTD2"]
    CONFIGID = ff[1].header["CONFIGID"]
    # CAPPLIED = ff[1].header["CAPPLIED"]
    dateObs = parser.parse(ff[1].header["DATE-OBS"])
    imgNum = int(fitsFileName.split("-")[-1].split(".")[0])
    if ff[6].name == "POSANGLES":
        positionerCoords = fitsTableToPandas(ff[6].data)
    elif ff[7].name == "POSANGLES":
        positionerCoords = fitsTableToPandas(ff[7].data)
    else:
        raise RuntimeError("couldn't find POSANGLES table!!!!")
    imgData = ff[1].data
    if outPath is None:
        outPath = "fvc_%i.csv"%(imgNum)

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
        # df["capplied"] = CAPPLIED

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
    df.to_csv(outPath)
    ff.close()


def getFile(imgNum, baseDir):
    zpad = ("%i"%imgNum).zfill(4)
    return baseDir + "/proc-fimg-fvc1n-%s.fits"%zpad


def doOne(fvcFile):
    outDir = "/uufs/chpc.utah.edu/common/home/u0449727/fpscommis/fvc/rot2/"
    imgNumber = fvcFile.split("-")[-1].split(".")[0]
    mjd = fvcFile.split("fcam/apo/")[-1].split("/")[0]
    outFile = outDir + "fvc-%s-%s.csv"%(mjd, imgNumber)
    # dataGen(fvcFile, outPath=outFile)

    try:
        dataGen(fvcFile, outPath=outFile)
        print(fvcFile, "worked")
        return None
    except:
        print(fvcFile, "failed")
        # import pdb; pdb.set_trace()
        return fvcFile


def doAll():
    baseDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/apo/59661"
    imgNums = list(range(26, 619))
    fileList = [getFile(x, baseDir) for x in imgNums]

    nCores = 30
    p = Pool(nCores)
    badFiles = p.map(doOne, fileList)

    for file in badFiles:
        if file is None:
            continue
        else:
            print(file)

if __name__ == "__main__":
    doAll()




