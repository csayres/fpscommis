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

# this is basically the exact same as fvcCentroidUtah.py


def dataGen(fitsFileName, sigma=[0.7], boxSize=[3], outPath=None):
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
        outPath = "fid_%i.csv"%(imgNum)
    gotOld = False

    dfList = []
    for _sigma in sigma:
        for _boxSize in boxSize:
            if _sigma == 0:
                if gotOld:
                    continue
                gotOld = True
                _boxSize = 3
                useWinpos = False
            else:
                useWinpos = True

            fvcT = FVCTransformAPO(
                imgData,
                positionerCoords,
                IPA,
                #plotPathPrefix="fig-%i-%i"%(mjd,imgNum)
            )

            fvcT.extractCentroids(
                winposSigma=_sigma,
                winposBoxSize=_boxSize
            )

            fvcT.fit(
                useWinpos=useWinpos
            )

            df = fvcT.fiducialCoordsMeas.copy()
            df["rotpos"] = ROTPOS
            df["ipa"] = IPA
            df["alt"] = ALT
            df["imgNum"] = imgNum
            df["useWinpos"] = useWinpos
            df["wpSig"] = _sigma
            df["boxSize"] = _boxSize
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

            for polid, coeff in zip(fvcT.polids, fvcT.fullTransform.coeffs):
                zpad = ("%i"%polid).zfill(2)
                df["ZB_%s"%zpad] = coeff

            dfList.append(df)

    df = pandas.concat(dfList)
    df.to_csv(outPath)
    ff.close()


def getFile(imgNum, baseDir):
    zpad = ("%i"%imgNum).zfill(4)
    return baseDir + "/proc-fimg-fvc1n-%s.fits"%zpad


def doRotation():
    # these images are the rotator scan with unmoving robots

    baseDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/apo/59611"
    imgNums = list(range(7, 496))
    boxSize = [3,5,7]
    sigma = [0, 0.5, 0.7, 0.9, 1]
    nCores = 30
    fileList = [getFile(x, baseDir) for x in imgNums]
    _dataGen = partial(dataGen, sigma=sigma, boxSize=boxSize)
    p = Pool(nCores)
    p.map(_dataGen, fileList)


def doOneHistory(fvcFile, sigma=[0, 0.7], boxSize=[3,5]):
    outDir = "/uufs/chpc.utah.edu/common/home/u0449727/fpscommis/fvc/histData/"
    imgNumber = fvcFile.split("-")[-1].split(".")[0]
    mjd = fvcFile.split("fcam/apo/")[-1].split("/")[0]
    outFile = outDir + "fid-%s-%s.csv"%(mjd, imgNumber)
    # dataGen(fvcFile, outPath=outFile)

    try:
        dataGen(fvcFile, sigma=sigma, boxSize=boxSize, outPath=outFile)
        print(fvcFile, "worked")
        return None
    except:
        print(fvcFile, "failed")
        # import pdb; pdb.set_trace()
        return fvcFile


def doHistory():
    # get a historical picture of what the FVC/robots have been doing
    baseDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/apo/"

    #mjdStart = 59557 # dec 8 2021 ap and metrology
    mjdStart = 59558 # dec 9
    mjdEnd = 59653 # dec 8 2021
    fileList = []
    for mjd in range(mjdStart, mjdEnd+1):
        mjdDir = baseDir + str(mjd)
        if not os.path.exists(mjdDir):
            continue

        fvcFiles = glob.glob(mjdDir + "/proc-fimg-fvc1n*.fits")
        fileList.extend(fvcFiles)


    # doOneHistory(fileList[0])
    # for file in fileList:
    #     doOneHistory(file)

    nCores = 30
    p = Pool(nCores)
    badFiles = p.map(doOneHistory, fileList)

    for file in badFiles:
        if file is None:
            continue
        else:
            print(file)



if __name__ == "__main__":
    doHistory()




