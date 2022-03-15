import glob
from astropy.io import fits
import numpy
import pandas
from coordio.transforms import FVCTransformAPO
from coordio.utils import fitsTableToPandas
from coordio.defaults import calibration
from multiprocessing import Pool
from functools import partial
from dateutil import parser


def dataGen(fitsFileName, sigma=[0.7], boxSize=[0.3], outPath=None):
    print("processing", fitsFileName)
    ff = fits.open(fitsFileName)
    IPA = ff[1].header["IPA"]
    LED = ff[1].header["LED1"]
    ROTPOS = ff[1].header["ROTPOS"]
    ALT = ff[1].header["ALT"]
    dateObs = parser.parse(ff[1].header["DATE-OBS"])
    imgNum = int(fitsFileName.split("-")[-1].split(".")[0])
    positionerCoords = fitsTableToPandas(ff[7].data)
    imgData = ff[1].data
    if outPath is None:
        outPath = "fvc_%i.csv"%(imgNum)
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
                ROTPOS
            )

            fvcT.extractCentroids(
                winposSigma=_sigma,
                winposBoxSize=_boxSize
            )

            fvcT.fit(
                useWinpos=useWinpos
            )

            df = fvcT.positionerTableMeas.copy()
            df["rotpos"] = ROTPOS
            df["alt"] = ALT
            df["imgNum"] = imgNum
            df["useWinpos"] = useWinpos
            df["wpSig"] = _sigma
            df["boxSize"] = _boxSize
            df["scale"] = fvcT.fullTransform.simTrans.scale
            df["xtrans"] = fvcT.fullTransform.simTrans.translation[0]
            df["ytrans"] = fvcT.fullTransform.simTrans.translation[0]
            df["fvcRot"] = numpy.degrees(fvcT.fullTransform.simTrans.rotation)
            df["fiducialRMS"] = fvcT.fiducialRMS
            df["positionerRMS"] = fvcT.positionerRMS
            df["positionerRMS_clipped"] = fvcT.positionerRMS_clipped
            df["nPositionerWarn"] = fvcT.nPositionerWarn
            df["date"] = dateObs

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


def doHistory():
    # get a historical picture of what the FVC/robots have been doing
    mjdStart = 59557 # dec 8 2021
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    doHistory()




