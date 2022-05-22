import glob
from astropy.io import fits
import os
from pydl.pydlutils.yanny import yanny
from numpy.lib.recfunctions import drop_fields
import pandas
from datetime import datetime
from coordio.transforms import FVCTransformAPO
from coordio.utils import fitsTableToPandas
from multiprocessing import Pool

# CENTTYPE = "nudge"
# POLIDS = list(range(33))

# note FVC measurement was changed on 59697 from winpos to nudge centroids
fvcDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/apo/"
confSumDir = "/uufs/chpc.utah.edu/common/home/sdss50/software/git/sdss/sdsscore/main/apo/summary_files/"
ditherDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/sandbox/commiss/apo/"
gimgDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/gcam/apo/"
outDir = "/uufs/chpc.utah.edu/common/home/u0449727/work/confReprocess/"


def getFVCFiles(mjd, designid):
    baseDir = fvcDir + "%i" % mjd
    allImgs = glob.glob(baseDir + "/proc-fimg-fvc1n*.fits")

    imgList = []
    confids = []
    for img in allImgs:
        h = fits.getheader(img, 1)
        if h["designid"] == designid:
            imgList.append(img)
            confids.append(h["configid"])
    return confids, imgList


def getGimgFiles(mjd, designid):
    baseDir = gimgDir + "%i" % mjd
    imgList = []
    confids = []
    for img in allImgs:
        h = fits.getheader(img, 1)
        if h["designid"] == designid:
            imgList.append(img)
            confids.append(h["configid"])
    return confids, imgList


def getConfsummaryFiles(confidList):
    confSummaryFiles = []
    confSummaryFFiles = []
    for confid in confidList:
        confidStr = ("%i"%confid).zfill(6)
        subdir = confidStr[:-2] + "XX"

        filename = "confSummary-%i.par"%confid
        fFilename = "confSummaryF-%i.par"%confid

        confSummaryFile = os.path.join(confSumDir, subdir, filename)
        confSummaryFFile = os.path.join(confSumDir, subdir, fFilename)

        if os.path.exists(confSummaryFile):
            confSummaryFiles.append(confSummaryFile)
        else:
            confSummaryFiles.append(None)

        if os.path.exists(confSummaryFFile):
            confSummaryFFiles.append(confSummaryFFile)
        else:
            confSummaryFFiles.append(None)

    return confSummaryFiles, confSummaryFFiles


def getApogeeFluxes(mjd, configid):
    ditherFiles = glob.glob(ditherDir + "%i/ditherAPOGEE*.fits"%mjd)


def parseConfSummary(ff):
    print("parsing sum file", ff)
    yf = yanny(ff)
    ft = yf["FIBERMAP"]
    magArr = ft["mag"]
    ft = drop_fields(ft, 'mag')
    df = pandas.DataFrame.from_records(ft)
    df["umag"] = magArr[:, 0]
    df["gmag"] = magArr[:, 1]
    df["rmag"] = magArr[:, 2]
    df["imag"] = magArr[:, 3]
    df["zmag"] = magArr[:, 4]
    df["holeId"] = df["holeId"].str.decode("utf-8")
    df["fiberType"] = df["fiberType"].str.decode("utf-8")
    df["category"] = df["category"].str.decode("utf-8")
    df["firstcarton"] = df["firstcarton"].str.decode("utf-8")

    #add headers
    df["configuration_id"] = int(yf["configuration_id"])
    df["design_id"] = int(yf["design_id"])
    df["field_id"] = int(yf["field_id"])
    df["epoch"] = float(yf["epoch"])
    obsTime = datetime.strptime(yf["obstime"], "%a %b %d %H:%M:%S %Y")
    df["obsTime"] = obsTime
    df["MJD"] = int(yf["MJD"])
    df["observatory"] = yf["observatory"]
    df["raCen"] = float(yf["raCen"])
    df["decCen"] = float(yf["decCen"])
    df["filename"] = ff
    if "fvc_image_path" in yf.pairs():
        df["fvc_image_path"] = yf["fvc_image_path"]
    else:
        df["fvc_image_path"] = None


    if "focal_scale" in yf.pairs():
        df["focal_scale"] = float(yf["focal_scale"])
    else:
        df["focal_scale"] = 1

    # import pdb; pdb.set_trace()
    # add an easier flag for sky fibers
    isSky = []
    for c in df.firstcarton.to_numpy():
        if "sky" in c or "skies" in c:
            isSky.append(True)
        else:
            isSky.append(False)

    df["isSky"] = isSky

    # add an easier flag for science fibers
    ot = df.on_target.to_numpy(dtype=bool)
    iv = df.valid.to_numpy(dtype=bool)
    aa = df.assigned.to_numpy(dtype=bool)
    dc = df.decollided.to_numpy(dtype=bool)

    df["activeFiber"] = ot & iv & aa & ~dc

    # last check if this is an "F" file
    if "confSummaryF" in ff:
        df["fvc"] = True
    else:
        df["fvc"] = False

    # figure out if its dithered or not
    if "parent_configuration" in yf.pairs():
        df["parent_configuration"] = int(yf["parent_configuration"])
        # df["is_dithered"] = True
        df["dither_radius"] = float(yf["dither_radius"])
    else:
        df["parent_configuration"] = -999
        # df["is_dithered"] = False
        df["dither_radius"] = -999

    # check for apogee or boss assignments
    _df = df[df.fiberType == "BOSS"]
    if 1 in _df.assigned.to_numpy():
        df["bossAssigned"] = True
    else:
        df["bossAssigned"] = False

    _df = df[df.fiberType == "APOGEE"]
    if 1 in _df.assigned.to_numpy():
        df["apogeeAssigned"] = True
    else:
        df["apogeeAssigned"] = False

    return df


def reprocessFVC(fvcFile):
    ff = fits.open(fvcFile)
    imgData = ff[1].data
    IPA = ff[1].header["IPA"]


    positionerCoords = fitsTableToPandas(ff["POSANGLES"].data)

    fvcT = FVCTransformAPO(
        imgData,
        positionerCoords,
        IPA
    )
    fvcT.extractCentroids()
    fvcT.fit()
    return fvcT.positionerTableMeas


def onePositionerTable(x):
    mjd, design = x
    confids, fvcImgList = getFVCFiles(mjd, design)
    confs, confFs = getConfsummaryFiles(confids)
    print("on mjd/design", mjd, design)
    for confF in confFs:
        if confF is not None:
            yf = yanny(confF)
            try:
                apoPath = yf["fvc_image_path"]
            except:
                print(confF, "has no fvc_path")
                continue
            tailPath = apoPath.split("/")[-1]
            utahPath = fvcDir + "%i/%s"%(mjd, tailPath)
            pt = reprocessFVC(utahPath)
            baseName = confF.split("/")[-1].split(".")[0]
            csvName = outDir + baseName + "-%i-%i-positionerTable.csv"%(mjd,design)
            pt.to_csv(csvName)


def updatePositionerTables():
    mjdDesignList = [
        [59618, 35969],
        [59619, 35975],
        [59620, 35957],
        [59620, 35981],
        [59620, 35989],
        [59621, 35934],
        [59621, 35970],
        [59623, 35986],
        [59624, 35919],
        [59624, 35935],
        [59624, 35988],
        [59624, 35990],
        [59625, 35925],
        [59625, 35986],
        [59626, 35989],
        [59652, 106921],
        [59654, 106935],
        [59662, 106935],
        [59663, 35962],
        [59663, 106935],
        [59664, 35961],
        [59664, 106935],
        [59665, 35964],
        [59665, 106936],
        [59666, 106936],
        [59669, 106935],
        [59671, 35962],
        [59671, 35993],
        [59672, 35962],
        [59672, 35993],
        [59674, 35962],
        [59674, 35993],
        [59677, 35994],
        [59678, 35961],
        [59678, 35994],
        [59683, 35996],
        [59684, 36004],
        [59686, 35963],
        [59687, 36004],
        [59690, 35987],
        [59691, 35988]
    ]

    p = Pool(25)
    p.map(onePositionerTable, mjdDesignList)
    p.close()


if __name__ == "__main__":
    updatePositionerTables()





