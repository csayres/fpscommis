import glob
from astropy.io import fits
import os
from pydl.pydlutils.yanny import yanny
from numpy.lib.recfunctions import drop_fields
import pandas
import datetime

CENTTYPE = "nudge"
POLIDS = list(range(33))

MJD = 59674
DESIGN = 35962

# note FVC measurement was changed on 59697 from winpos to nudge centroids
fvcDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/apo/"
confSumDir = "/uufs/chpc.utah.edu/common/home/sdss50/software/git/sdss/sdsscore/main/apo/summary_files"


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


def getConfsummaryFiles(confidList):
    confSummaryFiles = []
    confSummaryFFiles = []
    for confid in confidList:
        confidStr = ("%i"@confid).zfill(6)
        subdir = confidStr[:-2] + "XX"

        filename = "confSummary-%i.par"%confid
        fFilename = "confSummary-%i.par"%confid

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










