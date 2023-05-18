from coordio.transforms import FVCTransformAPO
from dateutil import parser
from coordio.utils import fitsTableToPandas
import os
import pandas
from astropy.io import fits
import glob
import numpy
import time
import matplotlib.pyplot as plt

imgdir = "/Users/csayres/Downloads/fvcpdf/"

imgs = glob.glob(imgdir + "*.fits")


def solveImage(ff, mjd=None, imgNum=None, writecsv=False, clobber=False):
    # print("processing", fitsFileName)
    #
    # mjd = int(sptPath[-2])

    if hasattr(ff, "lower"):
        sptPath = ff.split("/")
        mjd = -999 #int(sptPath[-2])
        imgNum = int(ff.split("-")[-1].split(".")[0])
        ffName = ff
        ff = fits.open(ff)

    print("processing image", mjd, imgNum)
    strImgNum = zpad = ("%i"%imgNum).zfill(4)
    csvName = "%sfvc-%i-%s.csv"%(imgdir, mjd, strImgNum)
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
    for sigma in [1]:
        fvcT = FVCTransformAPO(
            imgData,
            positionerCoords,
            IPA,
            plotPathPrefix=ffName + "-simple-%.2f"%sigma
            # polids=polids
        )

        t1 = time.time()
        fvcT.extractCentroids(
            simpleSigma=sigma,
        )
        print("extraction took", time.time()-t1)

        plt.figure(figsize=(10,10))
        plt.plot(fvcT.centroids.x, fvcT.centroids.y, "ok")
        plt.plot(fvcT.centroids.xNudge, fvcT.centroids.yNudge, "xr")
        plt.axis("equal")


        fvcT.fit(
            centType="simple"
        )

        print(sigma, "rms", fvcT.fiducialRMS, fvcT.positionerRMS)
        df = fvcT.positionerTableMeas.copy()
        df["rotpos"] = ROTPOS
        df["ipa"] = IPA
        df["alt"] = ALT
        df["imgNum"] = imgNum
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
        df.to_csv(csvName)
    return df

for img in imgs:
    solveImage(img, writecsv=True, clobber=True)

plt.show()

