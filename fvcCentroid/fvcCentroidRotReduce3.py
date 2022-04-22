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
CSV_DIR = WORK_DIR + "/rot3"

polidSet = {
    "nom": [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],
    # "desi": [0, 1, 2, 3, 4, 5, 6, 9, 20, 27, 28, 29, 30],
    "all": list(range(33)),
}

# rawDF = None
# resampDF = None

DO_REDUCE = False
DO_COMPILE = False
DO_RESAMP = False
DO_FILTER = False
DO_DEMEAN = True
DO_PLOTS = False
DISABLED_ROBOTS = [54, 463, 608, 1136, 1182, 184, 1042]

centtypeList = ["winpos", "simple", "sep"]


def getRawFile(mjd, imgNum):
    baseDir = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/apo/%i"%mjd
    zpad = ("%i"%imgNum).zfill(4)
    return baseDir + "/proc-fimg-fvc1n-%s.fits"%zpad


def solveImage(ff, mjd=None, imgNum=None, sigma=0.7, boxSize=3, writecsv=False, clobber=False):
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
            for centtype in centtypeList:

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
                    centType=centtype
                )

                df = fvcT.positionerTableMeas.copy()
                df["polidName"] = polidName
                df["rotpos"] = ROTPOS
                df["ipa"] = IPA
                df["alt"] = ALT
                df["imgNum"] = imgNum
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

    p = Pool(15)
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
    rawDF.to_csv(WORK_DIR + "/raw.csv")
    print("DO_COMPILE took", (time.time()-tstart)/60, "minutes")


# df = df[df.slewNum.isin([1,2,3,4,5])]

if DO_RESAMP:
    rawDF = pandas.read_csv(WORK_DIR + "/raw.csv")
    tstart = time.time()
    slewNums = list(set(df.slewNum))
    # slewNums = list(range(1,25))

    #### boot strap averages over multiple exposures #####
    ### append this to the data set ####
    # only average over images in the same slewNum (telescope not moved)
    navgList = [2, 3] #, 5]
    nresample = 11
    replace = True  # tecnhically for bootstrap we should replace


    def resampleAndAverage(slewNum):
        sn = rawDF[rawDF.slewNum == slewNum].copy()
        mjd = int(sn.mjd.to_numpy()[0])
        imgNums = sorted(list(set(sn.imgNum)))
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

                for polidName in list(set(sn.polidName)):
                    for centtype in centtypeList:
                        _df = sn[(sn.imgNum.isin(imgList)) & (sn.polidName==polidName) & (sn.centtype==centtype)]
                        date = _df.date.iloc[0]
                        _ddf = _df.groupby(["positionerID"]).mean().reset_index()
                        _ddf["navg"] = navg
                        _ddf["polidName"] = polidName
                        _ddf["slewNum"] = slewNum
                        _ddf["date"] = date
                        _ddf["centtype"] = centtype
                        _ddf["imgNum"] = imgList[0]
                        _ddf = dropColsDF(_ddf)
                        dfList.append(_ddf)


        return pandas.concat(dfList)

    p = Pool(15)
    dfList = p.map(resampleAndAverage, slewNums)
    p.close()

    resampDF = pandas.concat(dfList)

    resampDF.to_csv(WORK_DIR + "/averaged.csv")

    print("DO_RESAMP took", (time.time()-tstart)/60, "minutes")


# begin filtering data
if DO_FILTER:

    t1 = time.time()
    rawDF = pandas.read_csv(WORK_DIR + "/raw.csv")
    print("loading raw file took", (time.time()-t1)/60)


    t1 = time.time()
    resampDF = pandas.read_csv(WORK_DIR + "/averaged.csv")
    print("loading resampled file took", (time.time()-t1)/60)

    # rawDF = rawDF[rawDF.mjd == 59661]
    # resampDF = resampDF[resampDF.mjd == 59661]

    rawDF = rawDF[rawDF.rotpos > -20]
    resampDF = resampDF[resampDF.rotpos > -20]

    # rawDF = rawDF[~rawDF.positionerID.isin(DISABLED_ROBOTS)]
    # resampDF = resampDF[~resampDF.positionerID.isin(DISABLED_ROBOTS)]

    # rawDF = rawDF[rawDF.wokErr < 0.5]
    # resampDF = resampDF[resampDF.wokErr < 0.5]

    # rawDF = rawDF[rawDF.polidName.isin(["nom", "all"])]
    # resampDF = resampDF[resampDF.polidName.isin(["nom", "all"])]

    keepCols = [
        "positionerID", "configid", "rotpos", "alt", "mjd", "date", "temp", "slewNum", "navg",
        "x", "x2", "y", "y2", "xWinpos", "yWinpos", "xSimple", "ySimple", "flux", "peak", "scale", "xtrans", "ytrans",
        "xWokMeasMetrology", "yWokMeasMetrology", "fiducialRMS", "polidName", "centtype", "wokErr", "imgNum"
    ] + ["ZB_%s"%("%i"%polid).zfill(2) for polid in range(33)]

    rawDF = rawDF[keepCols]
    rawDF.to_csv(WORK_DIR + "/raw_filtered.csv")
    resampDF = resampDF[keepCols]
    resampDF.to_csv(WORK_DIR + "/averaged_filtered.csv")
    all_filtered = pandas.concat([rawDF, resampDF])
    all_filtered.to_csv(WORK_DIR + "/all_filtered.csv")


if DO_DEMEAN:
    raw_filtered = pandas.read_csv(WORK_DIR + "/raw_filtered.csv")
    # resampled_filtered = pandas.read_csv(WORK_DIR + "/resampled_filtered.csv")
    # all_filtered = pandas.read_csv(WORK_DIR + "/all_filtered.csv")

    mean_rotmarg = raw_filtered.groupby(["positionerID", "mjd", "configid", "polidName", "centtype"]).mean().reset_index()

    avgCols = ["temp", "x", "x2", "y", "y2", "xWinpos", "yWinpos", "xSimple", "ySimple",
                "flux", "peak", "scale", "xtrans", "ytrans", "wokErr",
                "xWokMeasMetrology", "yWokMeasMetrology", "fiducialRMS"] + ["ZB_%s"%("%i"%polid).zfill(2) for polid in range(33)]

    t1 = time.time()

    pids = list(set(mean_rotmarg.positionerID))
    cfgs = list(set(mean_rotmarg.configid))
    polns = list(set(mean_rotmarg.polidName))
    rots = list(set(raw_filtered.rotpos))
    mjds = list(set(raw_filtered.mjd))
    slews = list(set(raw_filtered.slewNum))

    def doOne(positionerID):
        print("rot marg processing positioner", positionerID)
        # dfPOS = all_filtered[all_filtered.positionerID==positionerID].copy()
        dfList = []
        for mjd in mjds:
            for configid in cfgs:
                for polidName in polns:
                    for centtype in centtypeList:
                        _mean = mean_rotmarg[
                            (mean_rotmarg.positionerID==positionerID) & \
                            (mean_rotmarg.configid==configid) & \
                            (mean_rotmarg.polidName==polidName) & \
                            (mean_rotmarg.centtype==centtype) & \
                            (mean_rotmarg.mjd==mjd)
                        ]

                        _df = raw_filtered[
                            (raw_filtered.positionerID==positionerID) & \
                            (raw_filtered.configid==configid) & \
                            (raw_filtered.polidName==polidName) & \
                            (raw_filtered.centtype==centtype) & \
                            (raw_filtered.mjd==mjd)
                        ].copy()

                        if len(_mean) == 0 or len(_df) == 0:
                            continue

                        for col in avgCols:
                            _df[col] = _df[col].to_numpy() - float(_mean[col])

                        dfList.append(_df)
        return pandas.concat(dfList)

    # p = Pool(25)
    # filtered_demean = p.map(doOne, pids)
    # p.close()

    # # filtered_demean = [doOne(positionerID) for positionerID in pids]

    # filtered_demean_rotmarg = pandas.concat(filtered_demean)
    # filtered_demean_rotmarg.to_csv(WORK_DIR + "/filtered_demean_rotmarg.csv")
    # mean_rotmarg.to_csv(WORK_DIR + "/mean_rotmarg.csv")
    # print("demean rotmarg took", (time.time()-t1)/60)


    t1 = time.time()
    mean = raw_filtered.groupby(["positionerID", "configid", "polidName", "centtype", "rotpos", "slewNum"]).mean().reset_index()


    def doOther(positionerID):
        print("processing positioner", positionerID)
        # dfPOS = all_filtered[all_filtered.positionerID==positionerID].copy()
        dfList = []
        for slewNum in slews:
            for polidName in polns:
                for centtype in centtypeList:
                    _mean = mean[
                        (mean.positionerID==positionerID) & \
                        (mean.polidName==polidName) & \
                        (mean.slewNum==slewNum) & \
                        (mean.centtype==centtype)
                    ]

                    _df = raw_filtered[
                        (raw_filtered.positionerID==positionerID) & \
                        (raw_filtered.polidName==polidName) & \
                        (raw_filtered.slewNum==slewNum) & \
                        (raw_filtered.centtype==centtype)
                    ].copy()

                    if len(_mean) == 0 or len(_df) == 0:
                        continue

                    for col in avgCols:
                        _df[col] = _df[col].to_numpy() - float(_mean[col])

                    dfList.append(_df)
        return pandas.concat(dfList)

    p = Pool(25)
    all_filtered_demean = p.map(doOther, pids)
    p.close()

    # all_filtered_demean = [doOther(positionerID) for positionerID in pids]

    all_filtered_demean = pandas.concat(all_filtered_demean)
    all_filtered_demean.to_csv(WORK_DIR + "/filtered_demean.csv")
    mean.to_csv(WORK_DIR + "/mean.csv")
    print("demean took", (time.time()-t1)/60)


if DO_PLOTS:
    figCounter = 0

    def nf():
        global figCounter
        figCounter += 1
        zf = ("%i"%figCounter).zfill(2)
        return "fig"+zf+".png"

    def gauss(std, xs):
        ys = (1/(std*numpy.sqrt(2*numpy.pi)))*numpy.exp(-0.5*(xs/std)**2)
        return ys

    def fiberCircle(units, ax):
        if units=="mm":
            r = 0.060
        else: # pixels
            r = 0.50
        theta = numpy.linspace(0, 2*numpy.pi, 1000)
        x = r * numpy.cos(theta)
        y = r * numpy.sin(theta)
        ax.plot(x,y,'--', ms=5, color="tab:olive", label="fiber core")


    def fiberDia(units, ax):
        if units=="mm":
            r = 0.060
        else: # pixels
            r = 0.50
        ax.axvline(-r, ls="--", ms=5, color="tab:olive", label="fiber core")
        ax.axvline(r, ls="--", ms=5, color="tab:olive")

    t1 = time.time()
    all_filtered_demean_rotmarg = pandas.read_csv(WORK_DIR + "/all_filtered_demean_rotmarg.csv")
    all_filtered_demean = pandas.read_csv(WORK_DIR + "/all_filtered_demean.csv")
    mean_rotmarg = pandas.read_csv(WORK_DIR + "/mean_rotmarg.csv")
    mean = pandas.read_csv(WORK_DIR + "/mean.csv")

    all_filtered_demean_rotmarg = all_filtered_demean_rotmarg[all_filtered_demean_rotmarg.polidName=="nom"].reset_index()
    all_filtered_demean = all_filtered_demean[all_filtered_demean.polidName=="nom"].reset_index()
    mean_rotmarg = mean_rotmarg[mean_rotmarg.polidName=="nom"].reset_index()
    mean = mean[mean.polidName=="nom"].reset_index()

    print("csv load took", (time.time()-t1)/60)

    if True:
        xyMeas_rotmarg = all_filtered_demean_rotmarg[
            (all_filtered_demean_rotmarg.navg == 1) &\
            (all_filtered_demean_rotmarg.polidName == "nom")
        ][["xWokMeasMetrology", "yWokMeasMetrology"]]
        xyMeas_rotmarg["rot"] = "all"


        xyMeas = all_filtered_demean[
            (all_filtered_demean.navg == 1) &\
            (all_filtered_demean.polidName == "nom")
        ][["xWokMeasMetrology", "yWokMeasMetrology"]]
        xyMeas["rot"] = "single"

        df = pandas.concat([xyMeas_rotmarg, xyMeas])

        plt.figure(figsize=(10,10))
        plt.plot(xyMeas_rotmarg.xWokMeasMetrology, xyMeas_rotmarg.yWokMeasMetrology, '.', color='tab:blue', alpha=0.6, label="any rot")
        plt.plot(xyMeas.xWokMeasMetrology, xyMeas.yWokMeasMetrology, '.', color="tab:orange", alpha=0.6, label="single rot")
        fiberCircle("mm", plt.gca())
        plt.xlabel("dx (mm)")
        plt.ylabel("dy (mm)")
        plt.xlim([-.150, .150])
        plt.ylim([-.150, .150])
        plt.legend()
        plt.savefig(nf(), dpi=350)


        bins = numpy.linspace(-.150, .150, 500)
        smooth = numpy.linspace(-.150, .150, 2000)
        plt.figure(figsize=(10,5))
        plt.hist(xyMeas_rotmarg.xWokMeasMetrology, bins=bins, color="tab:blue", alpha=1, density=True, histtype="step", label="any rot")
        sigma = numpy.std(xyMeas_rotmarg.xWokMeasMetrology)
        ys = gauss(sigma, smooth)
        plt.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6, label="$\sigma$=%.3f"%sigma)
        plt.hist(xyMeas.xWokMeasMetrology, bins=bins, color="tab:orange", alpha=1, density=True, histtype="step", label="single rot")
        sigma = numpy.std(xyMeas.xWokMeasMetrology)
        ys = gauss(sigma, smooth)
        plt.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6, label="$\sigma$=%.3f"%sigma)
        fiberDia("mm", plt.gca())
        plt.legend()
        plt.xlabel("dx (mm)")
        plt.yticks([])
        plt.ylabel("density")
        plt.xlim([-.150, .150])
        plt.savefig(nf(), dpi=350)


        bins = numpy.linspace(-.150, .150, 500)
        smooth = numpy.linspace(-.150, .150, 2000)
        plt.figure(figsize=(10,5))
        plt.hist(xyMeas_rotmarg.yWokMeasMetrology, bins=bins, color="tab:blue", alpha=1, density=True, histtype="step", label="any rot")
        sigma = numpy.std(xyMeas_rotmarg.yWokMeasMetrology)
        ys = gauss(sigma, smooth)
        plt.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6, label="$\sigma$=%.3f"%sigma)
        plt.hist(xyMeas.yWokMeasMetrology, bins=bins, color="tab:orange", alpha=1, density=True, histtype="step", label="single rot")
        sigma = numpy.std(xyMeas.yWokMeasMetrology)
        ys = gauss(sigma, smooth)
        plt.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6, label="$\sigma$=%.3f"%sigma)
        fiberDia("mm", plt.gca())
        plt.legend()
        plt.xlabel("dy (mm)")
        plt.yticks([])
        plt.ylabel("density")
        plt.xlim([-.150, .150])
        plt.savefig(nf(), dpi=350)




        ################### CENTROID #####################
        ##################################################

        xyMeas = all_filtered_demean[
            (all_filtered_demean.navg == 1) &\
            (all_filtered_demean.polidName == "nom")
        ][["x", "y"]]
        xyMeas["type"] = "sep extract"

        xyWpMeas = all_filtered_demean[
            (all_filtered_demean.navg == 1) &\
            (all_filtered_demean.polidName == "nom")
        ][["xWinpos", "yWinpos"]]
        xyWpMeas["type"] = "sep 3x3 winpos"
        xyWpMeas["x"] = xyWpMeas.xWinpos
        xyWpMeas["y"] = xyWpMeas.yWinpos
        xyWpMeas = xyWpMeas[["x", "y", "type"]]

        xyBoth = pandas.concat([xyMeas, xyWpMeas]).reset_index()

        plt.figure(figsize=(10,10))
        plt.plot(xyWpMeas.x, xyWpMeas.y, '.', color="tab:blue", alpha=0.6, label="sep 3x3 winpos")
        plt.plot(xyMeas.x, xyMeas.y, '.', color='tab:orange', alpha=0.6, label="sep extract")
        plt.xlabel("dx (pix)")
        plt.ylabel("dy (pix)")
        fiberCircle("pix", plt.gca())
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.legend()
        plt.savefig(nf(), dpi=350)

        palette = {"sep 3x3 winpos": "tab:blue", "sep extract": "tab:orange"}

        if False:
            # takes 9 mins to generate
            t1 = time.time()
            plt.figure(figsize=(10,10))
            sns.kdeplot(x="x", y="y", data=xyBoth, hue="type", palette=palette, levels=10)
            plt.xlabel("dx (pix)")
            plt.ylabel("dy (pix)")
            plt.xlim([-1.5, 1.5])
            plt.ylim([-1.5, 1.5])
            plt.legend()
            plt.savefig(nf(), dpi=350)
            print("kdeplot took", (time.time()-t1)/60)


        bins = numpy.linspace(-1.5, 1.5, 500)
        smooth = numpy.linspace(-1.5, 1.5, 2000)
        plt.figure(figsize=(10,5))
        plt.hist(xyMeas.x, bins=bins, color="tab:orange", alpha=1, density=True, histtype="step", label="sep extract")
        sigma = numpy.std(xyMeas.x)
        ys = gauss(sigma, smooth)
        plt.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6, label="$\sigma$=%.3f"%sigma)
        plt.hist(xyWpMeas.x, bins=bins, color="tab:blue", alpha=1, density=True, histtype="step", label="sep 3x3 winpos")
        sigma = numpy.std(xyWpMeas.x)
        ys = gauss(sigma, smooth)
        plt.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6, label="$\sigma$=%.3f"%sigma)
        fiberDia("pix", plt.gca())
        plt.legend()
        plt.xlabel("dx (pix)")
        plt.yticks([])
        plt.ylabel("density")
        plt.xlim([-1.5, 1.5])
        plt.savefig(nf(), dpi=350)

        bins = numpy.linspace(-1.5, 1.5, 500)
        smooth = numpy.linspace(-1.5, 1.5, 2000)
        plt.figure(figsize=(10,5))
        plt.hist(xyMeas.y, bins=bins, color="tab:orange", alpha=1, density=True, histtype="step", label="sep extract")
        sigma = numpy.std(xyMeas.y)
        ys = gauss(sigma, smooth)
        plt.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6, label="$\sigma$=%.3f"%sigma)
        plt.hist(xyWpMeas.y, bins=bins, color="tab:blue", alpha=1, density=True, histtype="step", label="sep 3x3 winpos")
        sigma = numpy.std(xyWpMeas.y)
        ys = gauss(sigma, smooth)
        plt.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6, label="$\sigma$=%.3f"%sigma)
        fiberDia("pix", plt.gca())
        plt.legend()
        plt.xlabel("dy (pix)")
        plt.yticks([])
        plt.ylabel("density")
        plt.xlim([-1.5, 1.5])
        plt.savefig(nf(), dpi=350)


        # do bigger errors cluster on the CCD?

        ### plot difference in avg winpos vs average sep extract positions
        xWinpos = mean.xWinpos.to_numpy()
        yWinpos = mean.yWinpos.to_numpy()
        x = mean.x.to_numpy()
        y = mean.y.to_numpy()

        dx = x - xWinpos
        dy = y - yWinpos

        plt.figure(figsize=(10,10))
        q = plt.quiver(x,y,dx,dy,angles="xy",units="xy", width=1, scale=0.01)
        ax = plt.gca()
        ax.quiverkey(q, 0.9, 0.9, 1, "1 pixel")
        plt.axis("equal")
        plt.xlabel("x (pix)")
        plt.ylabel("y (pix)")
        plt.title("sep xy - winpos xy")
        plt.savefig(nf(), dpi=350)

        # remove translation rotation and scale
        xy1 = numpy.array([x,y]).T
        xy2 = numpy.array([xWinpos, yWinpos]).T
        tf = SimilarityTransform()
        tf.estimate(xy1, xy2)
        xyFit = tf(xy1)
        dxy = xyFit - xy2
        plt.figure(figsize=(10,10))
        q = plt.quiver(xyFit[:,0], xyFit[:,1], dxy[:,0],dxy[:,1],angles="xy",units="xy", width=1, scale=0.01)
        ax = plt.gca()
        ax.quiverkey(q, 0.9, 0.9, 1, "1 pixel")
        plt.axis("equal")
        plt.xlabel("x (pix)")
        plt.ylabel("y (pix)")
        plt.title("sep xy - winpos xy\nremoving trans/rot/scale")
        plt.savefig(nf(), dpi=350)

        # effects of various flavors of averaging, bootstrap
        # compute a new statistic rErr
        df = all_filtered_demean[["xWinpos", "yWinpos", "navg", "avgType", "pixelCombine"]].copy()


        def panelAvgPlots(df, xName, yName, units):
            if units == "pix":
                lims = [-1.5,1.5]
                bins = numpy.linspace(-1.5, 1.5, 500)
            else:
                lims = [-0.15, 0.15]
                bins = numpy.linspace(-.15, .15, 500)

            fig1, axs = plt.subplots(2,2,figsize=(10,10))
            fig2, axs2 = plt.subplots(2,2,figsize=(10,5))
            axs = axs.flatten()
            axs2 = axs2.flatten()

            iax = 0
            colorset = ["tab:gray", "tab:blue", "tab:orange", "tab:red"]
            for avgType in list(set(df.avgType)):
                if avgType == "none":
                    continue
                for pixelCombine in list(set(df.pixelCombine)):
                    ax1 = axs[iax]
                    ax1.set_xlim(lims)
                    ax1.set_ylim(lims)
                    ax1.set_xlabel("dx (%s)"%units)
                    ax1.set_ylabel("dy (%s)"%units)

                    ax2 = axs2[iax]
                    ax2.set_xlim(lims)
                    ax2.set_xlabel("dy (%s)"%units)
                    ax2.set_ylabel("density")

                    for color, navg in zip(colorset, [1,2,3,5]):
                        if navg == 1:
                            xy = df[
                                (df.navg == 1)
                            ]
                        else:
                            xy = df[
                                (df.avgType==avgType) &\
                                (df.pixelCombine==pixelCombine) &\
                                (df.navg == navg)
                            ]
                        stdy = numpy.std(xy[yName])
                        label = "<n=%i> $\sigma_y$=%.3f"%(navg, stdy)
                        ax1.plot(xy[xName], xy[yName], '.', color=color, alpha=0.6, ms=5, label=label)
                        ax2.hist(xy[yName], bins=bins, color=color, alpha=1, density=True, histtype="step", label=label)

                    fiberCircle(units, ax1)
                    fiberDia(units, ax2)
                    title = "avgType=%s overPixels=%s"%(avgType, str(bool(pixelCombine)))
                    ax1.set_title(title, fontsize=6)
                    ax1.legend(prop={'size': 6}, loc="upper right")

                    ax2.set_title(title, fontsize=6)
                    ax2.legend(prop={'size': 6}, loc="upper right")
                    iax += 1


            fig1.savefig(nf(), dpi=350)
            fig2.savefig(nf(), dpi=350)
            plt.close("all")

        panelAvgPlots(df, "xWinpos", "yWinpos", "pix")

        df = all_filtered_demean[["xWokMeasMetrology", "yWokMeasMetrology", "navg", "avgType", "pixelCombine"]].copy()

        panelAvgPlots(df, "xWokMeasMetrology", "yWokMeasMetrology", "mm")


        ############### plot stuff as a function of CCD location #############
        df = mean.copy()
        df["x2/y2"] = df.x2/df.y2

        for hue in ["x2", "y2", "x2/y2", "flux", "peak"]:
            plt.figure(figsize=(10,10))
            sns.scatterplot(x="x", y="y", palette="rocket", hue=hue, data=df, s=7)
            plt.xlabel("x (pix)")
            plt.ylabel("y (pix)")
            # plt.axes("equal")
            plt.savefig(nf(), dpi=350)
            plt.close("all")


        # does xyWok measurement repeatability depend on CCD location?
        # this isn't working skip for now?

        df = all_filtered_demean[
            ["positionerID", "configid", "rotpos", "xWinpos", "yWinpos", "x2", "y2",
            "xWokMeasMetrology", "yWokMeasMetrology", "flux", "peak", "fiducialRMS"]
        ].copy()
        df["wokErr (mm)"] = numpy.sqrt(df.xWokMeasMetrology**2 + df.yWokMeasMetrology**2)
        df["x2/y2"] = df.x2/df.y2
        df_a = df.groupby(["positionerID", "configid", "rotpos"]).median().reset_index()
        df_a["medianWokErr (mm)"] = df_a["wokErr (mm)"]
        df_mean = mean[["positionerID", "configid", "rotpos", "xWinpos", "yWinpos", "x2", "y2",
            "xWokMeasMetrology", "yWokMeasMetrology", "flux", "peak", "fiducialRMS"]
        ]
        _df = df_mean.merge(df_a, on=["positionerID", "configid", "rotpos"], suffixes=(None, "_med"))
        plt.figure(figsize=(10,10))
        sns.scatterplot(x="xWokMeasMetrology", y="yWokMeasMetrology", palette="rocket", hue="medianWokErr (mm)", data=_df, s=14)
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.savefig(nf(), dpi=350)

        plt.figure(figsize=(10,10))
        sns.scatterplot(x="xWinpos", y="yWinpos", palette="rocket", hue="medianWokErr (mm)", data=_df, s=7)
        plt.xlabel("x (pix)")
        plt.ylabel("y (pix)")
        plt.savefig(nf(), dpi=350)




        positionerIDs = sorted(list(set(df.positionerID)))
        step = 50
        begi = 0
        for ii in range(11):
            print("on ii", ii)
            _pid = positionerIDs[begi:begi+step]
            if len(_pid) == 0:
                print('break!')
                break
            begi += step

            _df = df[df.positionerID.isin(_pid)]
            plt.figure(figsize=(10,5))
            sns.boxplot(x="positionerID", y="wokErr (mm)", data=_df) #, palette="rocket", hue="medianWokErr (mm)", data=_df, s=2)
            plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
            plt.xticks(rotation = 60)
            plt.ylim([0, 0.07])
            plt.savefig(nf(), dpi=350); plt.close("all")

        plt.figure(figsize=(10,5))
        sns.boxplot(x="rotpos", y="wokErr (mm)", data=_df) #, palette="rocket", hue="medianWokErr (mm)", data=_df, s=2)
        plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
        plt.ylim([0, 0.07])
        plt.xticks(rotation = 60)
        plt.savefig(nf(), dpi=350); plt.close("all")

        plt.figure(figsize=(10,10))
        plt.plot(df.fiducialRMS, df["wokErr (mm)"], '.k', alpha=0.005, markersize=1)
        plt.xlabel("fit rms - <rms> (mm)")
        plt.ylabel("median wokErr (mm)")
        plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
        plt.ylim([0, 0.07])
        plt.savefig(nf(), dpi=350)
        # plt.xlim([-0.25, 0.25])
        # plt.savefig(nf(), dpi=350)
        plt.xlim([-0.005, 0.005])
        plt.savefig(nf(), dpi=350)
        # plt.xlim([-0.25, -.2])
        # plt.savefig(nf(), dpi=350)


        plt.figure(figsize=(10,10))
        plt.plot(df.flux, df["wokErr (mm)"], '.k', alpha=0.005, markersize=1)
        plt.xlabel("flux - <flux> (counts)")
        plt.ylabel("wokErr (mm)")
        plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
        plt.ylim([0, 0.07])
        plt.xlim([-9000,9000])
        plt.savefig(nf(), dpi=350); plt.close("all")

        plt.figure(figsize=(10,10))
        plt.plot(df.peak, df["wokErr (mm)"], '.k', alpha=0.005, markersize=1)
        plt.xlabel("peak - <peak> (counts)")
        plt.ylabel("wokErr (mm)")
        plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
        plt.ylim([0, 0.07])
        plt.savefig(nf(), dpi=350); plt.close("all")

        plt.figure(figsize=(10,10))
        plt.plot(df.x2, df["wokErr (mm)"], '.k', alpha=0.005, markersize=1)
        plt.xlabel("x2 - <x2>")
        plt.ylabel("wokErr (mm)")
        plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
        plt.ylim([0, 0.07])
        plt.xlim([-0.75, 1.75])
        plt.savefig(nf(), dpi=350); plt.close("all")

        plt.figure(figsize=(10,10))
        plt.plot(df.y2, df["wokErr (mm)"], '.k', alpha=0.005, markersize=1)
        plt.xlabel("y2 - <y2>")
        plt.ylabel("wokErr (mm)")
        plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
        plt.ylim([0, 0.07])
        plt.xlim([-0.75, 1.75])
        plt.savefig(nf(), dpi=350); plt.close("all")


    # does one rotator angle best represent the mean?
    df =mean.merge(mean_rotmarg, on=["positionerID", "configid"], suffixes=(None, "_y"))
    print(len(mean), len(mean_rotmarg), len(df))
    df["dx"] = df.xWokMeasMetrology - df.xWokMeasMetrology_y
    df["dy"] = df.yWokMeasMetrology - df.yWokMeasMetrology_y
    df["wokErr (mm)"] = numpy.sqrt(df.dx**2+df.dy**2)

    plt.figure(figsize=(10,5))
    sns.boxplot(x="rotpos", y="wokErr (mm)", data=df) #, palette="rocket", hue="medianWokErr (mm)", data=_df, s=2)
    plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
    plt.ylim([0, 0.15])
    plt.xticks(rotation = 60)
    plt.title("xy - <xy>")
    plt.savefig(nf(), dpi=350); plt.close("all")

    mean_190 = mean[mean.rotpos==190]
    df =mean.merge(mean_190, on=["positionerID", "configid"], suffixes=(None, "_y"))
    df["dx"] = df.xWokMeasMetrology - df.xWokMeasMetrology_y
    df["dy"] = df.yWokMeasMetrology - df.yWokMeasMetrology_y
    df["wokErr (mm)"] = numpy.sqrt(df.dx**2+df.dy**2)

    plt.figure(figsize=(10,5))
    sns.boxplot(x="rotpos", y="wokErr (mm)", data=df) #, palette="rocket", hue="medianWokErr (mm)", data=_df, s=2)
    plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
    plt.ylim([0, 0.15])
    plt.xticks(rotation = 60)
    plt.title("xy - <xy 90 deg>")
    plt.savefig(nf(), dpi=350); plt.close("all")







    # plt.figure(figsize=(10,10))
    # plt.plot(df.x2/df.y2, df["wokErr (mm)"], '.k', alpha=0.005, markersize=1)
    # plt.xlabel("x2/y2")
    # plt.ylabel("wokErr (mm)")
    # plt.axhline(0.06, ls="--", color="tab:olive", label="fiber core")
    # plt.ylim([0, 0.07])
    # plt.savefig(nf(), dpi=350); plt.close("all")

    # does one rotator value best approximate the mean model?
    # import pdb; pdb.set_trace()
    # df = mean.merge(mean_rotmarg,


















