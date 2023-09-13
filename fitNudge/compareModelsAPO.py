import pandas
from coordio.utils import fitsTableToPandas
from coordio.transforms import FVCTransformAPO
from astropy.io import fits
import numpy
from skimage.transform import SimilarityTransform
import matplotlib.pyplot as plt
import time
import seaborn as sns
from multiprocessing import Pool
from functools import partial
from skimage.exposure import equalize_hist
import glob

# for comparing differences between initial nudge model,
# and nudge model after rotating the baffle

# data from commissioning
# MJD = 59807
baseDir = "/Volumes/futa/apo/data/fcam/" #%i"%MJD
# configid : [minImg, maxImg] map

cmap = {}
dd = {}
dd[121538] = []
dd[128788] = []
for img in sorted(list(glob.glob(baseDir+"/60100/proc*.fits"))):
    ff = fits.open(img)
    imgNum = int(img.split("-")[-1].split(".fits")[0])
    designID = ff[1].header["DESIGNID"]
    if designID in [121538, 128788]:
        dd[designID].append(imgNum)

cmap = {
    60100 : {
        121538 : [dd[121538][0], dd[121538][-1]],
        128788 : [dd[128788][0], dd[128788][-1]],
    },
    60200 : {
        1 : [4,111],
        2 : [148,255],
        3 : [256, 366],
        4 : [367, 474]
    }
}


# find image numbers and design ids from a previous apo rotator scan


IMAX = 32
# IMAX = 44 # maximum integer wave number
DELTAK = 2. * numpy.pi / 10000.0 # wave number spacing in inverse pixels
# SAVE_COEFFS = False


def extractOne(imgNum, mjd, centType="sep"):
    imgFilePath = getImgFile(mjd, imgNum)
    ff = fits.open(imgFilePath)
    imgData = ff[1].data
    pc = fitsTableToPandas(ff["POSANGLES"].data)
    fvct = FVCTransformAPO(
        imgData, pc, ff[1].header["IPA"],
        plotPathPrefix=imgFilePath,
    )
    fvct.extractCentroids() #ccdRotCenXY=numpy.array([3066., 3055.])) #centroidMinNpix=5)
    print("minNPix", fvct.centroidMinNpix)
    fvct.fit(centType=centType)
    ptm = fvct.positionerTableMeas
    fcm = fvct.fiducialCoordsMeas
    xyFIFexpect = fvct.similarityTransform.inverse(fvct.fiducialCoords[["xWok", "yWok"]].to_numpy())
    plt.figure(figsize=(8,8))
    plt.imshow(fvct.data_sub, origin="lower")
    plt.plot(xyFIFexpect[:,0], xyFIFexpect[:,1], 'o', mfc="none", mec="red", ms=5)

    plt.figure(figsize=(8,8))
    plt.imshow(numpy.exp(fvct.data_sub), origin="lower")
    plt.plot(xyFIFexpect[:,0], xyFIFexpect[:,1], 'o', mfc="none", mec="red", ms=5)

    plt.figure(figsize=(8,8))
    plt.imshow(numpy.exp(fvct.data_sub), origin="lower")
    plt.plot(fvct.centroids.x, fvct.centroids.y, 'o', mfc="none", mec="red", ms=5)
    plt.plot(fvct.centroids.xSimple, fvct.centroids.ySimple, 'x', mfc="none", mec="red", ms=5)

    # import pdb; pdb.set_trace()

    plt.show()

    # import pdb; pdb.set_trace()


def pix2wok(xCCD, yCCD, tx, ty, rot, scale):
    crot = numpy.cos(rot)
    srot = numpy.sin(rot)

    xWok = scale * (xCCD * crot - yCCD * srot) + tx
    yWok = scale * (xCCD * srot + yCCD * crot) + ty
    return xWok, yWok


def rotScaleWok2pix(dxWok, dyWok, rot, scale):
    crot = numpy.cos(-1*rot)
    srot = numpy.sin(-1*rot)
    # positive rot is for pixels to wok
    # negative rot wok to pixels

    dxWok = dxWok / scale
    dyWok = dyWok / scale
    dxCCD = dxWok * crot - dyWok * srot
    dyCCD = dxWok * srot + dyWok * crot
    return dxCCD, dyCCD


def getImgFile(mjd, imgNum):
    imgNumStr = str(imgNum).zfill(4)
    return baseDir + "/%i/proc-fimg-fvc1n-%s.fits"%(mjd,imgNumStr)


def fourier_functions(xs, ys):
    n = len(xs)
    assert len(ys) == n
    fxs = numpy.zeros((n, IMAX * 2 + 2))
    fys = numpy.zeros((n, IMAX * 2 + 2))
    iis = numpy.zeros(IMAX * 2 + 2).astype(int)
    for i in range(IMAX+1):
        fxs[:, i * 2] = numpy.cos(i * DELTAK * xs)
        fys[:, i * 2] = numpy.cos(i * DELTAK * ys)
        iis[i * 2] = i
        fxs[:, i * 2 + 1] = numpy.sin((i + 1) * DELTAK * xs)
        fys[:, i * 2 + 1] = numpy.sin((i + 1) * DELTAK * ys)
        iis[i * 2 + 1] = i + 1
    return fxs, fys, iis


def design_matrix(xs, ys):
    fxs, fys, iis = fourier_functions(xs, ys)
    n, p = fxs.shape
    Xbig = (fxs[:, :, None] * fys[:, None, :]).reshape((n, p * p))
    i2plusj2 = (iis[:, None] ** 2 + iis[None, :] ** 2).reshape(p * p)
    return Xbig[:, i2plusj2 <= IMAX ** 2]


def fitDistortion(xs, ys, dxs, dys, trainFrac=0.8):

    t1 = time.time()
    X = design_matrix(xs, ys)
    print("design_matrix took", time.time()-t1)
    print("X.shape", X.shape)

    n, p = X.shape

    numpy.random.seed(42)
    rands = numpy.random.uniform(size=n)

    train = rands <= trainFrac
    test = rands > trainFrac

    # train = rands <= 1
    # test = rands <=1
    # print(numpy.sum(train), numpy.sum(test))

    t1 = time.time()
    beta_x, resids, rank, s = numpy.linalg.lstsq(X[train], dxs[train], rcond=None)

    print("fit took", time.time()-t1)
    dxs_hat = X[test] @ beta_x

    print("original dx (test set) RMS:", numpy.sqrt(numpy.mean(dxs[test] ** 2)))
    print("dx - dx_hat (test set) RMS:", numpy.sqrt(numpy.mean((dxs[test] - dxs_hat) ** 2)))

    t1 = time.time()
    beta_y, resids, rank, s = numpy.linalg.lstsq(X[train], dys[train], rcond=None)
    print("fit took", time.time()-t1)
    dys_hat = X[test] @ beta_y

    print("original dy (test set) RMS:", numpy.sqrt(numpy.mean(dys[test] ** 2)))
    print("dy - dy_hat (test set) RMS:", numpy.sqrt(numpy.mean((dys[test] - dys_hat) ** 2)))

    return beta_x, beta_y


def applyDistortion(xs, ys, beta_x, beta_y):
    X = design_matrix(xs, ys)
    dxs_hat = X @ beta_x
    dys_hat = X @ beta_y
    return dxs_hat, dys_hat


def extractData(imgNum, mjd, configid, reprocess=False, centType="sep", simpleSigma=1, beta_x=None, beta_y=None, polids=None):
    print("extracting", mjd, imgNum)
    ff = fits.open(getImgFile(mjd, imgNum))
    if reprocess:
        imgData = ff[1].data
        pc = fitsTableToPandas(ff["POSANGLES"].data)
        fvct = FVCTransformAPO(
            imgData, pc, ff[1].header["IPA"], polids=polids
        )
        fvct.extractCentroids(centroidMinNpix=100, simpleSigma=simpleSigma, beta_x=beta_x, beta_y=beta_y)
        fvct.fit(centType=centType)
        ptm = fvct.positionerTableMeas
        fcm = fvct.fiducialCoordsMeas

    else:
        ptm = fitsTableToPandas(ff["POSITIONERTABLEMEAS"].data)
        fcm = fitsTableToPandas(ff["FIDUCIALCOORDSMEAS"].data)

    trans = numpy.array([ff[1].header["FVC_TRAX"], ff[1].header["FVC_TRAY"]])
    scale = ff[1].header["FVC_SCL"]
    rot = ff[1].header["FVC_ROT"]
    ipa = numpy.around(ff[1].header["IPA"], decimals=1)
    # import pdb; pdb.set_trace()
    # ptm = ptm[ptm.wokErrWarn] # remove bogus measurements (missing spots)
    ptm = ptm[["positionerID", "x", "y", "x2", "y2", "cflux", "xWokReportMetrology", "yWokReportMetrology", "xWokMeasMetrology", "yWokMeasMetrology"]]


    fcm["xWokReportMetrology"] = fcm.xWok
    fcm["yWokReportMetrology"] = fcm.yWok
    fcm["xWokMeasMetrology"] = fcm.xWok
    fcm["yWokMeasMetrology"] = fcm.yWok
    # fake an integer id for fiducials, make them negative for easy selection
    _positionerID = numpy.array([int(x.strip("F")) for x in fcm.id.to_numpy()])
    fcm["positionerID"] = -1*_positionerID

    fcm = fcm[["positionerID", "x", "y", "x2", "y2", "cflux", "xWokReportMetrology", "yWokReportMetrology", "xWokMeasMetrology", "yWokMeasMetrology"]]

    ptm = pandas.concat([fcm,ptm])

    ptm["config"] = configid
    ptm["imgNum"] = imgNum
    ptm["transx"] = trans[0]
    ptm["transy"] = trans[1]
    ptm["rot"] = numpy.radians(rot)
    ptm["scale"] = scale
    ptm["ipa"] = ipa
    ptm["mjd"] = mjd
    ptm["centType"] = centType
    ptm["simpleSigma"] = simpleSigma

    return ptm.reset_index()


def compileData(reprocess=True):
    dfList = []
    for mjd, dd in cmap.items():
        for config, (minImg, maxImg) in dd.items():
            imgNums = list(range(minImg, maxImg+1))
            for imgNum in imgNums:
                dfList.append(extractData(imgNum,mjd,config,reprocess=reprocess, centType="sep"))

            # _func = partial(extractData, mjd=mjd, configid=config, reprocess=reprocess)
            # p = Pool(2)
            # _dfList = p.map(_func, imgNums)
            # dfList.extend(_dfList)

            # for imgNum in range(minImg, maxImg+1):
            #     print("on mjd, img", mjd, imgNum)
            #     _df = extractData(imgNum, mjd, config, reprocess=reprocess)
            #     _df["mjd"] = mjd
            #     dfList.append(_df)

    df = pandas.concat(dfList)
    df.to_csv("rawMeas_comp.csv")


def _plotImgQuality():
    df = pandas.read_csv("rawMeas_comp.csv")

    ## histograms
    fig, axs = plt.subplots(2,2, figsize=(8,4))
    fig.suptitle("PSF spread (2nd moments)")
    for ii, mjd in enumerate(cmap.keys()):
        _df = df[df.mjd==mjd]
        axs[ii][0].hist(_df.x2, bins=200, density=True)
        axs[ii][0].set_xlim([5,30])
        axs[ii][0].set_ylabel("MJD: %i"%mjd)
        axs[ii][0].set_yticks([])

        axs[ii][1].hist(_df.y2, bins=200, density=True)
        axs[ii][1].set_xlim([5,30])
        axs[ii][1].set_yticks([])

    axs[1][0].set_xlabel("x $\sigma^2$ (pix$^2$)")
    axs[1][1].set_xlabel("y $\sigma^2$ (pix$^2$)")
    plt.tight_layout()

    plt.savefig("imgQualHist.png", dpi=250)


    ## 2D plots psf spread
    fig, axs = plt.subplots(2,2, figsize=(8,8))
    fig.suptitle("PSF spread (2nd moments)")
    for ii, mjd in enumerate(cmap.keys()):
        _df = df[df.mjd==mjd]
        axs[ii][0].scatter(_df.x,_df.y,c=_df.x2,s=0.5,vmin=5,vmax=30)
        axs[ii][0].set_ylabel("MJD: %i\ny (pix)"%mjd)
        axs[ii][0].set_aspect("equal")

        pcm = axs[ii][1].scatter(_df.x,_df.y,c=_df.y2,s=0.5,vmin=5,vmax=30)
        axs[ii][0].set_ylabel("MJD: %i\ny (pix)"%mjd)
        axs[ii][1].set_aspect("equal")
        fig.colorbar(pcm, orientation="vertical", shrink=0.6, ax=axs[ii][0], location="right")
        fig.colorbar(pcm, orientation="vertical", shrink=0.6, ax=axs[ii][1], location="right")


    axs[0][0].set_title("x $\sigma^2$")
    axs[0][1].set_title("y $\sigma^2$")
    axs[1][0].set_xlabel("x (pix)")
    axs[1][1].set_xlabel("x (pix)")
    plt.savefig("sigmaXY.png", dpi=250)
    # plt.colorbar()

    # 2D plots flux
    # calculate flux loss (median)
    df_med = df.groupby(["positionerID", "mjd", "centType"]).mean().reset_index()
    df = df.merge(df_med, on=["positionerID", "mjd", "centType"], suffixes=(None,"_m"))
    df["fluxPerc"] = df.cflux / df.cflux_m

    fig, axs = plt.subplots(1,2, figsize=(8,4))
    fig.suptitle("Flux")
    for ii, mjd in enumerate(cmap.keys()):
        _df = df[df.mjd==mjd]
        pcm = axs[ii].scatter(_df.x,_df.y,c=_df.fluxPerc,s=0.75,vmin=0.5,vmax=1)
        axs[ii].set_title("MJD: %i"%mjd)
        axs[ii].set_xlabel("x (pix)")
        axs[ii].set_aspect("equal")
        fig.colorbar(pcm, orientation="vertical", shrink=0.6, ax=axs[ii], location="right")

    axs[0].set_ylabel("y (pix)")
    plt.savefig("flux.png", dpi=250)


def measureMeanDistortion(includeFIFs=True, useZB=True, saveCoeffs=False):
    _df = pandas.read_csv("rawMeas_comp.csv")

    # for ii, mjd in enumerate([59807,60174,60176]):
    #     for centType, simpleSigma in [["sep", 1],["simple", 1], ["simple", 7]]:
    centType = "sep"
    simpleSigma = 1
    _df = _df[_df.centType==centType]
    _df = _df[_df.simpleSigma==simpleSigma]


    if useZB:
        _df["xWokSim"] = _df.xWokMeasMetrology
        _df["yWokSim"] = _df.yWokMeasMetrology
    elif centType == "sep":
        xWok, yWok = pix2wok(
            _df.x.to_numpy(), _df.y.to_numpy(), _df.transx.to_numpy(),
            _df.transy.to_numpy(), _df.rot.to_numpy(), _df.scale.to_numpy()
        )
        _df["xWokSim"] = xWok
        _df["yWokSim"] = yWok
    elif centType == "simple":
        xWok, yWok = pix2wok(
            _df.xSimple.to_numpy(), _df.ySimple.to_numpy(), _df.transx.to_numpy(),
            _df.transy.to_numpy(), _df.rot.to_numpy(), _df.scale.to_numpy()
        )
        _df["xWokSim"] = xWok
        _df["yWokSim"] = yWok

    _df_mean = _df.groupby(["positionerID", "config", "mjd", "centType", "simpleSigma"]).mean().reset_index()
    _df = _df.merge(_df_mean, on=["positionerID", "config", "mjd", "centType", "simpleSigma"], suffixes=(None, "_mean"))

    _df["dxWok"] = _df.xWokSim - _df.xWokSim_mean
    _df["dyWok"] = _df.yWokSim - _df.yWokSim_mean
    _df["drWok"] = numpy.sqrt(_df.dxWok**2+_df.dyWok**2)

    _df = _df[_df.drWok < 1]

    dxCCD, dyCCD = rotScaleWok2pix(
        _df.dxWok.to_numpy(), _df.dyWok.to_numpy(),
        _df.rot.to_numpy(), _df.scale.to_numpy()
    )

    _df["dxCCD"] = dxCCD
    _df["dyCCD"] = dyCCD

    if not includeFIFs:
        _df = _df[_df.positionerID >= 0]

    # print("mdj n points", len(_df))

    # plt.figure()
    # plt.hist(_df.drWok, bins=200)
    # plt.title("MJD: %i"%mjd)

    for mjd in cmap.keys():

        xs = _df[_df.mjd==mjd]["x"].to_numpy()
        ys = _df[_df.mjd==mjd]["y"].to_numpy()
        dxs = _df[_df.mjd==mjd]["dxCCD"].to_numpy()
        dys = _df[_df.mjd==mjd]["dyCCD"].to_numpy()

        dr = numpy.sqrt(dxs**2+dys**2)
        rms = numpy.sqrt(numpy.mean(dr**2))
        med = numpy.median(dr)
        perc95 = numpy.percentile(dr, 95)

        plt.figure(figsize=(8,8))
        plt.title("MJD: %i  centType: %s  simpleSigma: %i\nMean Distortion"%(mjd, centType, simpleSigma))
        q = plt.quiver(xs,ys,dxs,dys, angles="xy", units="xy", alpha=0.5, width=2, scale=0.005)
        ax = plt.gca()
        ax.quiverkey(q, 0.85, 0.85, 0.5, "units: pixels\nRMS: %.2f\nMedian: %.2f\nperc95: %.2f\nquiver length: 0.5"%(rms,med,perc95))
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        ax.set_aspect("equal")
        plt.savefig("meanDistortion_%i_%s_%i.png"%(mjd, centType,simpleSigma), dpi=250)

        # fit and plot distortion model
        beta_x, beta_y = fitDistortion(xs, ys, dxs, dys, trainFrac=0.8)
        if saveCoeffs:
            with open("beta_x_%i_%s_%i.npy"%(mjd,centType, simpleSigma), "wb") as f:
                numpy.save(f, beta_x)
            with open("beta_y_%i_%s_%i.npy"%(mjd, centType, simpleSigma), "wb") as f:
                numpy.save(f, beta_y)

        dx_hats, dy_hats = applyDistortion(xs,ys,beta_x,beta_y)

        r_hats = numpy.sqrt(dx_hats**2+dy_hats**2)
        rms = numpy.sqrt(numpy.mean(r_hats**2))
        median = numpy.median(r_hats)
        perc95 = numpy.percentile(r_hats, 95)

        plt.figure(figsize=(8,8))
        plt.title("MJD: %i   centType: %s  simpleSigma: %i\nDistortion (nudge) Model"%(mjd, centType,simpleSigma))
        q = plt.quiver(xs,ys,dx_hats,dy_hats, angles="xy", units="xy", alpha=0.5, width=2, scale=0.005)
        ax = plt.gca()
        ax.quiverkey(q, 0.85, 0.85, 0.5, "units: pixels\nRMS: %.2f\nMedian: %.2f\nperc95: %.2f\nquiver length: 0.5"%(rms,med,perc95))
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        ax.set_aspect("equal")
        plt.savefig("modelDistortion_%i_%s_%i.png"%(mjd, centType, simpleSigma), dpi=250)

        # plot model residuals
        dxResid = dxs - dx_hats
        dyResid = dys - dy_hats
        r_resid = numpy.sqrt(dxResid**2+dyResid**2)
        rms = numpy.sqrt(numpy.mean(r_resid**2))
        median = numpy.median(r_resid)
        perc95 = numpy.percentile(r_resid, 95)

        plt.figure(figsize=(8,8))
        plt.title("MJD: %i  centType: %s  simpleSigma: %i\nModel Residuals"%(mjd, centType, simpleSigma))
        q = plt.quiver(xs,ys,dxResid,dyResid, angles="xy", units="xy", alpha=0.5, width=2, scale=0.005)
        ax = plt.gca()
        ax.quiverkey(q, 0.85, 0.85, 0.5, "units: pixels\nRMS: %.2f\nMedian: %.2f\nperc95: %.2f\nquiver length: 0.5"%(rms,med,perc95))
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
        ax.set_aspect("equal")
        plt.savefig("residualDistortion_%i_%s_%i.png"%(mjd, centType, simpleSigma), dpi=250)

        plt.close("all")


def compareNudgeModels(clip=0.75):
    # clip is units of pixels
    df = pandas.read_csv("rawMeas_comp.csv")

    ny = 6132
    nx = 6110

    x = numpy.arange(nx, step=10)
    y = numpy.arange(ny, step=10)
    xx, yy = numpy.meshgrid(x,y)
    xs = xx.flatten()
    ys = yy.flatten()

    modelDict = {}

    for mjd in cmap.keys():

        _df = df[df.mjd==mjd]
        # _df = _df[_df.positionerID < 0]
        xCents = _df.x.to_numpy()/10
        yCents = _df.y.to_numpy()/10

        print("mjd", mjd)
        modelDict[mjd] = {}

        beta_x = numpy.load("beta_x_%i_sep_1.npy"%mjd)
        beta_y = numpy.load("beta_y_%i_sep_1.npy"%mjd)
        dxs,dys = applyDistortion(xs,ys,beta_x,beta_y)

        dxs[dxs<-1*clip] = -1*clip
        dys[dys<-1*clip] = -1*clip

        dxs[dxs>clip] = clip
        dys[dys>clip] = clip

        # dxs[numpy.logical_and(dxs>-clip, dxs<clip)] = 0
        # dys[numpy.logical_and(dys>-clip, dys<clip)] = 0

        # plt.figure()
        # plt.hist(dxs, bins=500)
        # plt.title("%i dxs"%mjd)

        # plt.figure()
        # plt.hist(dys, bins=500)
        # plt.title("%i dys"%mjd)

        modelDict[mjd]["dxs"] = dxs
        modelDict[mjd]["dys"] = dys

        plt.figure(figsize=(8,8))
        plt.imshow(dxs.reshape(xx.shape), cmap="seismic", origin="lower")
        # plt.plot(xCents,yCents,'x', color="cyan")

        plt.title("%i dxs"%mjd)
        plt.colorbar()
        plt.savefig("nudge_viz_dx_%i.png"%mjd, dpi=250)

        plt.figure(figsize=(8,8))
        plt.imshow(dys.reshape(xx.shape), cmap="seismic", origin="lower")
        # plt.plot(xCents,yCents,'x', color="cyan")
        plt.title("%i dys"%mjd)
        plt.colorbar()
        plt.savefig("nudge_viz_dy_%i.png"%mjd, dpi=250)


def applyNudgeModel():

    dfList = []
    for mjd, dd in cmap.items():
        for config, (minImg, maxImg) in dd.items():
            imgNums = list(range(minImg, maxImg+1))
            for imgNum in imgNums:
                beta_x = numpy.load("beta_x_%i_sep_1.npy"%mjd)
                beta_y = numpy.load("beta_y_%i_sep_1.npy"%mjd)
                dfList.append(extractData(imgNum, mjd, config, reprocess=True, centType="nudge", beta_x=beta_x, beta_y=beta_y, polids=numpy.arange(33)))

            # _func = partial(extractData, mjd=mjd, configid=config, reprocess=reprocess)
            # p = Pool(2)
            # _dfList = p.map(_func, imgNums)
            # dfList.extend(_dfList)

            # for imgNum in range(minImg, maxImg+1):
            #     print("on mjd, img", mjd, imgNum)
            #     _df = extractData(imgNum, mjd, config, reprocess=reprocess)
            #     _df["mjd"] = mjd
            #     dfList.append(_df)

    df = pandas.concat(dfList)
    df.to_csv("fitMeas_comp.csv")


def plotEndResult():
    df = pandas.read_csv("fitMeas_comp.csv")

    # remove fiducials
    # df = df[df.positionerID >= 0]

    df_mean = df_mean = df.groupby(["positionerID", "config", "mjd", "centType", "simpleSigma"]).mean().reset_index()
    df = df.merge(df_mean, on=["positionerID", "config", "mjd", "centType", "simpleSigma"], suffixes=(None, "_mean"))

    df["dxWok"] = df.xWokMeasMetrology - df.xWokMeasMetrology_mean
    df["dyWok"] = df.yWokMeasMetrology - df.yWokMeasMetrology_mean
    df["drWok"] = numpy.sqrt(df.dxWok**2+df.dyWok**2)

    for mjd in cmap.keys():
        _df = df[df.mjd==mjd]
        # xs = _df.xWokMeasMetrology.to_numpy()
        # ys = _df.yWokMeasMetrology.to_numpy()
        xs = _df.x.to_numpy()
        ys = _df.y.to_numpy()
        dxs = _df.dxWok.to_numpy()
        dys = _df.dyWok.to_numpy()

        dxs, dys = rotScaleWok2pix(
            _df.dxWok.to_numpy(), _df.dyWok.to_numpy(),
            _df.rot.to_numpy(), 1
        )

        drs = numpy.sqrt(dxs**2+dys**2)

        plt.figure()
        plt.hist(drs, bins=200)

        rms = numpy.sqrt(numpy.mean(_df.drWok**2))
        med = numpy.median(_df.drWok)
        perc95 = numpy.percentile(_df.drWok, 95)

        plt.figure(figsize=(8,8))
        plt.title("MJD: %i\nFull Fit incl. ZB's"%(mjd))
        q = plt.quiver(xs,ys,dxs,dys, angles="xy", units="xy", alpha=0.5, width=1, scale=0.0005)
        ax = plt.gca()
        ax.quiverkey(q, 0.85, 0.85, 0.01, "units: mm\nRMS: %.4f\nMedian: %.4f\nperc95: %.4f\nquiver length: 0.01"%(rms,med,perc95))
        ax.set_xlabel("x (CCD)")
        ax.set_ylabel("y (CCD)")
        ax.set_aspect("equal")
        plt.savefig("finalMeas_%i.png"%(mjd), dpi=250)

if __name__ == "__main__":
    # compileData(reprocess=False)
    # _plotImgQuality()
    # measureMeanDistortion(includeFIFs=True, useZB=False, saveCoeffs=True)
    compareNudgeModels()
    applyNudgeModel()
    plotEndResult()
    # plt.show()


    # plt.show()

