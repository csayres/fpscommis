import pandas
from coordio.utils import fitsTableToPandas
from coordio.transforms import FVCTransformLCO
from astropy.io import fits
import numpy
from skimage.transform import SimilarityTransform
import matplotlib.pyplot as plt
import time
import seaborn as sns

# data from commissioning
# MJD = 59807
# baseDir = "/Volumes/futa/lco/data/fcam/%i"%MJD
# # configid : [minImg, maxImg] map
# cmap = {
#     1: [7,116],
#     2: [117,226],
#     3: [227,336]
# }

# data after baffle re-installed and rotated
MJD = 60174
baseDir = "/Volumes/futa/lco/data/fcam/%i"%MJD
# configid : [minImg, maxImg] map
cmap = {
    1: [87,152],
    2: [153,218],
    3: [219,284]
}

IMAX = 32
# IMAX = 44 # maximum integer wave number
DELTAK = 2. * numpy.pi / 10000.0 # wave number spacing in inverse pixels
SAVE_COEFFS = False


def getImgFile(imgNum):
    imgNumStr = str(imgNum).zfill(4)
    return baseDir + "/proc-fimg-fvc1s-%s.fits"%imgNumStr


def extractData(configid, imgNum, reprocess=False):
    ff = fits.open(getImgFile(imgNum))
    if reprocess:
        imgData = ff[1].data
        pc = fitsTableToPandas(ff["POSANGLES"].data)
        fvct = FVCTransformLCO(
            imgData, pc, ff[1].header["IPA"]
        )
        fvct.extractCentroids()
        fvct.fit(centType="sep")
        ptm = fvct.positionerTableMeas

    else:
        ptm = fitsTableToPandas(ff["POSITIONERTABLEMEAS"].data)
    trans = numpy.array([ff[1].header["FVC_TRAX"], ff[1].header["FVC_TRAY"]])
    scale = ff[1].header["FVC_SCL"]
    rot = ff[1].header["FVC_ROT"]
    ipa = numpy.around(ff[1].header["IPA"], decimals=1)
    # import pdb; pdb.set_trace()
    # ptm = ptm[ptm.wokErrWarn] # remove bogus measurements (missing spots)
    ptm = ptm[["positionerID", "x", "y", "xWokReportMetrology", "yWokReportMetrology", "xWokMeasMetrology", "yWokMeasMetrology"]]

    ptm["config"] = configid
    ptm["imgNum"] = imgNum
    ptm["transx"] = trans[0]
    ptm["transy"] = trans[1]
    ptm["rot"] = numpy.radians(rot)
    ptm["scale"] = scale
    ptm["ipa"] = ipa

    return ptm.reset_index()


def compileData():
    dfList = []
    for config, (minImg, maxImg) in cmap.items():
        for imgNum in range(minImg, maxImg+1):
            print("on img", imgNum)
            dfList.append(extractData(config, imgNum))

    df = pandas.concat(dfList)
    df.to_csv("rawMeas.csv")


def measureMeanDistortion():
    df = pandas.read_csv("rawMeas.csv")
    dx = df.xWokReportMetrology - df.xWokMeasMetrology
    dy = df.yWokReportMetrology - df.yWokMeasMetrology
    dr = numpy.sqrt(dx**2+dy**2)

    df["dxWok"] = dx
    df["dyWok"] = dy
    df["drWok"] = dr

    df = df[df.drWok < 1]

    df_mean = df.groupby(["positionerID", "config"]).mean().reset_index()
    df_meanRot = df.groupby(["positionerID", "config", "ipa"]).mean().reset_index()
    df = df.merge(df_mean, on=["positionerID", "config"], suffixes=(None, "_mean"))
    df = df.merge(df_meanRot, on=["positionerID", "config", "ipa"], suffixes=(None, "_meanRot"))

    dxRotMarg = df.xWokMeasMetrology - df.xWokMeasMetrology_mean
    dyRotMarg = df.yWokMeasMetrology - df.yWokMeasMetrology_mean
    drRotMarg = numpy.sqrt(dxRotMarg**2+dyRotMarg**2)

    dxRotAll = df.xWokMeasMetrology - df.xWokMeasMetrology_meanRot
    dyRotAll = df.yWokMeasMetrology - df.yWokMeasMetrology_meanRot
    drRotAll = numpy.sqrt(dxRotAll**2+dyRotAll**2)

    plt.figure()
    plt.hist(drRotMarg, bins=200, histtype="step", color="red", label="all rot angles")
    plt.hist(drRotAll, bins=200, histtype="step", color="blue", label="one rot angle")
    plt.xlim([-0.01, 0.15])
    plt.legend()

    plt.savefig("hist_variation.png", dpi=250)
    plt.close("all")
    # average over multiple images at each rotator position
    #df = df_meanRot.merge(df_mean, on=["positionerID", "config"], suffixes=(None, "_mean"))


    # import pdb; pdb.set_trace()

    dxWokMeas = df.xWokMeasMetrology - df.xWokMeasMetrology_mean
    dyWokMeas = df.yWokMeasMetrology - df.yWokMeasMetrology_mean
    drWokMeas = numpy.sqrt(dxWokMeas**2+dyWokMeas**2)

    # convert dxyWokMeas into dxyCCDMeas
    cosTheta = numpy.cos(df.rot)
    sinTheta = numpy.sin(df.rot)

    dxCCDMeas = (dxWokMeas*cosTheta + dyWokMeas*sinTheta)/df.scale
    dyCCDMeas = (-dxWokMeas*sinTheta + dyWokMeas*cosTheta)/df.scale

    df["dxCCDMeas"] = dxCCDMeas
    df["dyCCDMeas"] = dyCCDMeas

    plt.figure(figsize=(8,8))
    q = plt.quiver(df.x, df.y, df.dxCCDMeas, df.dyCCDMeas, angles="xy", units="xy", width=3, scale=0.002)
    ax = plt.gca()
    ax.quiverkey(q, 0.9, 0.9, 0.3, "0.3 pix")
    ax.set_xlabel("xCCD (pix)")
    ax.set_ylabel("yCCD (pix)")
    # ax.set_xlim([1000,7000])
    # ax.set_ylim([0,6000])
    ax.set_aspect("equal")

    meanDistortion = df[["x", "y", "dxCCDMeas", "dyCCDMeas"]]
    meanDistortion.to_csv("meanDistortion.csv")

    plt.savefig("distorted_quiver.png", dpi=250)
    plt.close("all")
    # plt.show()


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


def fitDistortion(save_coeffs=True):
    meanDistortion = pandas.read_csv("meanDistortion.csv")

    xs = meanDistortion.x.to_numpy()
    ys = meanDistortion.y.to_numpy()
    dxs = meanDistortion.dxCCDMeas.to_numpy()
    dys = meanDistortion.dyCCDMeas.to_numpy()

    xm = xs - numpy.mean(xs)
    ym = ys - numpy.mean(ys)
    r = numpy.sqrt(xm**2+ym**2)
    rmm = r*120/1000

    print("max r", numpy.max(r)*120/1000.)

    t1 = time.time()
    X = design_matrix(xs, ys)
    print("design_matrix took", time.time()-t1)
    print("X.shape", X.shape)

    n, p = X.shape

    numpy.random.seed(42)
    rands = numpy.random.uniform(size=n)

    train = rands <= 0.8
    test = rands > 0.8

    # train = rands <= 1
    # test = rands <=1
    print(numpy.sum(train), numpy.sum(test))

    t1 = time.time()
    beta_x, resids, rank, s = numpy.linalg.lstsq(X[train], dxs[train], rcond=None)
    if save_coeffs:
        with open("beta_x.npy", "wb") as f:
            numpy.save(f, beta_x)

    print("fit took", time.time()-t1)
    dxs_hat = X[test] @ beta_x
    print(rank, min(s), max(s))

    print("original dx (test set) RMS:", numpy.sqrt(numpy.mean(dxs[test] ** 2)))
    print("original dx (test set) MAD:", numpy.sqrt(numpy.median(dxs[test] ** 2)))
    print("dx - dx_hat (test set) RMS:", numpy.sqrt(numpy.mean((dxs[test] - dxs_hat) ** 2)))
    print("dx - dx_hat (test set) MAD:", numpy.sqrt(numpy.median((dxs[test] - dxs_hat) ** 2)))

    t1 = time.time()
    beta_y, resids, rank, s = numpy.linalg.lstsq(X[train], dys[train], rcond=None)
    print("fit took", time.time()-t1)
    dys_hat = X[test] @ beta_y
    print(rank, min(s), max(s))

    if save_coeffs:
        with open("beta_y.npy", "wb") as f:
            numpy.save(f, beta_y)

    print("original dy (test set) RMS:", numpy.sqrt(numpy.mean(dys[test] ** 2)))
    print("original dy (test set) MAD:", numpy.sqrt(numpy.median(dys[test] ** 2)))
    print("dy - dy_hat (test set) RMS:", numpy.sqrt(numpy.mean((dys[test] - dys_hat) ** 2)))
    print("dy - dy_hat (test set) MAD:", numpy.sqrt(numpy.median((dys[test] - dys_hat) ** 2)))

    dxs_hat = X @ beta_x
    dys_hat = X @ beta_y

    resid_dx = dxs - dxs_hat
    resid_dy = dys - dys_hat
    resid_r = numpy.sqrt(resid_dx**2+resid_dy**2)
    rs = numpy.sqrt(dxs**2+dys**2)

    fit, ax = plt.subplots(1,1, figsize=(10,10))
    q = ax.quiver(xs, ys, dxs, dys, angles="xy", units="xy", width=2, scale=0.01)
    ax.quiverkey(q, 0.9, 0.9, 0.5, "0.5 pix")
    ax.set_xlabel("x CCD (pix)")
    ax.set_ylabel("y CCD (pix)")
    plt.axis("equal")
    plt.savefig("xy_vs_dxy.png", dpi=250)

    fit, ax = plt.subplots(1,1, figsize=(10,10))
    q = ax.quiver(xs, ys, resid_dx, resid_dy, angles="xy", units="xy", width=2, scale=0.01)
    ax.quiverkey(q, 0.9, 0.9, 0.5, "0.5 pix")
    ax.set_xlabel("x CCD (pix)")
    ax.set_ylabel("y CCD (pix)")
    plt.title("fit")
    plt.axis("equal")
    plt.savefig("xy_vs_xyresid.png", dpi=250)

    plt.figure()
    plt.hist(rs, bins=numpy.linspace(0,1,500), histtype="step", color="red", label="orig")
    plt.hist(resid_r, bins=numpy.linspace(0,1,500), histtype="step", color="blue", label="resid")
    plt.xlabel("pixels")
    plt.xlim([-0.01, 1])
    plt.legend()

    plt.savefig("model_hist.png", dpi=250)
    plt.close("all")

    # plt.show()


def verifyDistortion():
    dfList = []
    for config, (minImg, maxImg) in cmap.items():
        for imgNum in range(minImg, maxImg+1):
            print("on img", imgNum)
            dfSep = extractData(config, imgNum, False)
            dfSep["centType"] = "sep"
            dfNudge = extractData(config, imgNum, True)
            dfNudge["centType"] = "nudge"
            dfList.append(dfSep)
            dfList.append(dfNudge)
    df = pandas.concat(dfList)
    df.to_csv("verifyNudge.csv")


def plotDistortion():
    df = pandas.read_csv("verifyNudge.csv")
    dxBlind = df.xWokReportMetrology - df.xWokMeasMetrology
    dyBlind = df.xWokReportMetrology - df.xWokMeasMetrology
    drBlind = numpy.sqrt(dxBlind**2+dyBlind**2)

    df["dxBlind"] = dxBlind
    df["dyBlind"] = dyBlind
    df["drBlind"] = drBlind

    df_mean = df.groupby(["positionerID", "config", "centType"]).mean()

    df = df.merge(df_mean, on=["positionerID", "config", "centType"], suffixes=(None, "_m"))

    dx = df.xWokMeasMetrology - df.xWokMeasMetrology_m
    dy = df.yWokMeasMetrology - df.yWokMeasMetrology_m
    dr = numpy.sqrt(dx**2+dy**2)

    df["dx"] = dx
    df["dy"] = dy
    df["dr"] = dr


    plt.figure()
    sns.histplot(x="drBlind", hue="centType", element="step", data=df)
    plt.xlim([-0.01, .1])
    plt.savefig("drBlind.png", dpi=250)

    plt.figure()
    sns.histplot(x="dr", hue="centType", element="step", data=df)
    plt.xlim([-0.01, .1])
    plt.savefig("dr.png", dpi=250)

    rSep = df[df.centType=="sep"]["dr"]
    rNudge = df[df.centType=="nudge"]["dr"]
    print("sep", numpy.median(rSep), numpy.mean(rSep), numpy.percentile(rSep, 0.9))
    print("nudge", numpy.median(rNudge), numpy.mean(rNudge), numpy.percentile(rNudge, 0.9))

    # plt.show()
    plt.close("all")



if __name__ == "__main__":
    compileData()
    measureMeanDistortion()
    fitDistortion(True)
    # verifyDistortion()
    # plotDistortion()
