from coordio.utils import fitsTableToPandas
from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY
from astropy.io import fits
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import drop_fields
import pandas
import numpy
import seaborn as sns
from datetime import datetime
from pydl.pydlutils.yanny import yanny
import glob
from coordio.defaults import calibration
import seaborn as sns

# uses holtz' new dither.fits tables
# same as holtzReduce2 except fit the inner zone

# ffmpeg -r 10 -f image2 -i b1-%04d.png -pix_fmt yuv420p b1.mp4


def plotBetaOffsets(df, draCol, ddecCol, pngDir="figs", title=""):
    df = df.copy()
    alpha = df.alpha.to_numpy()
    beta = df.beta.to_numpy()
    alphaOff = df.alphaOffset.to_numpy()
    betaOff = df.betaOffset.to_numpy()
    totalRot = numpy.radians(alpha+beta+alphaOff+betaOff-90)
    df["totalRot"] = totalRot
    df["rot"] = numpy.round(numpy.degrees(totalRot)%360, decimals=-1)
    dra = df[draCol].to_numpy()
    ddec = df[ddecCol].to_numpy()

    dra_m = dra - numpy.nanmean(dra)
    ddec_m = ddec - numpy.nanmean(ddec)

    dxbeta = numpy.cos(totalRot)*dra + numpy.sin(totalRot)*ddec
    dybeta = -numpy.sin(totalRot)*dra + numpy.cos(totalRot)*ddec

    dxbeta_m = numpy.cos(totalRot)*dra_m + numpy.sin(totalRot)*ddec_m
    dybeta_m = -numpy.sin(totalRot)*dra_m + numpy.cos(totalRot)*ddec_m

    df["dxbeta"] = dxbeta
    df["dybeta"] = dybeta
    df["drBeta"] = numpy.sqrt(dxbeta**2+dybeta**2)
    df["dThetaBeta"] = numpy.degrees(numpy.arctan2(dybeta, dxbeta))

    df["dxbeta_m"] = dxbeta_m
    df["dybeta_m"] = dybeta_m
    df["drBeta_m"] = numpy.sqrt(dxbeta_m**2+dybeta_m**2)
    df["dThetaBeta_m"] = numpy.degrees(numpy.arctan2(dybeta_m, dxbeta_m))

    radialOff = numpy.array([])
    stdx = []
    stdy = []
    nConfig = []

    figCounter = 0
    fiberIDs = list(set(df.fiber))
    for fiber in fiberIDs:
        fig, axs = plt.subplots(1,1,figsize=(10,10))
        axs = [axs]
        # axs = axs.flatten()
        _ddf = df[df.fiber==fiber]
        nConfig.append(len(set(_ddf.design_id)))

        dxMean = numpy.nanmean(_ddf.dxbeta)
        dyMean = numpy.nanmean(_ddf.dybeta)
        dxCenter = _ddf.dxbeta - dxMean
        dyCenter = _ddf.dybeta - dyMean
        stdx.append(numpy.std(dxCenter))
        stdy.append(numpy.std(dyCenter))
        radialOff = numpy.hstack((radialOff, numpy.sqrt(dxCenter**2+dyCenter**2)))
        fig.suptitle(title + " fiber " + str(fiber))
        sns.scatterplot(x="dxbeta", y="dybeta", hue="rot", ax=axs[0], data=_ddf)
        axs[0].axhline(dyMean, color="red")
        axs[0].axvline(dxMean, color="red")
        axs[0].set_aspect("equal")
        axs[0].set_xlim([-2,2])
        axs[0].set_ylim([-2,2])
        axs[0].set_xlabel("xBeta (arcsec)")
        axs[0].set_ylabel("yBeta (arcsec)")
        axs[0].grid()

        # sns.scatterplot(x="drBeta", y="dThetaBeta", hue="totalRotRound", ax=axs[1], data=_ddf)
        # axs[1].grid()
        # axs[1].set_ylim([-180, 180])
        # axs[1].set_xlim([0, 2])

        # sns.scatterplot(x="dxbeta_m", y="dybeta_m", hue="totalRotRound", ax=axs[2], data=_ddf)
        # axs[2].set_aspect("equal")
        # axs[2].set_xlim([-2,2])
        # axs[2].set_ylim([-2,2])
        # axs[2].grid()

        # sns.scatterplot(x="drBeta_m", y="dThetaBeta_m", hue="totalRotRound", ax=axs[3], data=_ddf)
        # axs[3].grid()
        # axs[3].set_ylim([-180, 180])
        # axs[3].set_xlim([0, 2])


        strNum = ("%i"%figCounter).zfill(4)
        fig.savefig("%s/boff-%s.png"%(pngDir, strNum))


        figCounter += 1
        # plt.show()

        # import pdb; pdb.set_trace()
        plt.close("all")

    return radialOff, numpy.array(stdx), numpy.array(stdy), numpy.array(nConfig)


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
    df["mjd"] = int(yf["MJD"])
    df["observatory"] = yf["observatory"]
    df["raCen"] = float(yf["raCen"])
    df["decCen"] = float(yf["decCen"])
    df["filename"] = ff
    try:
        df["fvc_image_path"] = yf["fvc_image_path"]
    except:
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


def merge():

    ff = fits.open("dither-bossr1.fits")[1].data
    ff = drop_fields(ff, "fit")

    df = fitsTableToPandas(ff)

    configs = [
        5316, 5146, 5167, 5192, 5287, 5300, 5315, 5260, 5288, 5148,
        5169, 5194, 5302, 5301, 5319, 5286, 5144, 5165, 5189, 5259
    ]

    dfList = [parseConfSummary("configs/confSummary-%i.par"%x) for x in configs]

    dfCF = pandas.concat(dfList)

    dfCF = dfCF[dfCF.fiberType == "BOSS"]
    dfCF = dfCF[dfCF.on_target == True]
    dfCF["fiber"] = dfCF["fiberId"]
    df = df.merge(dfCF, on=["field_id", "mjd", "fiber"], suffixes=(None, "_r"))

    pt = calibration.positionerTable.reset_index()
    pt = pt[["positionerID", "alphaOffset", "betaOffset"]]
    pt["positionerId"] = pt["positionerID"]

    df = df.merge(pt, on="positionerId")

    df.to_csv("holtz2merged.csv")

# merge()

df = pandas.read_csv("holtz2merged.csv")
# import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()

dxCol = "fit_dx_res"
dyCol = "fit_dy_res"

# dxCol = "desi_xfiboff"
# dyCol = "desi_yfiboff"

df["roff"] = numpy.sqrt(df[dxCol]**2 + df[dyCol]**2)
df = df[df.roff < 1.75]

rms = numpy.sqrt(numpy.mean(df.roff**2))
perc90 = numpy.percentile(df.roff, 90)

plt.figure(figsize=(10,10))
plt.quiver(df.xwok, df.ywok, df[dxCol], df[dyCol], angles="xy", units="xy", width=0.5, scale=.05)
plt.axis("equal")
plt.title("Measured Offsets\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.xlabel("x wok (mm)")
plt.ylabel("y wok (mm)")
plt.savefig("raw_quiver3.png", dpi=250)

plt.figure()
bins = numpy.linspace(0,3,100)
plt.hist(df.roff, bins=bins)
plt.title("Measured Offsets\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.xlabel("dr (arcsec)")
plt.savefig("raw_hist3.png", dpi=250)

# plotBetaOffsets(df, "desi_xfiboff", "desi_yfiboff")

# import pdb; pdb.set_trace()

# plotBetaOffsets(df, "desi_xfiboff", "desi_yfiboff", title="pre wok correct: ")

betaResids, stdx1, stdy1, nConfig = plotBetaOffsets(df, dxCol, dyCol, title="pre wok correct: ")
keep = nConfig >= 3
stdx1 = stdx1[keep]
stdy1 = stdy1[keep]

rms = numpy.sqrt(numpy.mean(betaResids**2))
perc90 = numpy.percentile(betaResids, 90)
plt.figure()
bins = numpy.linspace(0,3,100)
plt.hist(betaResids, bins=bins)
plt.title("Beta off removed\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.xlabel("dr (arcsec)")
plt.savefig("beta_removed3.png", dpi=250)

# try to fit the pattern? in focal coords now but needs to be in wok coords later?
x = df.xwok.to_numpy()
y = df.ywok.to_numpy()
xp = x + df[dxCol].to_numpy()
yp = y + df[dyCol].to_numpy()

# numpy.random.seed(42)
# rands = numpy.random.uniform(size=len(x))
# train_inds = rands <= 0.8
# test_inds = rands > 0.8


# polids, coeffs = fitZhaoBurge(x[train_inds],y[train_inds],xp[train_inds],yp[train_inds], polids=numpy.arange(33))
# dx_fit, dy_fit = getZhaoBurgeXY(polids, coeffs, x[test_inds], y[test_inds])
# dx_test = df[dxCol].to_numpy()[test_inds]
# dy_test = df[dyCol].to_numpy()[test_inds]
# resid_xtest = dx_test - dx_fit
# resid_ytest = dy_test - dy_fit
# resid_rtest = numpy.sqrt(resid_xtest**2+resid_ytest**2)
# print("rms test error", numpy.sqrt(numpy.mean(resid_rtest**2)))

# now fit everything
polids, coeffs = fitZhaoBurge(x,y,xp,yp, polids=numpy.arange(33))
dx_fit, dy_fit = getZhaoBurgeXY(polids, coeffs, x, y)

# # these are hardcoded in the FVCTransformAPO
# zbCoeffs2 = numpy.array([
#     3.30790633e-01, 4.13533378e-01, 1.71951213e-03, -9.90141758e-04,
#     -4.48460907e-04, -3.68228879e-06, -2.45423918e-06, 3.06249618e-07,
#     1.52595326e-06, -1.73037375e-08, 9.09750974e-09, 1.40858150e-08,
#     -8.69986544e-11, 2.25009469e-09, 5.94658320e-12, 5.67065198e-12,
#     -1.99111356e-12, 6.33175666e-13, -2.82452951e-12, 1.94818210e-12,
#     4.03105186e-14, -2.79909483e-14, -1.46249329e-14, -5.04644513e-15,
#     1.41549916e-15, 1.27613775e-15, -8.70792948e-15, 2.00843998e-04,
#     -1.05007864e-06, 2.48271534e-07, -9.28099433e-10, 6.13172886e-10,
#     -1.31799608e-09
# ])

# dx_fit2, dy_fit2 = getZhaoBurgeXY(polids, zbCoeffs2, x, y)
# dr = numpy.sqrt((dx_fit2-dx_fit)**2+(dy_fit2-dy_fit)**2)

# plt.figure()
# plt.hist(dr,bins=100)
# print("max diff", numpy.max(dr)*1000)

print("polids", polids)
print("coeffs", coeffs)

print("mean offsets", numpy.mean(dx_fit), numpy.mean(dy_fit))

plt.figure(figsize=(10,10))
plt.quiver(df.xwok, df.ywok, dx_fit, dy_fit, angles="xy", units="xy", width=0.5, scale=.05)
plt.axis("equal")
plt.xlabel("x wok (mm)")
plt.ylabel("y wok (mm)")
plt.title("model")

plt.figure(figsize=(10,10))
plt.quiver(df.xwok, df.ywok, df[dxCol], df[dyCol], angles="xy", units="xy", width=0.5, scale=.05)
plt.axis("equal")
plt.xlabel("x wok (mm)")
plt.ylabel("y wok (mm)")
plt.title("data")


xNew = df.xwok.to_numpy() + dx_fit
yNew = df.ywok.to_numpy() + dy_fit

plt.figure(figsize=(10,10))
plt.quiver(xNew,yNew,xNew-xp,yNew-yp, angles="xy", units="xy", width=0.5, scale=.05)
plt.axis("equal")
plt.xlabel("x wok (mm)")
plt.ylabel("y wok (mm)")
plt.title("residual")

r = numpy.sqrt(xNew**2+yNew**2)
keep = r<100
_xNew = xNew[keep]
_yNew = yNew[keep]
_xpNew = xp[keep]
_ypNew = yp[keep]

polids, coeffs = fitZhaoBurge(_xNew,_yNew,_xpNew,_ypNew, polids=numpy.arange(33))
print("coeffs2")
print(coeffs)
dx_fit, dy_fit = getZhaoBurgeXY(polids, coeffs, _xNew, _yNew)

xNew[keep] += dx_fit
yNew[keep] += dy_fit

plt.figure(figsize=(10,10))
plt.quiver(xNew,yNew,xNew-xp,yNew-yp, angles="xy", units="xy", width=0.5, scale=.05)
plt.axis("equal")
plt.xlabel("x wok (mm)")
plt.ylabel("y wok (mm)")
plt.title("residual2")


# dx = xNew - xp
# dy = yNew - yp

# r = numpy.sqrt(x**2+y**2)
# dx = df[dxCol].to_numpy()
# dy = df[dyCol].to_numpy()
# ddx = dx_fit - dx
# ddy = dy_fit - dy



# # next fit the residuals where rWok < 100
# keep = r<=100
# _x = x[keep]
# _y = y[keep]
# _dx = ddx[keep]
# _dy = ddy[keep]
# _xp = _x + _dx
# _yp = _y + _dy

# plt.figure(figsize=(10,10))
# plt.quiver(_x,_y,_dx,_dy, angles="xy", units="xy", width=0.5, scale=.05)
# plt.axis("equal")
# plt.xlabel("x wok (mm)")
# plt.ylabel("y wok (mm)")
# plt.title("residual zoom")

# # polids = numpy.array([0,1,2,3,4,5,6,9,20,27,28,29,30], dtype=int)
# polids, coeffs = fitZhaoBurge(_x,_y,_xp,_yp, polids=None)
# dx_fit, dy_fit = getZhaoBurgeXY(polids, coeffs, _x, _y)

# plt.figure(figsize=(10,10))
# plt.quiver(_x,_y,dx_fit-_dx,dy_fit-_dy, angles="xy", units="xy", width=0.5) #, scale=.05)
# plt.axis("equal")
# plt.xlabel("x wok (mm)")
# plt.ylabel("y wok (mm)")
# plt.title("residual zoom residual")

# rErr = numpy.mean(numpy.sqrt(_dx**2+_dy**2))
# rErr2 = numpy.mean(numpy.sqrt((_dx-dx_fit)**2+(_dy-dy_fit)**2))
# print("error middle", rErr, rErr2)

# plt.title("ZB fit corrected\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
# plt.savefig("fit_quiver3.png", dpi=250)

# plt.figure()
# plt.hist(numpy.sqrt(dx_fit**2+dy_fit**2), bins=numpy.linspace(0,2,40))

plt.show()
import pdb; pdb.set_trace()


resid_x = df[dxCol].to_numpy() - dx_fit
resid_y = df[dyCol].to_numpy() - dy_fit
resid_r = numpy.sqrt(resid_x**2+resid_y**2)
df["dxZBFit"] = resid_x
df["dyZBFit"] = resid_y


rms = numpy.sqrt(numpy.mean(resid_r**2))
perc90 = numpy.percentile(resid_r, 90)

plt.figure(figsize=(10,10))
plt.quiver(df.xwok, df.ywok, resid_x, resid_y, angles="xy", units="xy", width=0.5, scale=.05)
plt.axis("equal")
plt.xlabel("x wok (mm)")
plt.ylabel("y wok (mm)")
plt.title("ZB fit corrected\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.savefig("fit_quiver3.png", dpi=250)



plt.figure()
bins = numpy.linspace(0,3,100)
plt.hist(resid_r, bins=bins)
plt.xlabel("dr (arcsec)")
plt.title("ZB fit corrected\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.savefig("fit_hist3.png", dpi=250)



# print("median radial error ZB fit", numpy.median(resid_r))
# print("rms error ZB fit", numpy.sqrt(numpy.mean(resid_r**2)))

# plotBetaOffsets(df, "dxZBFit", "dyZBFit")
betaResids, stdx2, stdy2, nConfig = plotBetaOffsets(df, "dxZBFit", "dyZBFit", pngDir="figsFit", title="post wok correct: ")
keep = nConfig >= 3
stdx2 = stdx2[keep]
stdy2 = stdy2[keep]

rms = numpy.sqrt(numpy.mean(betaResids**2))
perc90 = numpy.percentile(betaResids, 90)


plt.figure()
bins = numpy.linspace(0,3,100)
plt.hist(betaResids, bins=bins)
plt.title("Fit + Beta off removed\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.xlabel("dr (arcsec)")
plt.savefig("fit_beta_removed3.png", dpi=250)

plt.figure(figsize=(10,10))
plt.plot(stdx1, stdy1, '.', color="tab:orange", label="orig")
plt.plot(stdx2, stdy2, '.', color="tab:blue", label="ZB fit corrected")
sns.kdeplot(x=stdx1, y=stdy1, color="tab:orange", levels=3, ax=plt.gca())
sns.kdeplot(x=stdx2, y=stdy2, color="tab:blue", levels=3, ax=plt.gca())
plt.xlabel("std x beta (arcsec)")
plt.ylabel("std y beta (arcsec)")
plt.legend()
plt.axis("equal")
plt.title("beta frame scatter\n(fibers w/ >= 3 designs)")
plt.savefig("beta_scatter3.png", dpi=250)


# plt.show()







