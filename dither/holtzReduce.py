from coordio.utils import fitsTableToPandas
from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY
from astropy.io import fits
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import drop_fields
import pandas
import numpy
import seaborn as sns

# ffmpeg -r 10 -f image2 -i b1-%04d.png -pix_fmt yuv420p b1.mp4


def plotBetaOffsets(df, draCol, ddecCol, pngDir="figs", title=""):
    df = df.copy()
    alpha = df.alpha.to_numpy()
    beta = df.beta.to_numpy()
    totalRot = numpy.radians(alpha+beta)
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

    radialOff = []

    for _camera in [b"b1", b"APOGEE"]:
        camera = _camera.decode()
        figCounter = 0
        _df = df[(df.camera==_camera)]
        fiberIDs = list(set(_df.fiber))
        for fiber in fiberIDs:
            fig, axs = plt.subplots(1,1,figsize=(10,10))
            axs = [axs]
            # axs = axs.flatten()
            _ddf = _df[_df.fiber==fiber]
            dxMean = numpy.nanmean(_ddf.dxbeta)
            dyMean = numpy.nanmean(_ddf.dybeta)
            radialOff.append(numpy.sqrt(dxMean**2+dyMean**2))
            fig.suptitle(title + str(camera) + " fiber " + str(fiber))
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
            fig.savefig("%s/%s-%s.png"%(pngDir, camera, strNum))


            figCounter += 1
            # plt.show()

            # import pdb; pdb.set_trace()
            plt.close("all")

    return radialOff



ff = fits.open("dither.fits")[1].data
ff = drop_fields(ff, "fit")
ff = drop_fields(ff, "r1")



df = fitsTableToPandas(ff)
# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()

# dxCol = "fit_dx_res"
# dyCol = "fit_dy_res"

dxCol = "desi_xfiboff"
dyCol = "desi_yfiboff"

df["roff"] = numpy.sqrt(df[dxCol]**2 + df[dyCol]**2)

# plotBetaOffsets(df, "desi_xfiboff", "desi_yfiboff")

# import pdb; pdb.set_trace()

df = df[df.roff < 2]

# plotBetaOffsets(df, "desi_xfiboff", "desi_yfiboff", title="pre wok correct: ")

plotBetaOffsets(df, dxCol, dyCol, title="pre wok correct: ")


rms = numpy.sqrt(numpy.mean(df.roff**2))
perc90 = numpy.percentile(df.roff, 90)

plt.figure(figsize=(10,10))
plt.quiver(df.xfocal, df.yfocal, df[dxCol], df[dyCol], angles="xy", units="xy", width=0.5, scale=.02)
plt.axis("equal")
plt.title("Measured Offsets\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.xlabel("x focal (mm)")
plt.ylabel("x focal (mm)")
plt.savefig("raw_quiver.png", dpi=250)

plt.figure()
bins = numpy.linspace(0,3,100)
plt.hist(df.roff, bins=bins)
plt.title("Measured Offsets\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.xlabel("dr (arcsec)")
plt.savefig("raw_hist.png", dpi=250)

# print("median radial error dither", numpy.median(df.desi_roff))
print("rms radial error dither", numpy.sqrt(numpy.mean(df.roff**2)))
print("90 percentile", numpy.percentile(df.roff, 90))

# plt.show()


# try to fit the pattern? in focal coords now but needs to be in wok coords later?
x = df.xfocal.to_numpy()
y = df.yfocal.to_numpy()
xp = x + df[dxCol].to_numpy()
yp = y + df[dyCol].to_numpy()

numpy.random.seed(42)
rands = numpy.random.uniform(size=len(x))
train_inds = rands <= 0.8
test_inds = rands > 0.8


polids, coeffs = fitZhaoBurge(x[train_inds],y[train_inds],xp[train_inds],yp[train_inds], polids=numpy.arange(33))
dx_fit, dy_fit = getZhaoBurgeXY(polids, coeffs, x[test_inds], y[test_inds])
dx_test = df[dxCol].to_numpy()[test_inds]
dy_test = df[dyCol].to_numpy()[test_inds]
resid_xtest = dx_test - dx_fit
resid_ytest = dy_test - dy_fit
resid_rtest = numpy.sqrt(resid_xtest**2+resid_ytest**2)
print("rms test error", numpy.sqrt(numpy.mean(resid_rtest**2)))

# now fit everything
polids, coeffs = fitZhaoBurge(x,y,xp,yp, polids=numpy.arange(33))
dx_fit, dy_fit = getZhaoBurgeXY(polids, coeffs, x, y)


resid_x = df[dxCol].to_numpy() - dx_fit
resid_y = df[dyCol].to_numpy() - dy_fit
resid_r = numpy.sqrt(resid_x**2+resid_y**2)
df["dxZBFit"] = resid_x
df["dyZBFit"] = resid_y


rms = numpy.sqrt(numpy.mean(resid_r**2))
perc90 = numpy.percentile(resid_r, 90)

plt.figure(figsize=(10,10))
plt.quiver(df.xfocal, df.yfocal, resid_x, resid_y, angles="xy", units="xy", width=0.5, scale=.02)
plt.axis("equal")
plt.xlabel("x focal (mm)")
plt.ylabel("y focal (mm)")
plt.title("ZB fit corrected\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.savefig("fit_quiver.png", dpi=250)



plt.figure()
bins = numpy.linspace(0,3,100)
plt.hist(resid_r, bins=bins)
plt.xlabel("dr (arcsec)")
plt.title("ZB fit corrected\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
plt.savefig("fit_hist.png", dpi=250)



# print("median radial error ZB fit", numpy.median(resid_r))
# print("rms error ZB fit", numpy.sqrt(numpy.mean(resid_r**2)))

# plotBetaOffsets(df, "dxZBFit", "dyZBFit")
radialOff = plotBetaOffsets(df, "dxZBFit", "dyZBFit", pngDir="figsFit", title="post wok correct: ")
plt.figure()
bins = numpy.linspace(0,3,100)
plt.hist(radialOff, bins=bins)
plt.title("derived average fiber-fiber offset")
plt.xlabel("dr (arcsec)")
plt.savefig("hist_beta_off.png", dpi=250)


# plt.show()







