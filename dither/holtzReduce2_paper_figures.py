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
from coordio.defaults import PLATE_SCALE

# uses holtz' new dither.fits tables

# ffmpeg -r 10 -f image2 -i b1-%04d.png -pix_fmt yuv420p b1.mp4

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

cp = sns.color_palette("flare_r")
color1 = cp[0]
color2 = cp[-1]

def betaArmRot(df, dxWokCol, dyWokCol):
    df = df.copy()
    alpha = df.alpha.to_numpy()
    beta = df.beta.to_numpy()
    alphaOff = df.alphaOffset.to_numpy()
    betaOff = df.betaOffset.to_numpy()
    totalRot = numpy.radians(alpha+beta+alphaOff+betaOff-90)
    # totalRot = numpy.radians(alpha+beta-90)
    df["totalRot"] = totalRot
    df["rot"] = numpy.round(numpy.degrees(totalRot)%360, decimals=-1)
    dxWok = df[dxWokCol].to_numpy()
    dyWok = df[dyWokCol].to_numpy()

    # dra_m = dra - numpy.nanmean(dra)
    # ddec_m = ddec - numpy.nanmean(ddec)

    dxbeta = numpy.cos(totalRot)*dxWok + numpy.sin(totalRot)*dyWok
    dybeta = -numpy.sin(totalRot)*dxWok + numpy.cos(totalRot)*dyWok

    # dxbeta_m = numpy.cos(totalRot)*dra_m + numpy.sin(totalRot)*ddec_m
    # dybeta_m = -numpy.sin(totalRot)*dra_m + numpy.cos(totalRot)*ddec_m

    df["dxbeta"] = dxbeta
    df["dybeta"] = dybeta
    # df["drBeta"] = numpy.sqrt(dxbeta**2+dybeta**2)
    # df["dThetaBeta"] = numpy.degrees(numpy.arctan2(dybeta, dxbeta))

    # df["dxbeta_m"] = dxbeta_m
    # df["dybeta_m"] = dybeta_m
    # df["drBeta_m"] = numpy.sqrt(dxbeta_m**2+dybeta_m**2)
    # df["dThetaBeta_m"] = numpy.degrees(numpy.arctan2(dybeta_m, dxbeta_m))

    return df




df = pandas.read_csv("holtz2merged.csv")

# import pdb; pdb.set_trace()

# configIDs = [5259, 5260, 5144, 5146, 5148, 5286, 5287, 5288, 5165, 5167, 5169, 5300, 5301, 5302, 5315, 5316, 5189, 5319, 5192, 5194]
# print("len before", len(df))
# df = df[df.configuration_id.isin(configIDs)]
# print("len after", len(df))
# import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()

dxCol = "fit_dx_res"
dyCol = "fit_dy_res"

# convert arcseconds to microns
df["fit_dx_res"] = df["fit_dx_res"]/3600.0*PLATE_SCALE["APO"]*1000
df["fit_dy_res"] = df["fit_dy_res"]/3600.0*PLATE_SCALE["APO"]*1000
# dxCol = "desi_xfiboff"
# dyCol = "desi_yfiboff"

df["roff"] = numpy.sqrt(df[dxCol]**2 + df[dyCol]**2)
df = df[df.roff < 1.6*60]

print("n meas", len(df))
allFibers = list(set(df.fiberId))
print("n fibers", len(allFibers))
nMeas = []
for _fid in allFibers:
    _df = df[df.fiberId==_fid]
    nMeas.append(len(_df))
print("meas per fiber", numpy.min(nMeas), numpy.median(nMeas), numpy.mean(nMeas), numpy.max(nMeas))
configs = list(set(df.configuration_id))
designs = list(set(df.design_id))
mjds = list(set(df.mjd))

print("nDesigns", len(designs), "nConfigs", len(configs), "nMJD", len(mjds))

for design in designs:
    _ddf = df[df.design_id==design]
    nConfigs = len(set(_ddf.configuration_id))
    print("design, nconfig", design, nConfigs)

def plotErrs(df, dxCol, dyCol, filename=None):
    rErr = numpy.sqrt(df[dxCol]**2+df[dyCol]**2)
    rms = numpy.sqrt(numpy.mean(rErr**2))
    # perc90 = numpy.percentile(df.roff, 90)

    f, axs = plt.subplots(1,2,figsize=(11,5), gridspec_kw={'width_ratios': [3, 2]})
    # plt.figure(figsize=(8,8))
    q = axs[0].quiver(df.xwok, df.ywok, df[dxCol], df[dyCol], angles="xy", units="xy", width=0.5, scale=3)

    axs[0].quiverkey(q, X=0.94, Y=.98, U=60, label=r"60 $\mu$m", labelpos="S")
    axs[0].set_aspect("equal")
    # plt.title("Measured Offsets\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
    axs[0].text(220, -260, "RMS: %.0f"%rms + r" $\mu$m")
    axs[0].set_xlabel("x (mm)")
    axs[0].set_ylabel("y (mm)")
    # plt.savefig("raw_quiver_paper.png", dpi=250)

    # plt.figure()
    bins = numpy.linspace(0,120,40)
    axs[1].hist(rErr, bins=bins, alpha=0.5, color="gray")
    axs[1].hist(rErr, bins=bins, histtype="step", color="black")
    axs[1].set_xlabel("fiber position error ($\mu$m)")
    axs[1].set_ylabel("n measurements")
    # axs[1].set_savefig("raw_hist.png", dpi=250)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    # plt.show()


    # import pdb; pdb.set_trace()

plotErrs(df,"fit_dx_res", "fit_dy_res", "ditherRawQuiver.pdf")

# plotBetaOffsets(df, "desi_xfiboff", "desi_yfiboff")

# import pdb; pdb.set_trace()

# plotBetaOffsets(df, "desi_xfiboff", "desi_yfiboff", title="pre wok correct: ")

# betaResids, stdx1, stdy1, nConfig = plotBetaOffsets(df, dxCol, dyCol, title="pre wok correct: ")
# keep = nConfig >= 3
# stdx1 = stdx1[keep]
# stdy1 = stdy1[keep]

# rms = numpy.sqrt(numpy.mean(betaResids**2))
# perc90 = numpy.percentile(betaResids, 90)
# plt.figure()
# bins = numpy.linspace(0,3,100)
# plt.hist(betaResids, bins=bins)
# plt.title("Beta off removed\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
# plt.xlabel("dr (arcsec)")
# plt.savefig("beta_removed.png", dpi=250)

# try to fit the pattern? in focal coords now but needs to be in wok coords later?
x = df.xwok.to_numpy()
y = df.ywok.to_numpy()
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
# polids, coeffs = fitZhaoBurge(x,y,xp,yp, polids=numpy.arange(33))
dx_fit, dy_fit = getZhaoBurgeXY(polids, coeffs, x, y)

print("polids", polids)
print("coeffs", coeffs)

print("mean offsets", numpy.mean(dx_fit), numpy.mean(dy_fit))

plt.figure(figsize=(6,6))
q = plt.quiver(df.xwok, df.ywok, dx_fit, dy_fit, angles="xy", units="xy", width=0.5, scale=3)
ax = plt.gca()
ax.quiverkey(q, X=0.94, Y=.98, U=60, label=r"60 $\mu$m", labelpos="S")
plt.axis("equal")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.savefig("ditherZBModel.pdf", bbox_inches="tight")
# plt.title("model")

# plt.figure(figsize=(10,10))
# plt.quiver(df.xwok, df.ywok, df[dxCol], df[dyCol], angles="xy", units="xy", width=0.5, scale=.05)
# plt.axis("equal")
# plt.xlabel("x wok (mm)")
# plt.ylabel("y wok (mm)")
# plt.title("data")

df["dxZB"] = df[dxCol]-dx_fit
df["dyZB"] = df[dyCol]-dy_fit
df["drZB"] = numpy.sqrt(df.dxZB**2+df.dyZB**2)

plotErrs(df,"dxZB", "dyZB", "ditherZBQuiver.pdf")

df2 = betaArmRot(df,"dxZB","dyZB")

df_mean = df2.groupby("fiberId").mean().reset_index()
df_mean = df_mean.sort_values("drZB")

testIDs = set(df_mean[(df_mean.dxbeta < -15) & (df_mean.dybeta > 5)]["fiberId"])
print("test ids", testIDs)
dfSortedFiberIds = df_mean.fiberId.to_numpy()
plt.figure()
plt.plot(df_mean.fiberId, df_mean.drZB, 'o-')

# testFibers = [226, 332, 73, 270, 424] # 113
testFibers = [270, 226, 73, 424]
# testFibers = [356]

# {5, 6, 9, 22, 25, 28, 32, 35, 36, 45, 53, 54, 55, 58, 69, 71, 72, 78, 79, 80, 82, 84, 86, 87, 90, 92, 95, 98, 107, 109, 114, 116, 129, 141, 146, 148, 150, 155, 157, 163, 172, 175, 177, 179, 182, 185, 189, 190, 192, 198, 199, 200, 201, 205, 212, 214, 216, 220, 222, 228, 232, 234, 244, 252, 276, 281, 282, 286, 290, 298, 302, 318, 319, 322, 344, 347, 348, 353, 356, 361, 362, 363, 364, 365, 371, 375, 377, 379, 380, 391, 393, 394, 395, 398, 404, 407, 416, 418, 419, 421, 422, 425, 430, 433, 437, 448, 455, 461, 466, 468, 469, 481, 497, 500}


# dfTest = df2[df2.fiberId.isin(testFibers)]

# dfText["xOrigin"] = 0
# dfText["yOrigin"] = 0

# def _rotPlots(_df,dxCol,dyCol):
#     plt.figure(figsize=(8,8))
#     # dx = _df[dxCol].to_numpy()
#     # dy = _df[dyCol].to_numpy()
#     # x = numpy.zeros(len(dx))
#     # y = numpy.zeros(len(dx))

#     sns.scatterplot(x=dxCol, y=dyCol, style="fiberId", hue="fiberId", data=_df)
#     plt.axis("equal")
#     plt.grid("on")
#     plt.xlim([-100,100])
#     plt.ylim([-100,100])

#     # plt.plot(dx,dy,'ok')
#     # plt.quiver(x,y,dx,dy,units="xy",angles="xy")

# _rotPlots(dfTest, "dxZB", "dyZB")
# _rotPlots(dfTest, "dxbeta", "dybeta")
fig, axs = plt.subplots(2,2, figsize=(10,10))
axs = axs.flatten()
ii = 0
for fiber, ax in zip(testFibers,axs):

    _df = df2[df2.fiberId==fiber]


    x = _df.dxZB.to_numpy()
    y = _df.dyZB.to_numpy()
    mfc="none"
    mec=color2
    ms=13
    mew=2


    ax.plot(x,y,'o',ms=10, mfc=mfc,mec=mec, mew=mew, label="wok")

    x = _df.dxbeta.to_numpy()
    y = _df.dybeta.to_numpy()
    mfc=color1
    mec=color1

    ax.plot(x,y,'o', ms=10, mfc=mfc,mec=mec, mew=mew, alpha=0.5, label="beta arm")

    ax.set_xlim([-90,90])
    ax.set_ylim([-90,90])
    ax.set_aspect('equal')
    # ax.grid("on")
    # ax.set_xticks([-100,-75,-50,-25,0,25,50,75,100])
    # ax.set_yticks([-100,-75,-50,-25,0,25,50,75,100])

    ax.set_xticks([-90,-60,-30,0,30,60,90])
    ax.set_yticks([-90,-60,-30,0,30,60,90])

    if ii==0 or ii==1:
        ax.xaxis.set_ticklabels([])
    else:
        ax.set_xlabel("dx ($\mu$m)")
    if ii==1 or ii==3:
        ax.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel("dy ($\mu$m)")
    if ii==0:
        ax.legend(loc="upper left", title="coordinate system")



    medx = numpy.median(x)
    medy = numpy.median(y)
    ax.axvline(medx,ls="-",lw=2,color=color1, alpha=0.5, zorder=1)
    ax.axhline(medy,ls="-",lw=2,color=color1, alpha=0.5, zorder=1)

    ax.axhline(0, ls="--", lw=0.5, color="black",zorder=0)
    ax.axvline(0, ls="--", lw=0.5, color="black",zorder=0)

    fiberIDstr = "     fiber id = %i\n     n meas = %i\n"%(fiber, len(x))
    fiberIDstr += "beta offset = [%.0f, %.0f] "%(medx,medy) + r"$\mu$m"
    ax.text(-85,-85,fiberIDstr, ha="left",va="bottom")

    ii+=1

plt.tight_layout()
plt.savefig("wokVsbeta.pdf", bbox_inches="tight")
    # fig.suptitle(str(fiber))

# _rotPlots("dxbeta", "dybeta")


# plt.show()


# finally subtract median xy beta offsets from each measurment to find
# final distribution of fiber errors
df2_mean = df2.groupby("fiberId").mean().reset_index()
df_join = df2.merge(df2_mean, on="fiberId", suffixes=(None,"_med"))
dxFinal = df_join.dxbeta.to_numpy()-df_join.dxbeta_med.to_numpy()
dyFinal = df_join.dybeta.to_numpy()-df_join.dybeta_med.to_numpy()
drFinal = numpy.sqrt(dxFinal**2+dyFinal**2)

fig,ax = plt.subplots(1,1,figsize=(5,5.5))

bins = numpy.linspace(0,120,40)
ax.hist(drFinal, bins=bins, alpha=0.5, color="gray")
ax.hist(drFinal, bins=bins, histtype="step", color="black")
ax.set_xlabel("fiber position error ($\mu$m)")
ax.set_ylabel("n measurements")
plt.savefig("betaCorr.pdf", bbox_inches="tight")

rms = numpy.sqrt(numpy.mean(drFinal**2))
print("final rms:", rms)

percs = [50,60,75,90,95,99]
for perc in percs:
    p = numpy.percentile(drFinal, perc)
    print("perc %.0f %.0f %.1f"%(perc, p, p*3600.0/PLATE_SCALE["APO"]/1000))

plt.show()
import pdb; pdb.set_trace()

# plotErrs(df,"dxbeta","dybeta")

# plt.figure(figsize=(10,10))
# plt.quiver(df.xwok, df.ywok, df[dxCol]-dx_fit, df[dyCol]-dy_fit, angles="xy", units="xy", width=0.5, scale=.05)
# plt.axis("equal")
# plt.xlabel("x wok (mm)")
# plt.ylabel("y wok (mm)")
# plt.title("residual")

# plt.title("ZB fit corrected\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
# plt.savefig("fit_quiver.png", dpi=250)

plt.show()
import pdb; pdb.set_trace()

# plt.figure()
# plt.hist(numpy.sqrt(dx_fit**2+dy_fit**2), bins=numpy.linspace(0,2,40))

# plt.show()
# import pdb; pdb.set_trace()

# resid_x = df[dxCol].to_numpy() - dx_fit
# resid_y = df[dyCol].to_numpy() - dy_fit
# resid_r = numpy.sqrt(resid_x**2+resid_y**2)
# df["dxZBFit"] = resid_x
# df["dyZBFit"] = resid_y
# df["drZBFit"] = numpy.sqrt(df.dxZBFit**2+df.dyZBFit**2)


# rms = numpy.sqrt(numpy.mean(resid_r**2))
# perc90 = numpy.percentile(resid_r, 90)

# plt.figure(figsize=(10,10))
# plt.quiver(df.xwok, df.ywok, resid_x, resid_y, angles="xy", units="xy", width=0.5, scale=.05)
# plt.axis("equal")
# plt.xlabel("x wok (mm)")
# plt.ylabel("y wok (mm)")
# plt.title("ZB fit corrected\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
# plt.savefig("fit_quiver.png", dpi=250)



# plt.figure()
# bins = numpy.linspace(0,3,100)
# plt.hist(resid_r, bins=bins)
# plt.xlabel("dr (arcsec)")
# plt.title("ZB fit corrected\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
# plt.savefig("fit_hist.png", dpi=250)



# # print("median radial error ZB fit", numpy.median(resid_r))
# # print("rms error ZB fit", numpy.sqrt(numpy.mean(resid_r**2)))

# # plotBetaOffsets(df, "dxZBFit", "dyZBFit")
# betaResids, stdx2, stdy2, nConfig = plotBetaOffsets(df, "dxZBFit", "dyZBFit", pngDir="figsFit", title="post wok correct: ")
# keep = nConfig >= 3
# stdx2 = stdx2[keep]
# stdy2 = stdy2[keep]

# rms = numpy.sqrt(numpy.mean(betaResids**2))
# perc90 = numpy.percentile(betaResids, 90)


# plt.figure()
# bins = numpy.linspace(0,3,100)
# plt.hist(betaResids, bins=bins)
# plt.title("Fit + Beta off removed\nrms=%.3f  perc90=%.3f (arcsec)"%(rms, perc90))
# plt.xlabel("dr (arcsec)")
# plt.savefig("fit_beta_removed.png", dpi=250)

# plt.figure(figsize=(10,10))
# plt.plot(stdx1, stdy1, '.', color="tab:orange", label="orig")
# plt.plot(stdx2, stdy2, '.', color=color1, label="ZB fit corrected")
# sns.kdeplot(x=stdx1, y=stdy1, color="tab:orange", levels=3, ax=plt.gca())
# sns.kdeplot(x=stdx2, y=stdy2, color=color1, levels=3, ax=plt.gca())
# plt.xlabel("std x beta (arcsec)")
# plt.ylabel("std y beta (arcsec)")
# plt.legend()
# plt.axis("equal")
# plt.title("beta frame scatter\n(fibers w/ >= 3 designs)")
# plt.savefig("beta_scatter.png", dpi=250)


# plt.show()







