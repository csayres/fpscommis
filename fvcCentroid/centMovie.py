import pandas
import seaborn as sns
import time
import matplotlib.pyplot as plt
import numpy
from skimage.transform import SimilarityTransform

# positioners that had at least one wokErr > 0.5
bad_positioners = [256, 962, 130, 1300, 565, 985, 1114]
UM_PER_PIX = 121.5
UM_PER_MM = 1000

t1 = time.time()


centtype = "nudge"
sigma = 2
savedxys = False

if centtype=="simple" and sigma!=2:
    dataDir = "/Volumes/futa/utah/rot5/"
    df = pandas.read_csv(dataDir+"raw_filtered_59661.csv")
    df = df[df.mjd==59661]
    df = df[df.simpleSigma==sigma]
    df = df[~df.positionerID.isin(bad_positioners)]
    df = df[df.centtype==centtype].copy()
elif centtype=="nudge":
    dataDir = "/Volumes/futa/utah/rot6/"
    df = pandas.read_csv(dataDir+"raw_filtered.csv")
    df = df[df.mjd==59661]
    df = df[~df.positionerID.isin(bad_positioners)]
    df = df[df.centtype==centtype].copy()
else:
    dataDir = "/Volumes/futa/utah/rot3/"
    df = pandas.read_csv(dataDir+"raw_filtered.csv")
    df = df[df.mjd==59661]
    df = df[~df.positionerID.isin(bad_positioners)]
    df = df[df.centtype==centtype].copy()

dfm = df.groupby(["centtype", "slewNum", "positionerID", "configid", "polidName"]).mean().reset_index()
dfmr = df.groupby(["centtype", "positionerID", "configid", "polidName"]).mean().reset_index()
df = df.merge(dfm, on=["positionerID", "configid", "slewNum", "polidName"], suffixes=(None, "_m"))
df = df.merge(dfmr, on=["positionerID", "configid", "polidName"], suffixes=(None, "_mr"))



def rot2CCD(dx, dy, ipa, fvcRot=0, aboutCenter=False):

    if aboutCenter:
        mx = numpy.mean(dx)
        my = numpy.mean(dy)
        dx = dx - numpy.mean(dx)

    ipa = numpy.radians(ipa)
    fvcRot = numpy.radians(fvcRot)
    rotAngRef = numpy.radians(135.4)
    rot = ipa - rotAngRef

    cosRot = numpy.cos(rot)
    sinRot = numpy.sin(rot)

    rotMat = numpy.array([
                [cosRot, sinRot],
                [-sinRot, cosRot]
            ])
    dxy = numpy.array([dx,dy])
    dxyRot = rotMat @ dxy
    dxOut = dxyRot[0]
    dyOut = dxyRot[1]

    if aboutCenter:
        dxOut += mx
        dyOut += my
    return dxOut, dyOut


if centtype == "simple":
    df["dxPix"] = df.xSimple - df.xSimple_m
    df["dyPix"] = df.ySimple - df.ySimple_m
    df["xPix"] = df.xSimple_m
    df["yPix"] = df.ySimple_m
elif centtype == "nudge":
    df["dxPix"] = df.xNudge - df.xNudge_m
    df["dyPix"] = df.yNudge - df.yNudge_m
    df["xPix"] = df.xNudge_m
    df["yPix"] = df.yNudge_m
elif centtype == "winpos":
    df["dxPix"] = df.xWinpos - df.xWinpos_m
    df["dyPix"] = df.yWinpos - df.yWinpos_m
    df["xPix"] = df.xWinpos_m
    df["yPix"] = df.yWinpos_m
else:
    df["dxPix"] = df.x - df.x_m
    df["dyPix"] = df.y - df.y_m
    df["xPix"] = df.x_m
    df["yPix"] = df.y_m

df["dxWok"] = df.xWokMeasMetrology - df.xWokMeasMetrology_m
df["dyWok"] = df.yWokMeasMetrology - df.yWokMeasMetrology_m

df["dxWok_rotMarg"] = df.xWokMeasMetrology - df.xWokMeasMetrology_mr
df["dyWok_rotMarg"] = df.yWokMeasMetrology - df.yWokMeasMetrology_mr

imgNums = sorted(list(set(df.imgNum)))

dxRotSave = numpy.array([])
dyRotSave = numpy.array([])
xRotSave = numpy.array([])
yRotSave = numpy.array([])

movieCounter = 0
for imgNum in imgNums:
    _df = df[df.imgNum==imgNum]

    movieNumStr = ("%i"%movieCounter).zfill(4)
    imgNumStr = ("%i"%imgNum).zfill(4)
    mjdStr = "%i"%(list(set(_df.mjd))[0])
    rotpos = float(list(set(_df.rotpos))[0])
    rotposStr = "%.2f"%(rotpos)
    configIDStr = "%i"%(list(set(_df.configid))[0])

    # just use all model for pixels, not applied but dont want double
    x_pix = _df[_df.polidName=="all"]["xPix"].to_numpy()
    y_pix = _df[_df.polidName=="all"]["yPix"].to_numpy()

    dx_pix = _df[_df.polidName=="all"]["dxPix"].to_numpy()
    dy_pix = _df[_df.polidName=="all"]["dyPix"].to_numpy()

    rerr = numpy.sqrt(dx_pix**2+dy_pix**2)
    r50_pix = numpy.percentile(rerr, 50)
    r90_pix = numpy.percentile(rerr, 90)
    rMax_pix = numpy.max(rerr)


    dx_pix_dshift = dx_pix - numpy.mean(dx_pix)
    dy_pix_dshift = dy_pix - numpy.mean(dy_pix)

    rerr = numpy.sqrt(dx_pix_dshift**2+dy_pix_dshift**2)
    r50_pix_dshift = numpy.percentile(rerr, 50)
    r90_pix_dshift = numpy.percentile(rerr, 90)
    rMax_pix_dshift = numpy.max(rerr)

    # within rotator angle/slewNum
    x_nom = _df[_df.polidName=="nom"]["xWokMeasMetrology_m"].to_numpy()
    y_nom = _df[_df.polidName=="nom"]["yWokMeasMetrology_m"].to_numpy()
    dx_nom = _df[_df.polidName=="nom"]["dxWok"].to_numpy()
    dy_nom = _df[_df.polidName=="nom"]["dyWok"].to_numpy()


    rerr = numpy.sqrt(dx_nom**2+dy_nom**2)
    r50_nom = numpy.percentile(rerr, 50)
    r90_nom = numpy.percentile(rerr, 90)
    rMax_nom = numpy.max(rerr)

    # rms_nom = numpy.sqrt(numpy.mean([dx_nom**2, dy_nom**2]))
    # max_nom = numpy.max(numpy.sqrt(dx_nom**2+dy_nom**2))

    x_all = _df[_df.polidName=="all"]["xWokMeasMetrology_m"].to_numpy()
    y_all = _df[_df.polidName=="all"]["yWokMeasMetrology_m"].to_numpy()
    dx_all = _df[_df.polidName=="all"]["dxWok"].to_numpy()
    dy_all = _df[_df.polidName=="all"]["dyWok"].to_numpy()

    rerr = numpy.sqrt(dx_all**2+dy_all**2)
    r50_all = numpy.percentile(rerr, 50)
    r90_all = numpy.percentile(rerr, 90)
    rMax_all = numpy.max(rerr)

    # between rotator angle/slewNum
    x_nom_rotMarg = _df[_df.polidName=="nom"]["xWokMeasMetrology_mr"].to_numpy()
    y_nom_rotMarg = _df[_df.polidName=="nom"]["yWokMeasMetrology_mr"].to_numpy()
    dx_nom_rotMarg = _df[_df.polidName=="nom"]["dxWok_rotMarg"].to_numpy()
    dy_nom_rotMarg = _df[_df.polidName=="nom"]["dyWok_rotMarg"].to_numpy()

    rerr = numpy.sqrt(dx_nom_rotMarg**2+dy_nom_rotMarg**2)
    r50_nom_rotMarg = numpy.percentile(rerr, 50)
    r90_nom_rotMarg = numpy.percentile(rerr, 90)
    rMax_nom_rotMarg = numpy.max(rerr)

    x_all_rotMarg = _df[_df.polidName=="all"]["xWokMeasMetrology_mr"].to_numpy()
    y_all_rotMarg = _df[_df.polidName=="all"]["yWokMeasMetrology_mr"].to_numpy()
    dx_all_rotMarg = _df[_df.polidName=="all"]["dxWok_rotMarg"].to_numpy()
    dy_all_rotMarg = _df[_df.polidName=="all"]["dyWok_rotMarg"].to_numpy()

    rerr = numpy.sqrt(dx_all_rotMarg**2+dy_all_rotMarg**2)
    r50_all_rotMarg = numpy.percentile(rerr, 50)
    r90_all_rotMarg = numpy.percentile(rerr, 90)
    rMax_all_rotMarg = numpy.max(rerr)

    # rotate errors into ccd frame
    dx_nom_ccdRot, dy_nom_ccdRot = rot2CCD(dx_nom_rotMarg, dy_nom_rotMarg, rotpos)
    dx_all_ccdRot, dy_all_ccdRot = rot2CCD(dx_all_rotMarg, dy_all_rotMarg, rotpos)
    x_nom_ccdRot, y_nom_ccdRot = rot2CCD(x_nom_rotMarg, y_nom_rotMarg, rotpos, aboutCenter=True)
    x_all_ccdRot, y_all_ccdRot = rot2CCD(x_all_rotMarg, y_all_rotMarg, rotpos, aboutCenter=True)

    dxRotSave = numpy.concatenate((dxRotSave, dx_all_ccdRot))
    dyRotSave = numpy.concatenate((dyRotSave, dy_all_ccdRot))
    xRotSave = numpy.concatenate((xRotSave, x_pix))
    yRotSave = numpy.concatenate((yRotSave, y_pix))
    # print("worked")



    fig, axs = plt.subplots(4,2, figsize=(5*2,7*2))
    axs = axs.flatten()

    # pixels
    q = axs[0].quiver(x_pix, y_pix, dx_pix, dy_pix, angles="xy", units="xy", width=10, scale=0.0005)
    axs[0].quiverkey(q, 0.9, 0.9, 0.1, "0.1 pix (12 $\mu m$)")
    axs[0].set_xlabel("xCCD (pix)")
    axs[0].set_ylabel("yCCD (pix)")
    axs[0].set_xlim([1000,7000])
    axs[0].set_ylim([0,6000])
    axs[0].set_aspect("equal")
    axs[0].text(1500, 100, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_pix*UM_PER_PIX, r90_pix*UM_PER_PIX, rMax_pix*UM_PER_PIX), verticalalignment="bottom", horizontalalignment="left")

    q = axs[1].quiver(x_pix, y_pix, dx_pix_dshift, dy_pix_dshift, angles="xy", units="xy", width=10, scale=0.0005)
    axs[1].quiverkey(q, 0.9, 0.9, 0.1, "0.1 pix (12 $\mu m$)")
    axs[1].set_xlabel("xCCD - mean_shift_x (pix)")
    axs[1].set_ylabel("yCCD - mean_shift_y (pix)")
    axs[1].set_xlim([1000,7000])
    axs[1].set_ylim([0,6000])
    axs[1].set_aspect("equal")
    axs[1].text(1500, 100, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_pix_dshift*UM_PER_PIX, r90_pix_dshift*UM_PER_PIX, rMax_pix_dshift*UM_PER_PIX), verticalalignment="bottom", horizontalalignment="left")


    # within slew
    q = axs[2].quiver(x_nom, y_nom, dx_nom, dy_nom, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00002/(UM_PER_PIX/UM_PER_MM))
    axs[2].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    axs[2].set_xlabel("xWok (mm)")
    axs[2].set_ylabel("yWok (mm)")
    axs[2].set_xlim([-350, 350])
    axs[2].set_ylim([-350, 350])
    axs[2].set_aspect("equal")
    axs[2].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_nom*UM_PER_MM, r90_nom*UM_PER_MM, rMax_nom*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")
    axs[2].text(-300, 300, "ZB Terms=11", verticalalignment="bottom", horizontalalignment="left")

    q = axs[3].quiver(x_all, y_all, dx_all, dy_all, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00002/(UM_PER_PIX/UM_PER_MM))
    axs[3].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    axs[3].set_xlabel("xWok (mm)")
    axs[3].set_ylabel("yWok (mm)")
    axs[3].set_xlim([-350,350])
    axs[3].set_ylim([-350, 350])
    axs[3].set_aspect("equal")
    axs[3].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_all*UM_PER_MM, r90_all*UM_PER_MM, rMax_all*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")
    axs[3].text(-300, 300, "ZB Terms=33", verticalalignment="bottom", horizontalalignment="left")

    # between slews
    q = axs[4].quiver(x_nom_rotMarg, y_nom_rotMarg, dx_nom_rotMarg, dy_nom_rotMarg, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00008/(UM_PER_PIX/UM_PER_MM))
    axs[4].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    axs[4].set_xlabel("xWok (mm)")
    axs[4].set_ylabel("yWok (mm)")
    axs[4].set_xlim([-350, 350])
    axs[4].set_ylim([-350, 350])
    axs[4].set_aspect("equal")
    axs[4].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_nom_rotMarg*UM_PER_MM, r90_nom_rotMarg*UM_PER_MM, rMax_nom_rotMarg*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")
    axs[4].text(-300, 300, "ZB Terms=11", verticalalignment="bottom", horizontalalignment="left")

    q = axs[5].quiver(x_all_rotMarg, y_all_rotMarg, dx_all_rotMarg, dy_all_rotMarg, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00008/(UM_PER_PIX/UM_PER_MM))
    axs[5].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    axs[5].set_xlabel("xWok (mm)")
    axs[5].set_ylabel("yWok (mm)")
    axs[5].set_xlim([-350,350])
    axs[5].set_ylim([-350, 350])
    axs[5].set_aspect("equal")
    axs[5].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_all_rotMarg*UM_PER_MM, r90_all_rotMarg*UM_PER_MM, rMax_all_rotMarg*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")
    axs[5].text(-300, 300, "ZB Terms=33", verticalalignment="bottom", horizontalalignment="left")

    ## rotate to align with ccd frame
    # q = axs[6].quiver(x_nom_ccdRot, y_nom_ccdRot, dx_nom_ccdRot, dy_nom_ccdRot, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00008/(UM_PER_PIX/UM_PER_MM))
    # axs[6].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    # axs[6].set_xlabel(r"rot($\theta_{ccd}$) xWok (mm)")
    # axs[6].set_ylabel(r"rot($\theta_{ccd}$) yWok (mm)")
    # axs[6].set_xlim([-350, 350])
    # axs[6].set_ylim([-350, 350])
    # axs[6].set_aspect("equal")
    # axs[6].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_nom_rotMarg*UM_PER_MM, r90_nom_rotMarg*UM_PER_MM, rMax_nom_rotMarg*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")

    # q = axs[7].quiver(x_all_ccdRot, y_all_ccdRot, dx_all_ccdRot, dy_all_ccdRot, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00008/(UM_PER_PIX/UM_PER_MM))
    # axs[7].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    # axs[7].set_xlabel(r"rot($\theta_{ccd}$) xWok (mm)")
    # axs[7].set_ylabel(r"rot($\theta_{ccd}$) yWok (mm)")
    # axs[7].set_xlim([-350,350])
    # axs[7].set_ylim([-350, 350])
    # axs[7].set_aspect("equal")
    # axs[7].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_all_rotMarg*UM_PER_MM, r90_all_rotMarg*UM_PER_MM, rMax_all_rotMarg*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")


    ## rotated back to ccd frame
    q = axs[6].quiver(x_pix, y_pix, dx_nom_ccdRot*UM_PER_MM/UM_PER_PIX, dy_nom_ccdRot*UM_PER_MM/UM_PER_PIX, angles="xy", units="xy", width=10, scale=0.0005)
    axs[6].quiverkey(q, 0.9, 0.9, 0.1, "12 $\mu m$")
    axs[6].set_xlabel("xCCD (pix)")
    axs[6].set_ylabel("yCCD (pix)")
    axs[6].set_xlim([1000,7000])
    axs[6].set_ylim([0,6000])
    axs[6].set_aspect("equal")
    axs[6].text(1500, 100, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_nom_rotMarg*UM_PER_MM, r90_nom_rotMarg*UM_PER_MM, rMax_nom_rotMarg*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")
    axs[6].text(1500, 5500, "ZB Terms=11", verticalalignment="bottom", horizontalalignment="left")

    q = axs[7].quiver(x_pix, y_pix, dx_all_ccdRot*UM_PER_MM/UM_PER_PIX, dy_all_ccdRot*UM_PER_MM/UM_PER_PIX, angles="xy", units="xy", width=10, scale=0.0005)
    axs[7].quiverkey(q, 0.9, 0.9, 0.1, "12 $\mu m$")
    axs[7].set_xlabel("xCCD (pix)")
    axs[7].set_ylabel("yCCD (pix)")
    axs[7].set_xlim([1000,7000])
    axs[7].set_ylim([0,6000])
    axs[7].set_aspect("equal")
    axs[7].text(1500, 100, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_all_rotMarg*UM_PER_MM, r90_all_rotMarg*UM_PER_MM, rMax_all_rotMarg*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")
    axs[7].text(1500, 5500, "ZB Terms=33", verticalalignment="bottom", horizontalalignment="left")


    if centtype=="simple":
        imgName = "movie3/dxy-%s-%.1f-%s.png"%(centtype, sigma, movieNumStr)
        fig.suptitle("centtype=%s ($\sigma=%.1f px$) config=%s rot=%s imgNum=%s"%(centtype, sigma, configIDStr, rotposStr, imgNumStr))
    else:
        imgName = "movie3/dxy-%s-%s.png"%(centtype, movieNumStr)
        fig.suptitle("centtype=%s config=%s rot=%s imgNum=%s"%(centtype, configIDStr, rotposStr, imgNumStr))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(imgName, dpi=200)

    movieCounter += 1

    plt.close("all")

if savedxys:
    dxyDF = pandas.DataFrame({
        "dx": dxRotSave.flatten()*UM_PER_MM/UM_PER_PIX,  # pixels
        "dy": dyRotSave.flatten()*UM_PER_MM/UM_PER_PIX, # pixels
        "x": xRotSave.flatten(),
        "y": yRotSave.flatten()
        })

    dxyDF.to_csv("dxyPixels.csv")

    # ffmpeg -r 10 -f image2 -i dxy-pix-%04d.png -pix_fmt yuv420p dxy-pix.mp4











