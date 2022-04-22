import pandas
import seaborn as sns
import time
import matplotlib.pyplot as plt
import numpy
from skimage.transform import SimilarityTransform

# positioners that had at least one wokErr > 0.5
bad_positioners = [256, 962, 130, 1300, 565, 985, 1114]
UM_PER_PIX = 120
UM_PER_MM = 1000

t1 = time.time()

dataDir = "/Volumes/futa/utah/rot3/"
centtype = "sep"


def rot2CCD(dx,dy,ipa,fvcRot=0):
    ipa = numpy.radians(ipa)
    fvcRot = numpy.radians(fvcRot)
    rotAngRef = numpy.radians(135.4)
    rot = ipa - rotAngRef

    cosRot = numpy.cos(-rot)
    sinRot = numpy.sin(-rot)

    rotMat = numpy.array([
                [cosRot, -sinRot],
                [sinRot, cosRot]
            ])
    dxy = numpy.array([dx,dy])
    dxyRot = rotMat @ dxy
    return dxy[0], dxy[1]


df = pandas.read_csv(dataDir+"raw_filtered.csv")
df = df[df.mjd==59661]
df = df[~df.positionerID.isin(bad_positioners)]
df = df[df.centtype==centtype].copy()

dfm = df.groupby(["centtype", "slewNum", "positionerID", "configid", "polidName"]).mean().reset_index()
dfmr = df.groupby(["centtype", "positionerID", "configid", "polidName"]).mean().reset_index()
df = df.merge(dfm, on=["positionerID", "configid", "slewNum", "polidName"], suffixes=(None, "_m"))
df = df.merge(dfmr, on=["positionerID", "configid", "polidName"], suffixes=(None, "_mr"))

if centtype == "simple":
    df["dxPix"] = df.xSimple - df.xSimple_m
    df["dyPix"] = df.ySimple - df.ySimple_m
elif centtype=="winpos":
    df["dxPix"] = df.xWinpos - df.xWinpos_m
    df["dyPix"] = df.yWinpos - df.yWinpos_m
else:
    df["dxPix"] = df.x - df.x_m
    df["dyPix"] = df.y - df.y_m

df["dxWok"] = df.xWokMeasMetrology - df.xWokMeasMetrology_m
df["dyWok"] = df.yWokMeasMetrology - df.yWokMeasMetrology_m

df["dxWok_rotMarg"] = df.xWokMeasMetrology - df.xWokMeasMetrology_mr
df["dyWok_rotMarg"] = df.yWokMeasMetrology - df.yWokMeasMetrology_mr

imgNums = sorted(list(set(df.imgNum)))

movieCounter = 0
for imgNum in imgNums:
    _df = df[df.imgNum==imgNum]

    movieNumStr = ("%i"%movieCounter).zfill(4)
    imgNumStr = ("%i"%imgNum).zfill(4)
    mjdStr = "%i"%(list(set(_df.mjd))[0])
    rotpos = float(list(set(_df.rotpos))[0])
    rotposStr = "%.2f"%(rotpos)
    configIDStr = "%i"%(list(set(_df.configid))[0])

    # just use nominal model for pixels, not applied but dont want double
    if centtype=="simple":
        x_pix = _df[_df.polidName=="nom"]["xSimple_m"]
        y_pix = _df[_df.polidName=="nom"]["ySimple_m"]
    elif centtype=="winpos":
        x_pix = _df[_df.polidName=="nom"]["xWinpos_m"]
        y_pix = _df[_df.polidName=="nom"]["yWinpos_m"]
    else:
        x_pix = _df[_df.polidName=="nom"]["x_m"]
        y_pix = _df[_df.polidName=="nom"]["y_m"]

    dx_pix = _df[_df.polidName=="nom"]["dxPix"].to_numpy()
    dy_pix = _df[_df.polidName=="nom"]["dyPix"].to_numpy()

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
    dx_nom_ccdRot, dy_nom_ccdRot = rot2CCD(dx_nom_rotMarg, dy_nom_rotMarg, rotPos)
    dx_all_ccdRot, dy_all_ccdRot = rot2CCD(dx_all_rotMarg, dy_all_rotMarg, rotPos)



    fig, axs = plt.subplots(4,2, figsize=(5*2,7*2))
    axs = axs.flatten()

    # pixels
    q = axs[0].quiver(x_pix, y_pix, dx_pix, dy_pix, angles="xy", units="xy", width=10, scale=0.0005)
    axs[0].quiverkey(q, 0.9, 0.9, 0.1, "0.1 pix (12 $\mu m$)")
    axs[0].set_xlabel("x (pix)")
    axs[0].set_ylabel("y (pix)")
    axs[0].set_xlim([1000,7000])
    axs[0].set_ylim([0,6000])
    axs[0].set_aspect("equal")
    axs[0].text(1500, 100, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_pix*UM_PER_PIX, r90_pix*UM_PER_PIX, rMax_pix*UM_PER_PIX), verticalalignment="bottom", horizontalalignment="left")

    q = axs[1].quiver(x_pix, y_pix, dx_pix_dshift, dy_pix_dshift, angles="xy", units="xy", width=10, scale=0.0005)
    axs[1].quiverkey(q, 0.9, 0.9, 0.1, "0.1 pix (12 $\mu m$)")
    axs[1].set_xlabel("x (pix)")
    axs[1].set_ylabel("y (pix)")
    axs[1].set_xlim([1000,7000])
    axs[1].set_ylim([0,6000])
    axs[1].set_aspect("equal")
    axs[1].text(1500, 100, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_pix_dshift*UM_PER_PIX, r90_pix_dshift*UM_PER_PIX, rMax_pix_dshift*UM_PER_PIX), verticalalignment="bottom", horizontalalignment="left")


    # within slew
    q = axs[2].quiver(x_nom, y_nom, dx_nom, dy_nom, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00002/(UM_PER_PIX/UM_PER_MM))
    axs[2].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    axs[2].set_xlabel("x (mm)")
    axs[2].set_ylabel("y (mm)")
    axs[2].set_xlim([-350, 350])
    axs[2].set_ylim([-350, 350])
    axs[2].set_aspect("equal")
    axs[2].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_nom*UM_PER_MM, r90_nom*UM_PER_MM, rMax_nom*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")

    q = axs[3].quiver(x_all, y_all, dx_all, dy_all, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00002/(UM_PER_PIX/UM_PER_MM))
    axs[3].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    axs[3].set_xlabel("x (mm)")
    axs[3].set_ylabel("y (mm)")
    axs[3].set_xlim([-350,350])
    axs[3].set_ylim([-350, 350])
    axs[3].set_aspect("equal")
    axs[3].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_all*UM_PER_MM, r90_all*UM_PER_MM, rMax_all*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")

    # between slews
    q = axs[4].quiver(x_nom_rotMarg, y_nom_rotMarg, dx_nom_rotMarg, dy_nom_rotMarg, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00008/(UM_PER_PIX/UM_PER_MM))
    axs[4].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    axs[4].set_xlabel("x (mm)")
    axs[4].set_ylabel("y (mm)")
    axs[4].set_xlim([-350, 350])
    axs[4].set_ylim([-350, 350])
    axs[4].set_aspect("equal")
    axs[4].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_nom_rotMarg*UM_PER_MM, r90_nom_rotMarg*UM_PER_MM, rMax_nom_rotMarg*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")

    q = axs[5].quiver(x_all_rotMarg, y_all_rotMarg, dx_all_rotMarg, dy_all_rotMarg, angles="xy", units="xy", width=10*(UM_PER_PIX/UM_PER_MM), scale=0.00008/(UM_PER_PIX/UM_PER_MM))
    axs[5].quiverkey(q, 0.9, 0.9, 0.012, "12 $\mu m$")
    axs[5].set_xlabel("x (mm)")
    axs[5].set_ylabel("y (mm)")
    axs[5].set_xlim([-350,350])
    axs[5].set_ylim([-350, 350])
    axs[5].set_aspect("equal")
    axs[5].text(-300, -300, "$p_{50}$=%.0f $p_{90}$=%.0f MAX=%.0f $\mu m$"%(r50_all_rotMarg*UM_PER_MM, r90_all_rotMarg*UM_PER_MM, rMax_all_rotMarg*UM_PER_MM), verticalalignment="bottom", horizontalalignment="left")

    ## rotated back to ccd frame
    q = axs[6].quiver(x_pix, y_pix, dx_nom_ccdRot*UM_PER_MM/UM_PER_PIX, dy_nom_ccdRot*UM_PER_MM/UM_PER_PIX, angles="xy", units="xy", width=10, scale=0.0005)
    axs[6].quiverkey(q, 0.9, 0.9, 0.1, "12 $\mu m$")
    axs[6].set_xlabel("x (pix)")
    axs[6].set_ylabel("y (pix)")
    axs[6].set_xlim([1000,7000])
    axs[6].set_ylim([0,6000])
    axs[6].set_aspect("equal")

    q = axs[7].quiver(x_pix, y_pix, dx_nom_ccdRot*UM_PER_MM/UM_PER_PIX, dy_nom_ccdRot*UM_PER_MM/UM_PER_PIX, angles="xy", units="xy", width=10, scale=0.0005)
    axs[7].quiverkey(q, 0.9, 0.9, 0.1, "12 $\mu m$")
    axs[7].set_xlabel("x (pix)")
    axs[7].set_ylabel("y (pix)")
    axs[7].set_xlim([1000,7000])
    axs[7].set_ylim([0,6000])
    axs[7].set_aspect("equal")

    imgName = "movie3/dxy-%s-%s.png"%(centtype, movieNumStr)
    fig.suptitle("centtype=%s config=%s  rot=%s imgNum=%s"%(centtype, configIDStr, rotposStr, imgNumStr))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(imgName, dpi=200)

    movieCounter += 1

    plt.close("all")

    # ffmpeg -r 10 -f image2 -i dxy-pix-%04d.png -pix_fmt yuv420p dxy-pix.mp4











