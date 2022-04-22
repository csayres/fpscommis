import pandas
import seaborn as sns
import time
import matplotlib.pyplot as plt
import numpy
from skimage.transform import SimilarityTransform

# positioners that had at least one wokErr > 0.5
bad_positioners = [256, 962, 130, 1300, 565, 985, 1114]
UM_PER_PIX = 120
MM_PER_UM = 0.001

t1 = time.time()

dataDir = "/Volumes/futa/fvc/"

if False:
    af = pandas.read_csv(dataDir+"all_filtered.csv")
    af = af[~af.positionerID.isin(bad_positioners)].copy()
    af = af[af.navg==1].copy()
    afd = pandas.read_csv(dataDir+"all_filtered_demean.csv")
    afd = afd[~afd.positionerID.isin(bad_positioners)].copy()
    afd = afd[afd.navg==1].copy()

    # import pdb; pdb.set_trace()

    df = af.merge(afd, on=["positionerID", "polidName", "rotpos", "configid", "centtype", "imgNum"], suffixes=(None, "_dm")).reset_index()
    af = None
    afd = None

    df.to_csv("/Volumes/futa/fvc/all_filtered_merged.csv")

df = pandas.read_csv("/Volumes/futa/fvc/all_filtered_merged.csv")
mean = pandas.read_csv("/Volumes/futa/fvc/mean.csv")
mean = mean[~mean.positionerID.isin(bad_positioners)].copy()
mean = mean[mean.navg==1].copy()

df = df[df.polidName=="nom"].reset_index()
mean = mean[mean.polidName=="nom"].reset_index()


def gauss(std, xs):
    ys = (1/(std*numpy.sqrt(2*numpy.pi)))*numpy.exp(-0.5*(xs/std)**2)
    return ys


def histPlot(data, ax, lims, color, label):
    bins = numpy.linspace(lims[0], lims[1], 500)
    smooth = numpy.linspace(lims[0], lims[1], 2000)
    std = numpy.std(data)
    label = label + " ($\sigma$=%.3f mm)"%std
    ax.hist(data, bins=bins, color=color, alpha=1, density=True, histtype="step", label=label)
    # std = numpy.std(data)
    ys = gauss(std, smooth)
    ax.plot(smooth, ys, "-", color='black', lw=0.5, alpha=0.6)


def cdf(data, ax, color, label):
    rms = numpy.sqrt(numpy.mean(data**2))
    label = label + " (RMS=%.3f mm)"%rms
    bins = numpy.linspace(-0.003, 0.060, 500)
    ax.hist(data, bins=bins, density=True, histtype="step", cumulative=True, label=label)


figCounter = 0
def nf():
    global figCounter
    figCounter += 1
    return "fig-%i.png"%figCounter

lims = [-.03, .03]
bins = numpy.linspace(lims[0], lims[1], 500)
colorset = ["tab:blue", "tab:orange", "tab:red"]
centtypeList = ["sep", "winpos", "blanton"]
fig1, ax1 = plt.subplots(figsize=(10, 10))
fig2, ax2 = plt.subplots(figsize=(10, 5))
fig3, ax3 = plt.subplots(figsize=(10, 5))
fig4, ax4 = plt.subplots(figsize=(10, 10))
for color, centtype in zip(colorset, centtypeList):
    dxy = df[df.centtype==centtype][["xWokMeasMetrology_dm", "yWokMeasMetrology_dm"]].to_numpy()
    dr = numpy.linalg.norm(dxy,axis=1)

    ax1.plot(dxy[:,0], dxy[:,1], '.', ms=4, color=color, label=centtype, alpha=0.1)
    histPlot(dxy[:,0], ax2, lims, color, centtype)
    histPlot(dxy[:,1], ax3, lims, color, centtype)
    cdf(dr, ax4, color, centtype)


ax4.grid(True)
# ax4.axhline(0.998, label="499th robot")


ax1.set_xlabel("dx (mm)")
ax1.set_ylabel("dy (mm)")

ax2.set_xlabel("dx (mm)")
ax2.set_ylabel("density")

ax3.set_xlabel("dy (mm)")
ax3.set_ylabel("density")

ax4.set_ylabel("CDF")
ax4.set_xlabel("dr (mm)")

ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")
ax4.legend(loc="lower right")

ax1.set_xlim(lims)
ax2.set_xlim(lims)
ax3.set_xlim(lims)

ax1.set_ylim(lims)
ax2.set_yticks([])
ax3.set_yticks([])


fig1.savefig(nf(), dpi=350)
fig2.savefig(nf(), dpi=350)
fig3.savefig(nf(), dpi=350)
fig4.savefig(nf(), dpi=350)

ax4.set_ylim([0.9, 1.01])
ax4.axhline(499/500., linestyle="--", color="tab:red", label="499th robot")
ax4.legend(loc="lower right")
fig4.savefig(nf(), dpi=350)

plt.close("all")

xySep = mean[mean.centtype=="sep"][["x", "y"]].to_numpy()
xySimple = mean[mean.centtype=="blanton"][["xBlanton", "yBlanton"]].to_numpy()

dxy = xySep - xySimple

fig, ax1 = plt.subplots(figsize=(10,10))
q = ax1.quiver(xySep[:,0], xySep[:,1], dxy[:,0], dxy[:,1], angles="xy",units="xy", width=1, scale=0.01)
ax1.set_title("xySep - xySimple")
ax1.quiverkey(q, 0.9, 0.9, 1, "1 pixel")
ax1.set_xlabel("x (pix)")
ax1.set_ylabel("y (pix)")
fig.savefig(nf(), dpi=350)

tf = SimilarityTransform()
tf.estimate(xySep, xySimple)
resid = tf(xySep)

fig, ax2 = plt.subplots(figsize=(10,10))
dxy = resid - xySimple
ax2.set_title("xySep - xySimple \n(trans/rot/scale removed)")
q = ax2.quiver(resid[:,0], resid[:,1], dxy[:,0], dxy[:,1], angles="xy",units="xy", width=1, scale=0.005)
ax2.quiverkey(q, 0.9, 0.9, 0.25, "0.25 pixel")
ax2.set_xlabel("x (pix)")
ax2.set_ylabel("y (pix)")
fig.savefig(nf(), dpi=350)

# first look at average centroid deviation in the CCD frame

_df = df[(df.polidName=="nom") & (df.centtype=="blanton")]
_mean = mean[(mean.polidName=="nom") & (mean.centtype=="blanton")]

configidList = list(set(_mean.configid))
rotposList = sorted(list(set(_mean.rotpos)))
positionerIDList = list(set(_mean.positionerID))

movieCounter = 0
for configid in configidList:
    for rotpos in rotposList:
        _m = _mean[
            (_mean.configid==configid) &\
            (_mean.rotpos==rotpos)
        ]

        _ddf = _df[
            (_df.configid==configid) &\
            (_df.rotpos==rotpos)
        ]

        imgNums = sorted(list(set(_ddf.imgNum)))

        for imgNum in imgNums:
            _mx = []
            _my = []
            _x = []
            _y = []

            _mxw = []
            _myw = []
            _xw = []
            _yw = []
            for positionerID in positionerIDList:
                mxy = _m[_m.positionerID==positionerID][["xBlanton", "yBlanton"]].to_numpy()
                assert len(mxy)==1
                mx, my = mxy.flatten()

                xy = _ddf[(_ddf.imgNum==imgNum) & (_ddf.positionerID==positionerID)][["xBlanton", "yBlanton"]].to_numpy()
                assert len(xy)==1
                x, y = xy.flatten()

                _mx.append(mx)
                _my.append(my)
                _x.append(x)
                _y.append(y)

                # wok coords
                mxyw = _m[_m.positionerID==positionerID][["xWokMeasMetrology", "yWokMeasMetrology"]].to_numpy()
                assert len(mxy)==1
                mxw, myw = mxyw.flatten()

                xyw = _ddf[(_ddf.imgNum==imgNum) & (_ddf.positionerID==positionerID)][["xWokMeasMetrology", "yWokMeasMetrology"]].to_numpy()
                assert len(xy)==1
                xw, yw = xy.flatten()

                _mxw.append(mxw)
                _myw.append(myw)
                _xw.append(xw)
                _yw.append(yw)


            mxy = numpy.array([_mx, _my]).T
            xy = numpy.array([_x, _y]).T
            dxy = xy - mxy

            # subtract the means (just translation)
            dxy3 = dxy - numpy.mean(dxy, axis=0)

            mxyw = numpy.array([_mxw, _myw]).T
            xyw = numpy.array([_xw, _yw]).T
            dxyw = xyw - mxyw
            print("mean offset wok?", numpy.mean(numpy.linalg.norm(dxyw,axis=1)))
            # print(numpy.sum(dxyw))




            zimgNum = ("%i"%imgNum).zfill(4)

            fig, axs = plt.subplots(2,2,figsize=(10,10))
            axs = axs.flatten()

            q = axs[0].quiver(mxy[:,0], mxy[:,1], dxy[:,0], dxy[:,1], angles="xy", units="xy", width=4, scale=0.001)
            axs[0].quiverkey(q, 0.9, 0.9, 0.1, "0.1 pixel\n(12 $\mu m$)")
            axs[0].set_xlabel("x (pix)")
            axs[0].set_ylabel("y (pix)")
            axs[0].set_xlim([1000,7000])
            axs[0].set_ylim([0,6000])
            axs[0].set_aspect("equal")

            axs[0].set_title("dxy")

            imgNumStr = ("%i"%movieCounter).zfill(4)
            # imgName = "movie/global-%s.png"%(imgNumStr)
            # fig.savefig(imgName, dpi=200)

            q = axs[1].quiver(mxy[:,0], mxy[:,1], dxy3[:,0], dxy3[:,1], angles="xy",units="xy", width=4, scale=0.001)
            axs[1].quiverkey(q, 0.9, 0.9, 0.1, "0.1 pixel\n(12 $\mu m$)")
            axs[1].set_xlabel("x (pix)")
            # axs[1].set_ylabel("y (pix)")
            axs[1].set_xlim([1000,7000])
            axs[1].set_ylim([0,6000])
            axs[1].set_title("dxy - <dxy>")
            axs[1].set_aspect("equal")

            # fig, ax = plt.subplots(figsize=(10,10))

            q = axs[2].quiver(mxyw[:,0], mxyw[:,1], dxyw[:,0], dxyw[:,1], angles="xy",units="xy", width=2, scale=1)
            axs[2].quiverkey(q, 0.1, 0.1, 0.120, "120 $\mu m$")
            axs[2].set_xlabel("x (mm)")
            axs[2].set_ylabel("y (mm)")
            axs[2].set_xlim([-350, 350])
            axs[2].set_ylim([-350, 350])
            axs[2].set_title("dxyWok")
            axs[2].set_aspect("equal")

            # imgName = "movie/sim-%s.png"%(imgNumStr)
            # fig.savefig(imgName, dpi=200)

            # fig, ax = plt.subplots(figsize=(10,10))



            # q = axs[3].quiver(mxyw[:,0], mxyw[:,1], dxyw[:,0], dxyw[:,1], angles="xy",units="xy", width=4*UM_PER_PIX*MM_PER_UM) #, scale=(UM_PER_PIX*MM_PER_UM)/0.001)
            # axs[3].quiverkey(q, 0.9, 0.9, 12*MM_PER_UM, "12 $\mu m$")
            # axs[3].set_xlabel("x wok (mm)")
            # axs[3].set_ylabel("y wok (mm)")
            axs[3].set_xlim([-350, 350])
            axs[3].set_ylim([-350, 350])
            # axs[3].set_title("dxy wok - <dxy wok>")
            axs[3].set_aspect("equal")




            imgName = "movie/dxy-pix-%s.png"%(imgNumStr)
            fig.suptitle("config=%i  rot=%.2f imgNum=%s"%(configid, rotpos, zimgNum))
            plt.tight_layout()
            fig.savefig(imgName, dpi=200)

            plt.close("all")

            movieCounter += 1





# plt.show()

# plt.show()

# tstart=time.time() # takes 17 minutes
# sns.jointplot(x="xWokMeasMetrology_dm", y="yWokMeasMetrology_dm", hue="centtype", kind="kde", data=_df)
# print("kdeplot took", (time.time()-tstart)/60)
# plt.show()


#ax2.hist(xy[yName], bins=bins, color=color, alpha=1, density=True, histtype="step", label=label)




# print("load took", (time.time()-t1)/60)

# t1 = time.time()
# tossout = df[df.wokErr > 0.5]
# print("tossout took", (time.time()-t1)/60)

# plt.figure()
# plt.hist(df.wokErr.to_numpy(), bins=400)
# plt.show()

# import pdb; pdb.set_trace()