from coordio.transforms import FVCTransformLCO
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
from coordio.utils import fitsTableToPandas
from astropy.io import fits
from coordio.defaults import calibration
import numpy

pt = calibration.positionerTable.reset_index()
fc = calibration.fiducialCoords.reset_index()
wc = calibration.wokCoords.reset_index()
pt = pt.merge(wc, on="holeID")

xwok = pt.xWok.to_numpy() + pt.dx.to_numpy()
ywok = pt.yWok.to_numpy() + pt.dy.to_numpy()
_positionerID = pt.positionerID.to_numpy()
_holeType = pt.holeType.to_numpy()

mjd = 59741
maxFinalDist = 1.5

# met illuminated
img1 = "/Volumes/futa/osu/lco/%i/proc-fimg-fvclab-0002.fits"%(mjd)
ff = fits.open(img1)
imgData = ff[1].data

posAngles = fitsTableToPandas(ff["POSANGLES"].data)
IPA = 135.4
centtpe = "sep"

fvct = FVCTransformLCO(
    imgData,
    posAngles,
    IPA,
    plotPathPrefix="metrology",
    # positionerTable=pt,
    # wokCoords=wc,
    # fiducialCoords=fc
)

fvct.extractCentroids(centroidMinNpix=20, backgroundSigma=2.5)
# increase max final distance to get matches easier
# robots are a bit off (mostly trans/rot/scale)
fvct.fit(centType="sep", maxFinalDist=maxFinalDist)

# first find fiducials, transform them to CCD pix
tf = fvct.fullTransform.simTrans

holeID = fc.holeID.to_numpy()
xyWok = fc[["xWok","yWok"]].to_numpy()
xyPix = tf.inverse(xyWok)

plt.imshow(equalize_hist(imgData), origin="lower")
plt.savefig("allmet.png", dpi=350)
plt.figure()
plt.imshow(imgData, origin="lower")
# plt.plot(xyPix[:,0], xyPix[:,1], 'o', mec="red", mfc="none", ms="10")

for hid, (x, y) in zip(holeID, xyPix):
    boxSize = 20
    xCen = int(x)
    yCen = int(y)
    plt.xlim([xCen-boxSize, xCen+boxSize])
    plt.ylim([yCen-boxSize, yCen+boxSize])
    plt.title("fiducial: " + hid)
    plt.savefig("fiducial-%s.png"%hid, dpi=200)

############## now find metrology fiber locations ###############
# _cut = fvct.positionerTableMeas[~fvct.positionerTableMeas.wokErrWarn]
xyWok = fvct.positionerTableMeas[["xWokReportMetrology", "yWokReportMetrology"]].to_numpy()
holeID = fvct.positionerTableMeas.holeID.to_numpy()
positionerID = fvct.positionerTableMeas.positionerID.to_numpy()
warn = fvct.positionerTableMeas.wokErrWarn.to_numpy()
xyPix = tf.inverse(xyWok)

print("missing met positioners", set(fvct.positionerTableMeas[fvct.positionerTableMeas.wokErrWarn]["positionerID"]))

for hid, pid, _warn, (x, y) in zip(holeID, positionerID, warn, xyPix):
    boxSize = 20
    xCen = int(x)
    yCen = int(y)
    plt.xlim([xCen-boxSize, xCen+boxSize])
    plt.ylim([yCen-boxSize, yCen+boxSize])
    tt = plt.title("positioner %i: "%pid + hid)
    plt.savefig("met-positioner%i-%s.png"%(pid,hid), dpi=200)


# boss fiber plots
img1 = "/Volumes/futa/osu/lco/%i/proc-fimg-fvclab-0008.fits"%(mjd)
ff = fits.open(img1)
imgData = ff[1].data
plt.figure()
plt.imshow(imgData, origin="lower")
for hid, pid, _warn, (x, y) in zip(holeID, positionerID, warn, xyPix):
    boxSize = 20
    xCen = int(x)
    yCen = int(y)
    plt.xlim([xCen-boxSize, xCen+boxSize])
    plt.ylim([yCen-boxSize, yCen+boxSize])
    tt = plt.title("positioner %i: "%pid + hid)
    plt.savefig("boss-positioner%i-%s.png"%(pid,hid), dpi=200)

# apogee fiber plots
img1 = "/Volumes/futa/osu/lco/%i/proc-fimg-fvclab-0005.fits"%(mjd)
ff = fits.open(img1)
imgData = ff[1].data
plt.figure()
plt.imshow(imgData, origin="lower")

for hid, pid, _warn, (x, y) in zip(holeID, positionerID, warn, xyPix):
    ii = numpy.argwhere(pid==_positionerID)
    ht = _holeType[int(ii)]
    if "Apogee" not in ht:
        continue
    boxSize = 20
    xCen = int(x)
    yCen = int(y)
    plt.xlim([xCen-boxSize, xCen+boxSize])
    plt.ylim([yCen-boxSize, yCen+boxSize])
    tt = plt.title("positioner %i: "%pid + hid)
    plt.savefig("ap-positioner%i-ap-%s.png"%(pid,hid), dpi=200)


    # import pdb; pdb.set_trace()