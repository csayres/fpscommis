import numpy
from skimage.transform import EuclideanTransform, SimilarityTransform
import pandas
from coordio.defaults import calibration, POSITIONER_HEIGHT
import matplotlib.pyplot as plt

# gfaCoord columns
# site,id,xWok,yWok,zWok,ix,iy,iz,jx,jy,jz,kx,ky,kz
# note zwok is set to 0 because we assume we're using a flat wok
# which may come back to bite us someday?
# should probably set this to POSITIONER_HEIGHT?  <---yes it should!!
zWok = 0 #POSITIONER_HEIGHT
kx = 0
ky = 0
kz = 1
iz = 0
jz = 0

# GFA scale = 13.5 (um/pix)
# scale = 13.5/1000 # (mm/pix)

fc = calibration.fiducialCoords.reset_index()

# file used is GFA_Metrology.xlsx LCO tab, and fps_calibrations fiducialCoords.csv,
# to build fps_calibrations gfaCoords.csv file

# FIF1 is at +x in tangent coords, FIF2 is at -x in tangent coords
# assumed that (1,1) means center of LL pixel...
# Pixel(1,1) ll is at (-x,-y) tangent
# Pixel(1,2048) ul is at (-x,+y) tangent
# Pixel(2048,2048) ur is at (+x,+y) tangent
# Pixel(2048,1) lr is at (+x,-y) tangent

# csv exported is duPontGFATableExport.csv



# create mapping for FIF 1 and 2 for each GFA
# [FIF1, FIF2]
gfaFIFMap = {
    "1": ["F3", "F2"],
    "2": ["F6", "F5"],
    "3": ["F9", "F8"],
    "4": ["F12", "F11"],
    "5": ["F15", "F14"],
    "6": ["F18", "F17"]
}

# this is an extracted csv from ricks GFA measurements
gfaMet = pandas.read_csv("sloanGFATableExport.csv")

_id = []
_xWok = []
_yWok = []
_ix = []
_iy = []

def pix2cmm(xPix,yPix,sextant):
    _met = gfaMet[gfaMet.Sextant==sextant]
    xCMM = _met.a_x0 + _met.a_xx*xPix + _met.a_xy*yPix
    yCMM = _met.a_y0 + _met.a_yx*xPix + _met.a_yy*yPix
    return numpy.array([xCMM,yCMM]).squeeze()

_site = []
_id = []
_xWok = []
_yWok = []
_zWok = []
_ix = []
_iy = []
_iz = []
_jx = []
_jy = []
_jz = []
_kx = []
_ky = []
_kz = []



for ii in range(1,7):
    # iter over sextants 1-6
    fif1, fif2 = gfaFIFMap[str(ii)]

    xyWokFif1 = fc[fc.holeID==fif1][["xWok", "yWok"]].to_numpy().flatten()
    xyWokFif2 = fc[fc.holeID==fif2][["xWok", "yWok"]].to_numpy().flatten()

    print("wok dist fifs", numpy.linalg.norm(xyWokFif2-xyWokFif1))

    _met = gfaMet[gfaMet.Sextant==ii]
    xyCMMFif1 = _met[["xFIF1", "yFIF1"]].to_numpy().flatten()
    xyCMMFif2 = _met[["xFIF2", "yFIF2"]].to_numpy().flatten()

    # xyCen = numpy.array([1024.5, 1024.5])
    xyCen = pix2cmm(1024.5,1024.5,ii)
    xyLL = pix2cmm(1,1, ii)
    xyLR = pix2cmm(2048,1, ii)
    # xyUL = pix2cmm(1,2048, ii)

    # xHatCCD = (xyLR-xyLL)/numpy.linalg.norm(xyLR-xyLL)
    # _yHatCCD = (xyUL-xyLL)/numpy.linalg.norm(xyUL-xyLL)

    # yHatCCD = numpy.array([[0,-1],[1,0]]) @ xHatCCD

    # check the directions
    # print("x dot y", xHatCCD @ yHatCCD, yHatCCD-_yHatCCD)

    # _xyCMMFif1 = xyCMMFif1 - xyCen
    # _xyCMMFif2 = xyCMMFif2 - xyCen

    # get direction on wok from Fif2 to Fif1
    # fit translation and rotation between CMM and Wok
    # ec = EuclideanTransform()
    st = SimilarityTransform()
    xyWok = numpy.array([xyWokFif1, xyWokFif2])
    xyCMM = numpy.array([xyCMMFif1, xyCMMFif2])

    # ec.estimate(xyCMM, xyWok)
    st.estimate(xyCMM, xyWok)
    # import pdb; pdb.set_trace()

    # compute pixels to wok
    cmmPix = numpy.array([xyLL, xyLR, xyCen])
    wokLL, wokLR, wokCen = st(cmmPix)

    xHat = (wokLR-wokLL)/numpy.linalg.norm(wokLR-wokLL)
    yHat = numpy.array([[0,-1],[1,0]]) @ xHat

    # dirWokFif = (xyWokFif1-xyWokFif2)/numpy.linalg.norm(xyWokFif1-xyWokFif2)
    # dirCMMFif = (xyCMMFif1-xyCMMFif2)/numpy.linalg.norm(xyCMMFif1-xyCMMFif2)

    print("xHat sextant", ii, xHat, yHat)
    # import pdb; pdb.set_trace()

    _site.append("APO")
    _id.append(ii)
    _xWok.append(wokCen[0])
    _yWok.append(wokCen[1])
    _zWok.append(zWok)
    _ix.append(xHat[0])
    _iy.append(xHat[1])
    _iz.append(iz)
    _jx.append(yHat[0])
    _jy.append(yHat[1])
    _jz.append(jz)
    _kx.append(kx)
    _ky.append(ky)
    _kz.append(kz)



    # calculate fiducial locations in CCD frame


    # convert

    #cmm coords, find chip directions
    # plt.figure()
    # plt.text(*pix2cmm(1,0,ii), "LL")
    # plt.text(*pix2cmm(1,2048,ii), "UL")
    # plt.text(*pix2cmm(2048,2048,ii), "UR")
    # plt.text(*pix2cmm(2048,1,ii), "LR")
    # plt.text(*xyCMMFif1, "FIF1")
    # plt.text(*xyCMMFif2, "FIF2")

    # plt.plot(*pix2cmm(1,0,ii), alpha=0)
    # plt.plot(*pix2cmm(1,2048,ii), alpha=0)
    # plt.plot(*pix2cmm(2048,2048,ii), alpha=0)
    # plt.plot(*pix2cmm(2048,1,ii), alpha=0)
    # plt.plot(*xyCMMFif1, alpha=0)
    # plt.plot(*xyCMMFif2, alpha=0)

    # plt.title("sextant %i"%ii)
    # plt.axis("equal")

# plt.show()

df = pandas.DataFrame({
    "site": _site,
    "id": _id,
    "xWok": _xWok,
    "yWok": _yWok,
    "zWok": _zWok,
    "ix": _ix,
    "iy": _iy,
    "iz": _iz,
    "jx": _jx,
    "jy": _jy,
    "jz": _jz,
    "kx": _kx,
    "ky": _ky,
    "kz": _kz,
})

df.to_csv("gfaCoordsAPO.csv")


    # import pdb; pdb.set_trace()
