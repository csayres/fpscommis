import glob
from astropy.io import fits
from astropy.wcs import WCS
# from astropy.coordinates import angular_separation
# from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.time import Time, TimeDelta
from astropy import units as u
import pandas
from datetime import datetime
import numpy
from coordio.utils import fitsTableToPandas, wokxy2radec, radec2wokxy, wokCurveAPO
from coordio.site import Site
from coordio.sky import Observed, ICRS
from coordio.telescope import Field, FocalPlane
from coordio.wok import Wok
from coordio.defaults import calibration, INST_TO_WAVE, POSITIONER_HEIGHT
from coordio.conv import tangentToWok, wokToTangent, tangentToGuide, guideToTangent
import matplotlib.pyplot as plt
from skimage import filters
import socket
import matplotlib.pyplot as plt
from multiprocessing import Pool

numpy.random.seed(0)

GFA_TIME_OFF_SECS = 37

gfaCoords = calibration.gfaCoords.reset_index()

GIMG_BASE = "/data/gcam"
nProcs = 3

hostname = socket.gethostname()
if "Conors" in hostname:
    GIMG_BASE = "/Users/csayres/fpscommis/ptModel"
    nProcs = 3

if hostname == "sdssadmin":
    GIMG_BASE = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/gcam/apo"
    nProcs = 20


class ProcGimg(object):
    def __init__(self, filename, mjd):
        self.filename = filename
        self.mjd = mjd
        self.ff = fits.open(filename)
        self.solved = self.ff[1].header["SOLVED"]
        # get img number
        self.imgNum = int(filename.split("-")[-1].split(".fits")[0])
        self.gfaNum = int(filename.split("-")[-2].split("gfa")[-1].split("n")[0])

        self.site = Site("APO")
        self.site.set_time(self.dateObs.jd)

        self.actualAlt = None
        self.actualAz = None
        self.actualAzNorth = None

        if self.grms is not None:
            # get expected alt/az for telescope
            icrsCen = ICRS([[self.raField, self.decField]])
            obsCen = Observed(icrsCen, site=self.site).squeeze()
            self.actualAlt = obsCen[0]
            self.actualAzNorth = obsCen[1]
            self.actualAz = -1*self.actualAzNorth + 180
            diff = self.actualAz - self.tccAz
            if diff < -10:
                self.actualAz += 360
            if diff > 10:
                self.actualAz -= 360.

            # assert numpy.abs(self.actualAz - self.tccAz) < 10

    @property
    def grms(self):
        try:
            return self.ff[1].header["RMS"]
        except:
            return None

    @property
    def tccAz(self):
        # tcc's definition of az is south through east
        return self.ff[1].header["AZ"]

    @property
    def tccAzNorth(self):
        # coordio's definition of az is from north through east
        az = -1*(self.tccAz - 180)
        if az >= 360:
            az -= 360
        if az < 0:
            az += 360
        return az

    @property
    def tccAlt(self):
        return self.ff[1].header["ALT"]

    @property
    def tccRot(self):
        return self.ff[1].header["IPA"]

    @property
    def raField(self):
        return self.ff[1].header["RAFIELD"] + self.offRA/numpy.cos(numpy.radians(self.decField))

    @property
    def decField(self):
        return self.ff[1].header["DECFIELD"] + self.offDec

    @property
    def paField(self):
        return self.ff[1].header["FIELDPA"] + self.offPA

    @property
    def dateObs(self):
        astroTime = Time(self.ff[1].header["DATE-OBS"], format="iso", scale="tai")
        # it appears that GFA 1 and 5 have a 37 second offset (roughly the
        # difference between tai and utc), blanket apply it for now
        # and only use gfa1 or gfa5 data later
        return astroTime + TimeDelta(37.0 * u.s)

    @property
    def offRA(self):
        # ra offset in degrees
        return (self.ff[1].header["OFFRA"])/3600. # + self.ff[1].header["AOFFRA"])/3600.

    @property
    def offDec(self):
        return (self.ff[1].header["OFFDEC"])/3600. # + self.ff[1].header["AOFFDEC"])/3600.

    @property
    def configid(self):
        return self.ff[1].header["CONFIGID"]

    @property
    def designid(self):
        return self.ff[1].header["DESIGNID"]

    @property
    def exptime(self):
        return self.ff[1].header["EXPTIME"]

    def output(self):
        d = {}
        d["mjd"] = self.mjd
        d["imgNum"] = self.imgNum
        d["gfaNum"] = self.gfaNum
        d["configid"] = int(self.configid)
        d["designid"] = int(self.designid)
        d["exptime"] = self.exptime
        d["dateObsMJD"] = self.dateObs.mjd
        d["solved"] = self.solved
        d["grms"] = self.grms
        d["tccAlt"] = self.tccAlt
        d["tccAz"] = self.tccAz
        d["tccRot"] = self.tccRot
        d["actAlt"] = self.actualAlt
        d["actAz"] = self.actualAz
        return d


# class ProcGimg2(object):
#     def __init__(self, filename, gfaID, focalScale=1):
#         """
#         Parameters
#         ------------------
#         filename : string
#             path to proc-gimg file
#         gfaID : int
#             gfa id (1-6)
#         extract : bool
#             if True, re-extract centroids.  Else, use extractions
#             present from fits file
#         focalScale : float
#             scale to use for coordio conversions

#         """
#         self.filename = filename
#         self.gfaID = gfaID
#         self.focalScale = focalScale

#         self.ff = fits.open(filename)
#         self.solved = self.ff[1].header["SOLVED"]


#         self.centroids = fitsTableToPandas(self.ff["CENTROIDS"].data)

#         gfaRow = gfaCoords[gfaCoords.id == gfaID]
#         self.b = gfaRow[["xWok", "yWok", "zWok"]].to_numpy().squeeze()
#         self.iHat = gfaRow[["ix", "iy", "iz"]].to_numpy().squeeze()
#         self.jHat = gfaRow[["jx", "jy", "jz"]].to_numpy().squeeze()
#         self.kHat = gfaRow[["kx", "ky", "kz"]].to_numpy().squeeze()

#         self.site = Site("APO")
#         self.site.set_time(self.dateObs.jd)

#         altAz = numpy.array([[self.tccAlt, self.coordioAz]])
#         self.obsCenReport = Observed(altAz, site=self.site, wavelength=INST_TO_WAVE["GFA"])
#         altAz = numpy.array([[self.raField, self.decField]])
#         self.obsCenActual = Observed(altAz, site=self.site, wavelength=INST_TO_WAVE["GFA"])

#         if self.solved:
#             self.wcs = WCS(self.ff[1].header)
#             # if there is a wcs solved determine the chip
#             # center in degrees O.5 pixel shift is because
#             # astropy takes (0,0) to be center of LL pixel
#             # and coordio uses (0,0) is LL corner of LL pixel
#             sky = self.wcs.pixel_to_world(1024.5,1024.5)

#             self.raDecChipMeas = numpy.array([sky.ra.degree, sky.dec.degree])
#             _icrsMeas = ICRS([self.raDecChipMeas])
#             _obsCen = Observed(_icrsMeas, site=self.site, wavelength=INST_TO_WAVE["GFA"])
#             self.altAzChipMeas = numpy.array(_obsCen).squeeze()
#             self.altAzChipExpect = self.getAltAzChipExpect([self.tccAlt, self.coordioAz, self.paField])

#             # compute expected alt/az from coordio
#             # rWok = numpy.linalg.norm(self.b[:2])
#             # zWok = POSITIONER_HEIGHT + wokCurveAPO(rWok)
#             # wok = Wok(
#             #     numpy.array([[self.b[0], self.b[1], zWok]]),
#             #     site=self.site, obsAngle=self.paField
#             # )
#             # focal = FocalPlane(wok, wavelength=INST_TO_WAVE["GFA"],
#             #     site=self.site, fpScale=self.focalScale
#             # )
#             # fieldReport = Field(focal, field_center=self.obsCenReport)
#             # fieldActual = Field(focal, field_center=self.obsCenActual)
#             # obsReportChip = Observed(fieldReport, site=self.site, wavelength=INST_TO_WAVE["GFA"])
#             # obsActualChip = Observed(fieldActual, site=self.site, wavelength=INST_TO_WAVE["GFA"])

#     def getAltAzChipExpect(self, altAzPaBore):
#         alt, az, pa = altAzPaBore
#         _altAz = numpy.array([[alt,az]])
#         _obsCen = Observed(_altAz, site=self.site, wavelength=INST_TO_WAVE["GFA"])

#         rWok = numpy.linalg.norm(self.b[:2])
#         zWok = POSITIONER_HEIGHT + wokCurveAPO(rWok)
#         wok = Wok(
#             numpy.array([[self.b[0], self.b[1], zWok]]),
#             site=self.site, obsAngle=pa
#         )
#         focal = FocalPlane(wok, wavelength=INST_TO_WAVE["GFA"],
#             site=self.site, fpScale=self.focalScale
#         )
#         field = Field(focal, field_center=_obsCen)
#         observed = Observed(field, site=self.site, wavelength=INST_TO_WAVE["GFA"])

#         return numpy.array(observed).squeeze()


#     def raDecField2AltAz(self):
#         icrsCen = ICRS([[self.raField, self.decField]])
#         obsCen = Observed(icrsCen, site=self.site).squeeze()
#         import pdb; pdb.set_trace()


#     @property
#     def offRA(self):
#         # ra offset in degrees
#         return (self.ff[1].header["OFFRA"] + self.ff[1].header["AOFFRA"])/3600.

#     @property
#     def offDec(self):
#         return (self.ff[1].header["OFFDEC"] + self.ff[1].header["AOFFDEC"])/3600.

#     @property
#     def offPA(self):
#         return (self.ff[1].header["AOFFPA"])/3600.

#     @property
#     def raField(self):
#         return self.ff[1].header["RAFIELD"] + self.offRA/numpy.cos(numpy.radians(self.decField))

#     @property
#     def decField(self):
#         return self.ff[1].header["DECFIELD"] + self.offDec

#     @property
#     def paField(self):
#         return self.ff[1].header["FIELDPA"] + self.offPA

#     @property
#     def dateObs(self):
#         return Time(self.ff[1].header["DATE-OBS"], format="iso", scale="tai")

#     @property
#     def configid(self):
#         return self.ff[1].header["CONFIGID"]

#     @property
#     def designid(self):
#         return self.ff[1].header["DESIGNID"]

#     @property
#     def exptime(self):
#         return self.ff[1].header["EXPTIME"]

#     @property
#     def focus(self):
#         return self.ff[1].header["M2PISTON"]

#     @property
#     def grms(self):
#         try:
#             return self.ff[1].header["RMS"]
#         except:
#             return None

#     @property
#     def tccAz(self):
#         return self.ff[1].header["AZ"]

#     @property
#     def tccAlt(self):
#         return self.ff[1].header["ALT"]

#     @property
#     def tccRot(self):
#         return self.ff[1].header["IPA"]

#     @property
#     def coordioAz(self):
#         az = -1*(self.tccAz - 180)
#         if az >= 360:
#             az -= 360
#         if az < 0:
#             az += 360
#         return az


# class GuideBundle(object):
#     def __init__(self, mjd, imgNum):
#         self.mjd = mjd
#         self.imgNum = imgNum

#         imgNumStr = str(imgNum).zfill(4)
#         self.gfaDict = {}
#         for gfaNum in range(1,7):
#             imgFile = GIMG_BASE + "/%i/proc-gimg-gfa%in-%s.fits"%(mjd,gfaNum,imgNumStr)
#             self.gfaDict[gfaNum] = ProcGimg(imgFile, gfaNum)

#     @property
#     def tccAz(self):
#         return self.gfaDict[1].tccAz

#     @property
#     def tccAlt(self):
#         return self.gfaDict[1].tccAlt

#     @property
#     def coordioAz(self):
#         return self.gfaDict[1].coordioAz

#     @property
#     def paField(self):
#         return self.gfaDict[1].paField

#     @property
#     def grms(self):
#         return self.gfaDict[1].grms

#     @property
#     def exptime(self):
#         return self.gfaDict[1].exptime


#     def ptErr(self, altAzPaBore=None):
#         if altAzPaBore is None:
#             altAzPaBore = [self.tccAlt, self.coordioAz, self.paField]
#         for gfa in self.gfaDict.values():
#             altAzExpect = gfa.getAltAzChipExpect(altAzPaBore)
#             altAzMeas = gfa.altAzChipMeas
#             dAltAz = altAzExpect - altAzMeas
#             lon1 = numpy.radians(altAzExpect[1])
#             lat1 = numpy.radians(90 - altAzExpect[0])
#             lon2 = numpy.radians(altAzMeas[1])
#             lat2 = numpy.radians(90 - altAzMeas[0])
#             angSep = angular_separation(lon1, lat1, lon2, lat2)
#             angSep = numpy.degrees(angSep)
#             print("gfa %i sep %.2f"%(gfa.gfaID, angSep))
#             print("dAltAz", dAltAz)
#             print("\n")
#             gfa.raDecField2AltAz()

def doOneMJD(mjd):
    print("processing mjd", mjd)
    dictList = []
    files = glob.glob(GIMG_BASE+"/%i/proc-gimg-gfa1n-*.fits"%mjd)
    for file in files:
        gimg = ProcGimg(file,mjd)
        if gimg.grms is None:
            continue
        if gimg.grms < 1 and gimg.offRA == 0 and gimg.offDec == 0:
            dictList.append(gimg.output())
    df = pandas.DataFrame(dictList)
    return df


def collectData(mjdMin, mjdMax, filename="ptData.csv"):
    mjdList = list(range(mjdMin, mjdMax+1))

    # dfList = []
    # for mjd in mjdList:
    #     dfList.append(doOneMJD(mjd))

    p = Pool(nProcs)
    dfList = p.map(doOneMJD, mjdList)
    df = pandas.concat(dfList)

    df.to_csv(filename, index=False)




if __name__ == "__main__":
    # gb = GuideBundle(59825, 22)
    # gb.ptErr()
    # import pdb; pdb.set_trace()
    collectData(59825, 59874)
    # import pdb; pdb.set_trace()
