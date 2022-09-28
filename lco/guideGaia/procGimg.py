import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
import sys
import pandas
from datetime import datetime
from multiprocessing import Pool
from functools import partial
import time
import numpy
from coordio.utils import fitsTableToPandas, wokxy2radec, radec2wokxy
from coordio.defaults import PLATE_SCALE, calibration, POSITIONER_HEIGHT # mm/deg
from coordio.conv import tangentToWok, wokToTangent, tangentToGuide
from coordio.transforms import arg_nearest_neighbor
import peewee
from sdssdb.peewee.sdss5db import database, catalogdb
database.set_profile('operations')
from skimage.exposure import equalize_hist
import matplotlib.pyplot as plt
import sep
from skimage import filters
from skimage.registration import phase_cross_correlation

gfaCoords = calibration.gfaCoords.reset_index()


GIMG_BASE = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/gcam/lco"
GAIA_EPOCH = 2457206

def getImgNums(mjd, configid, gfaID=1):
    allFiles = glob.glob(GIMG_BASE + "/%i/proc-gimg-gfa%is-*.fits"%(mjd,gfaID))
    imgNums = []
    for f in allFiles:
        ff = fits.open(f)
        if ff[1].header["CONFIGID"] == configid:
            imgNum = int(f.split("-")[-1].split(".fits")[0])
            imgNums.append(imgNum)
    return imgNums

def getMetaData(imgNum, mjd):
    imgNumStr = str(imgNum).zfill(4)
    _fwhm = []
    _solved = []
    _imgNum = []
    _gfaID = []
    _nStars = []
    _fieldRA = []
    _fieldDec = []
    _fieldPA = []
    _mjd = []
    _exptime = []
    _dateObs = []
    _grms = []
    _offra = []
    _offdec = []
    _offpa = []
    _configid = []
    for gfaID in range(1,7):
        imgFile = GIMG_BASE + "/%i/proc-gimg-gfa%is-%s.fits"%(mjd,gfaID,imgNumStr)
        ff = fits.open(imgFile)
        try:
            _fwhm.append(ff[1].header["FWHM"])
        except:
            _fwhm.append(numpy.nan)
        _solved.append(ff[1].header["SOLVED"])
        _imgNum.append(imgNum)
        _gfaID.append(gfaID)
        _nStars.append(len(ff["CENTROIDS"].data))
        _fieldRA.append(ff[1].header["RAFIELD"])
        _fieldDec.append(ff[1].header["DECFIELD"])
        _fieldPA.append(ff[1].header["FIELDPA"])
        _mjd.append(mjd)
        _exptime.append(ff[1].header["EXPTIME"])
        _configid.append(ff[1].header["CONFIGID"])
        dateObs = datetime.fromisoformat(ff[1].header["DATE-OBS"])
        _dateObs.append(dateObs)
        try:
            _grms.append(ff[1].header["RMS"])
        except:
            _grms.append(numpy.nan)
        _offra.append(ff[1].header["OFFRA"])
        _offdec.append(ff[1].header["OFFDEC"])
        _offpa.append(ff[1].header["OFFPA"])


    df = pandas.DataFrame(
        {
        "fwhm": _fwhm,
        "solved": _solved,
        "imgNum": _imgNum,
        "gfaID": _gfaID,
        "nStars": _nStars,
        "fieldRA": _fieldRA,
        "fieldDec": _fieldDec,
        "fieldPA": _fieldPA,
        "mjd": _mjd,
        "exptime": _exptime,
        "dateObs": _dateObs,
        "grms": _grms,
        "offra": _offra,
        "offdec": _offdec,
        "offpa": _offpa,
        "configid": _configid
        }
    )

    return df


def doConfig(mjd, configid):
    imgNums = getImgNums(mjd, configid)

    p = Pool(25)
    _getMetaData = partial(getMetaData, mjd=mjd)
    tstart = time.time()
    dfList = p.map(_getMetaData, imgNums)
    print("took", time.time()-tstart)

    # dfList = [getMetaData(imgNum, mjd) for imgNum in imgNums]
    if dfList:
        df = pandas.concat(dfList)
        return df
    else:
        return None


def gimgMeta():
    mjdConfigList = [
        [59835, 10000193],
        [59835, 10000196],
        [59835, 10000197],
        [59836, 10000202],
        [59836, 10000203],
        [59836, 10000204],
        [59837, 10000205],
        [59837, 10000207],
        [59837, 10000208],
        [59837, 10000209],
        [59838, 10000259],
        [59839, 10000275],
        [59839, 10000277],
        [59839, 10000278],
        [59843, 10000279],
        [59843, 10000280],
        [59843, 10000281],

    ]

    dfList = []
    for mjd, configid in mjdConfigList:
        print("processing", mjd, configid)
        dfList.append(doConfig(mjd,configid))

    dfList = [x for x in dfList if x is not None]
    df = pandas.concat(dfList)
    df.to_csv("gimgMeta.csv")


def getCentroids(data):
    data = numpy.array(data, dtype=numpy.float64)
    bkg = sep.Background(data)
    bkg_image = bkg.back()
    bkg_rms = bkg.rms()
    data_sub = data - bkg
    objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
    objects = fitsTableToPandas(objects)
    objects = objects[objects.npix>50]
    return objects

def queryGaia(raCen, decCen, radius=0.08):
    # all inputs in degrees
    MAX_MAG = 18

    results = catalogdb.Gaia_DR2.select(
        catalogdb.Gaia_DR2.solution_id,
        catalogdb.Gaia_DR2.source_id,
        catalogdb.Gaia_DR2.ra,
        catalogdb.Gaia_DR2.dec,
        catalogdb.Gaia_DR2.phot_g_mean_mag,
        catalogdb.Gaia_DR2.parallax,
        catalogdb.Gaia_DR2.pmra,
        catalogdb.Gaia_DR2.pmdec
    ).where(
        (peewee.fn.q3c_radial_query(
            catalogdb.Gaia_DR2.ra,
            catalogdb.Gaia_DR2.dec,
            raCen,
            decCen,
            radius)
        ) & \
        (catalogdb.Gaia_DR2.phot_g_mean_mag < MAX_MAG)
    )

    import pdb; pdb.set_trace()
    return list(results)


def getShift(xyGaia, xyDetect):
    """
    xyGaia Gaia Sources
    xyDetect CCD detections

    """
    n = len(xyGaia)
    m = len(xyDetect)
    dx = numpy.zeros((n,m))
    dy = numpy.zeros((n,m))
    for ii, (x,y) in enumerate(xyGaia):
        dx[ii, :] = x - xyDetect[:,0]
        dy[ii, :] = y - xyDetect[:,1]
    # import pdb; pdb.set_trace()
    _dx = numpy.median(dx)
    _dy = numpy.median(dy)
    print(_dx, _dy)

    return _dx, _dy


def getShift2(xyGaia, xyDetect):
    """ use "fake" images to get cross correlation solution
    between the two lists, takes 1.5 seconds on my machine
    """
    tstart = time.time()
    gaiaFull = numpy.zeros((2048,2048)) #CCD size
    detectFull = numpy.zeros((2048,2048)) #CCD size
    xyGaiaInt = numpy.array(xyGaia, dtype=int)
    xyDetectInt = numpy.array(xyDetect, dtype=int)

    # gaiaFull[xyGaiaInt] = 1
    # detectFull[xyDetectInt] = 1

    for x,y in xyGaiaInt:
        gaiaFull[x,y] = 1

    for x,y in xyDetectInt:
        detectFull[x,y] = 1

    # give both a gaussian blur
    gaiaFull = filters.gaussian(gaiaFull, sigma=10)
    detectFull = filters.gaussian(detectFull, sigma=10)


    (xOff, yOff), error, phasediff = phase_cross_correlation(gaiaFull, detectFull, space="real")
    # import pdb; pdb.set_trace()
    print("detected shift", xOff, yOff)
    print("took %.2f seconds\n\n"%(time.time()-tstart))

    return xOff, yOff


class ProcGimg(object):
    def __init__(self, filename, gfaID, extract=True, focalScale=1):
        """
        Parameters
        ------------------
        filename : string
            path to proc-gimg file
        gfaID : int
            gfa id (1-6)
        extract : bool
            if True, re-extract centroids.  Else, use extractions
            present from fits file
        focalScale : float
            scale to use for coordio conversions

        """
        self.filename = filename
        self.gfaID = gfaID
        self.extract = extract
        self.focalScale = focalScale

        self.ff = fits.open(filename)
        self.astroNetSolved = self.ff[1].header["SOLVED"]

        if extract:
            self.centroids = self._extract()
        else:
            self.centroids = fitsTableToPandas(self.ff["CENTROIDS"].data)

        gfaRow = gfaCoords[gfaCoords.id == gfaID]
        self.b = gfaRow[["xWok", "yWok", "zWok"]].to_numpy()
        self.iHat = gfaRow[["ix", "iy", "iz"]].to_numpy()
        self.jHat = gfaRow[["jx", "jy", "jz"]].to_numpy()
        self.kHat = gfaRow[["kx", "ky", "kz"]].to_numpy()

        if self.astroNetSolved:
            self.wcs = WCS(self.ff[1].header)
            # if there is a wcs solved determine the chip
            # center in degrees O.5 pixel shift is because
            # astropy takes (0,0) to be center of LL pixel
            # and coordio uses (0,0) is LL corner of LL pixel
            sky = self.wcs.pixel_to_world(1024.5,1024.5)
            # edge = w.pixel_to_world(0,0)
            # sky corner to center separation is 0.058 deg
            # (use this for querying radius for gaia)
            self.astroNetCenter = numpy.array([sky.ra.degree, sky.dec.degree])
        else:
            self.wcs = None
            self.astroNetCenter = None

        # next predict the chip center using coordio
        # propogate (0,0) tangent to the sky
        # calculation includes intentional cherno offsets
        raExpect, decExpect, warn = wokxy2radec(
            self.b[0], self.b[1], "GFA", self.raField, self.decField,
            self.paField, "LCO", self.dateObs.jd, focalScale=self.focalScale
        )
        self.coordioCenter = numpy.array([raExpect, decExpect])



    @property
    def offRA(self):
        # ra offset in degrees
        return (ff[1].header["OFFRA"] + ff[1].header["AOFFRA"])/3600.

    @property
    def offDec(self):
        return (self.ff[1].header["OFFDEC"] + self.ff[1].header["AOFFDEC"])/3600.

    @property
    def offPA(self):
        return (self.ff[1].header["OFFPA"] + self.ff[1].header["AOFFPA"])/3600.

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
        return Time(self.ff[1].header["DATE-OBS"], format="iso", scale="tai")

    @property
    def nDetect(self):
        return len(self.centroids)

    @property
    def configid(self):
        return self.ff[1].header["CONFIGID"]

    @property
    def deignid(self):
        return self.ff[1].header["DESIGNID"]

    @property
    def exptime(self):
        return self.ff[1].header["EXPTIME"]



    def _extract(self, minNPix=50):
        """
        Parameters
        -----------
        minNPix : int
            minimum number of pixels for bone fide detection
        """
        data = numpy.array(self.ff[1].data, dtype=numpy.float64)
        bkg = sep.Background(data)
        bkg_image = bkg.back()
        bkg_rms = bkg.rms()
        data_sub = data - bkg
        objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
        objects = fitsTableToPandas(objects)
        objects = objects[objects.npix>minNPix]
        return objects

    def _getGuideStars(self, maxMag=18, queryRadius=0.08):
        """
        Parameters
        --------------
        maxMag : float
            maximum g mag for a gaia source
        queryRadius : float
            degrees, radius around the chip center to query for

        Get gaia sources that should land on or near this chip.
        Filter on gaia g mag.  propogate ra, decs to chip coordinates,
        finally trim so that only sources on the chip exist
        """

        results = catalogdb.Gaia_DR2.select(
            catalogdb.Gaia_DR2.solution_id,
            catalogdb.Gaia_DR2.source_id,
            catalogdb.Gaia_DR2.ra,
            catalogdb.Gaia_DR2.dec,
            catalogdb.Gaia_DR2.phot_g_mean_mag,
            catalogdb.Gaia_DR2.parallax,
            catalogdb.Gaia_DR2.pmra,
            catalogdb.Gaia_DR2.pmdec
        ).where(
            (peewee.fn.q3c_radial_query(
                catalogdb.Gaia_DR2.ra,
                catalogdb.Gaia_DR2.dec,
                self.coordioCenter[0],
                self.coordioCenter[1],
                queryRadius)
            ) & \
            (catalogdb.Gaia_DR2.phot_g_mean_mag < maxMag)
        )

        results = list(results)

        # convert results to DataFrame
        solution_id = []
        source_id = []
        ra = []
        dec = []
        g_mag = []
        parallax = []
        pmra = []
        pmdec = []

        for obj in results:
            solution_id.append(obj.solution_id)
            source_id.append(obj.source_id)
            ra.append(obj.ra)
            dec.append(obj.dec)
            g_mag.append(obj.phog_g_mean_mag)
            parallax.append(obj.parallax)
            pmra.append(obj.pmra)
            pmdec.append(obj.pmdec)





def solve6(mjd, imgNum, extract=False):
    imgNumStr = str(imgNum).zfill(4)
    rasMeas = []
    decsMeas = []
    rasExpect1 = []
    decsExpect1 = []
    rasExpect2 = []
    decsExpect2 = []
    for gfaNum in range(1,7):
        imgFile = GIMG_BASE + "/%i/proc-gimg-gfa%is-%s.fits"%(mjd,gfaNum,imgNumStr)
        ff = fits.open(imgFile)
        if extract:
            centroids = getCentroids(ff[1].data)
        else:
            centroids = fitsTableToPandas(ff["CENTROIDS"].data)
        print("got %i centroids"%len(centroids))

        # expected field center
        raField = ff[1].header["RAFIELD"]
        decField = ff[1].header["DECFIELD"]
        paField = ff[1].header["FIELDPA"]
        # guide offsets
        offRA = (ff[1].header["OFFRA"] + ff[1].header["AOFFRA"])/3600.
        offDec = (ff[1].header["OFFDEC"] + ff[1].header["AOFFDEC"])/3600.
        offPA = (ff[1].header["OFFPA"] + ff[1].header["AOFFPA"])/3600.
        dateObs = Time(ff[1].header["DATE-OBS"], format="iso", scale="tai")

        # find the chip center on sky
        # first via WCS
        w = WCS(ff[1].header)
        sky = w.pixel_to_world(1024,1024)
        # edge = w.pixel_to_world(0,0)
        # sky corner to center separation is 0.058 deg (use this for querying radius for gaia)
        raCenMeas = sky.ra.degree
        decCenMeas = sky.dec.degree

        # then by coordio
        gfaRow = gfaCoords[gfaCoords.id == gfaNum]
        b = gfaRow[["xWok", "yWok", "zWok"]].to_numpy()
        iHat = gfaRow[["ix", "iy", "iz"]].to_numpy()
        jHat = gfaRow[["jx", "jy", "jz"]].to_numpy()
        kHat = gfaRow[["kx", "ky", "kz"]].to_numpy()
        # xWok = float(gfaRow.xWok)
        # yWok = float(gfaRow.yWok)


        _raField = raField + offRA/numpy.cos(numpy.radians(decField))
        _decField = decField + offDec
        _paField = paField + offPA
        raExpect1, decExpect1, warn = wokxy2radec(
            gfaRow.xWok, gfaRow.yWok, "GFA", _raField, _decField, _paField,
            "LCO", dateObs.jd, focalScale=1
        )

        raExpect2, decExpect2, warn = wokxy2radec(
            gfaRow.xWok, gfaRow.yWok, "GFA", raField, decField, paField,
            "LCO", dateObs.jd, focalScale=1
        )


        # import pdb; pdb.set_trace()
        skyMeas = w.pixel_to_world(centroids.x, centroids.y)
        raMeas = skyMeas.ra.degree
        decMeas = skyMeas.dec.degree
        centroids["raMeas"] = raMeas
        centroids["decMeas"] = decMeas

        #save for plotting
        rasExpect1.append(float(raExpect1))
        decsExpect1.append(float(decExpect1))
        rasExpect2.append(float(raExpect2))
        decsExpect2.append(float(decExpect2))
        rasMeas.append(raCenMeas)
        decsMeas.append(decCenMeas)


        # next query for gaia stars around raCen/decCen
        tstart = time.time()
        results = queryGaia(raCenMeas, decCenMeas)
        # results = catalogdb.Gaia_DR2.select(catalogdb.Gaia_DR2.solution_id, catalogdb.Gaia_DR2.source_id, catalogdb.Gaia_DR2.ra, catalogdb.Gaia_DR2.dec, catalogdb.Gaia_DR2.phot_g_mean_mag, catalogdb.Gaia_DR2.parallax, catalogdb.Gaia_DR2.pmra, catalogdb.Gaia_DR2.pmdec).where(peewee.fn.q3c_radial_query(catalogdb.Gaia_DR2.ra, catalogdb.Gaia_DR2.dec, raCenMeas, decCenMeas, 0.08))
        # results = list(results)
        ras = numpy.array([x.ra for x in results])
        decs = numpy.array([x.dec for x in results])
        mag = numpy.array([x.phot_g_mean_mag for x in results])
        pmras = numpy.array([x.pmra for x in results])
        pmdecs = numpy.array([x.pmdec for x in results])
        # keep = mag < 18
        # ras = ras[keep]
        # decs = decs[keep]
        # mag = mag[keep]
        flux = numpy.exp(-mag/2.5)*50000
        print("query took", time.time()-tstart)

        plt.figure(figsize=(10,10))
        plt.scatter(ras, decs, s=flux, c="red")
        plt.scatter(centroids.raMeas, centroids.decMeas, s=15, c="blue", marker="x") #s=centroids.flux/5000, c="blue")
        plt.savefig("gfa%i.pdf"%gfaNum)

        # plt.figure(figsize=(10,4))
        # plt.hist(centroids.npix) #, bins=numpy.linspace(0,100,100))
        # plt.ylim([0,50])
        # plt.title("npix")
        # plt.savefig("npix_gfa%i.pdf"%gfaNum)
        plt.close('all')

        # query gaia based on expected chip centers
        results = queryGaia(raExpect1[0], decExpect1[0])
        # results = catalogdb.Gaia_DR2.select(catalogdb.Gaia_DR2.solution_id, catalogdb.Gaia_DR2.source_id, catalogdb.Gaia_DR2.ra, catalogdb.Gaia_DR2.dec, catalogdb.Gaia_DR2.phot_g_mean_mag, catalogdb.Gaia_DR2.parallax, catalogdb.Gaia_DR2.pmra, catalogdb.Gaia_DR2.pmdec).where(peewee.fn.q3c_radial_query(catalogdb.Gaia_DR2.ra, catalogdb.Gaia_DR2.dec, raCenMeas, decCenMeas, 0.08))
        # results = list(results)
        ras = numpy.array([x.ra for x in results])
        decs = numpy.array([x.dec for x in results])
        mag = numpy.array([x.phot_g_mean_mag for x in results])
        pmras = numpy.array([x.pmra for x in results])
        pmdecs = numpy.array([x.pmdec for x in results])
        flux = numpy.exp(-mag/2.5)*50000

        # propogate sources to expected GFA coords
        xWok, yWok, fieldWarn, HA, PA = radec2wokxy(
            ras, decs, GAIA_EPOCH, "GFA", raField, decField, paField,
            "LCO", dateObs.jd, focalScale=1, pmra=pmras, pmdec=pmdecs
        )
        zWok = numpy.array([POSITIONER_HEIGHT]*len(xWok))

        # remove any nans (why are they here?)
        keep = ~numpy.isnan(xWok)
        xWok = xWok[keep]
        yWok = yWok[keep]
        zWok = zWok[keep]
        flux = flux[keep]

        # convert from xyWok to gfa pixels
        xT, yT, zT = wokToTangent(
            xWok, yWok, zWok, b, iHat, jHat, kHat
        )
        # convert from tangent to guide
        xExpect, yExpect = tangentToGuide(xT,yT)

        keep = (xExpect < 2048) & (xExpect > 0) & (yExpect < 2048) & (yExpect >= 0)
        xExpect = xExpect[keep]
        yExpect = yExpect[keep]
        flux = flux[keep]

        keep = (centroids.x < 2048) & (centroids.x >= 0) & (centroids.y < 2048) & (centroids.y >=0 )
        centroids = centroids[keep]
        # keep = centroids.npix > 100
        # centroids = centroids[keep]


        plt.figure(figsize=(10,10))
        plt.scatter(centroids.x, centroids.y, s=centroids.flux/5000, c="red")
        plt.scatter(xExpect, yExpect, s=flux, c="blue", marker="x") #s=centroids.flux/5000, c="blue")
        plt.plot([0,2048], [0,0], '-k')
        plt.plot([0,2048], [2048,2048], '-k')
        plt.plot([0,0], [0,2048], '-k')
        plt.plot([2048,2048], [0,2048], '-k')
        plt.savefig("ccdCoords%i.pdf"%gfaNum)

        xyGaia = numpy.array([xExpect, yExpect]).T
        xyDetect = centroids[["x", "y"]].to_numpy()

        dx,dy = getShift2(xyGaia, xyDetect)
        xDetectShift = centroids.x + dx
        yDetectShift = centroids.y + dy

        xyDetectShift = numpy.array([xDetectShift, yDetectShift]).T

        ida, idb, dist = arg_nearest_neighbor(xyDetectShift, xyGaia, atol=10)
        # 10 pixel max distance after shift

        xyMatched = xyDetectShift[ida]


        # print("distances", dist)

        plt.figure(figsize=(10,10))
        plt.scatter(xDetectShift, yDetectShift, s=centroids.flux/5000, c="red")
        plt.plot(xyMatched[:,0], xyMatched[:,1], "o", ms=20, mfc="none", mec="black")
        plt.scatter(xExpect, yExpect, s=flux, c="blue", marker="x") #s=centroids.flux/5000, c="blue")
        plt.plot([0,2048], [0,0], '-k')
        plt.plot([0,2048], [2048,2048], '-k')
        plt.plot([0,0], [0,2048], '-k')
        plt.plot([2048,2048], [0,2048], '-k')
        plt.savefig("ccdCoordsShift%i.pdf"%gfaNum)





    # rasMeas     = numpy.array(rasMeas    )
    # decsMeas    = numpy.array(decsMeas   )
    # rasExpect1  = numpy.array(rasExpect1 )
    # decsExpect1 = numpy.array(decsExpect1)
    # rasExpect2  = numpy.array(rasExpect2 )
    # decsExpect2 = numpy.array(decsExpect2)

    # import pdb; pdb.set_trace()

    plt.figure(figsize=(10,10))
    plt.plot(rasMeas, decsMeas, 'xr', label="meas")
    plt.plot(rasExpect1, decsExpect1, 'o', mfc="none", mec="black", ms=3, label="+off")
    plt.plot(rasExpect2, decsExpect2, 'o', color="cyan", mfc="none", mec="cyan", ms=3, label="no off")
    plt.legend()
    plt.savefig("expect.pdf")

    plt.close("all")


        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    solve6(59843, 22, extract=True)





