import glob
from astropy.io import fits
from astropy.wcs import WCS
import sys
import pandas
from datetime import datetime
from multiprocessing import Pool
from functools import partial
import time
import numpy
from coordio.utils import fitsTableToPandas
from coordio.defaults import PLATE_SCALE, calibration # mm/deg
from coordio.conv import tangentToWok
import peewee
from sdssdb.peewee.sdss5db import database, catalogdb
database.set_profile('operations')
from skimage.exposure import equalize_hist
import matplotlib.pyplot as plt
import sep

gfaCoords = calibration.gfaCoords.reset_index()


GIMG_BASE = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/gcam/lco"

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
        (catalogdb.Gaia_DR2.phot_g_mean_mag < 18)
    )

    return list(results)


def solve6(mjd, imgNum, extract=False):
    imgNumStr = str(imgNum).zfill(4)
    for gfaNum in range(1,7):
        imgFile = GIMG_BASE + "/%i/proc-gimg-gfa%is-%s.fits"%(mjd,gfaNum,imgNumStr)
        ff = fits.open(imgFile)
        if extract:
            centroids = getCentroids(ff[1].data)
        else:
            centroids = fitsTableToPandas(ff["CENTROIDS"].data)
        print("got %i centroids"%len(centroids))

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
        iHat = gfaRow[["ix", "iy", "iz"]].to_numpy()
        jHat = gfaRow[["jx", "jy", "jz"]].to_numpy()
        kHat = gfaRow[["kx", "ky", "kz"]].to_numpy()
        b = gfaRow[["xWok", "yWok", "zWok"]].to_numpy()

        # expected field center
        raField = ff["RAFIELD"]
        decField = ff["DECFIELD"]
        paField = ff["FIELDPA"]
        # guide offsets
        offRA = ff["OFFRA"]
        offDec = ff["OFFDEC"]
        offPA = ff["OFFPA"]

        skyMeas = w.pixel_to_world(centroids.x, centroids.y)
        raMeas = skyMeas.ra.degree
        decMeas = skyMeas.dec.degree
        centroids["raMeas"] = raMeas
        centroids["decMeas"] = decMeas

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

        plt.figure(figsize=(10,4))
        plt.hist(centroids.npix) #, bins=numpy.linspace(0,100,100))
        plt.ylim([0,50])
        plt.title("npix")
        plt.savefig("npix_gfa%i.pdf"%gfaNum)
        plt.close('all')
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    solve6(59843, 22, extract=True)





