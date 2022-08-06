import numpy
from astropy.io import fits
from coordio.utils import fitsTableToPandas
import matplotlib.pyplot as plt
import pandas
from coordio.defaults import calibration
from coordio.utils import refinexy, simplexy
import sep
from photutils.detection import DAOStarFinder

# image sequence was met only exptime 1 sec, all fibers exptime 1 sec, all
# fibers exptime 2 sec.

bossExclude = [50, 1015, 224, 231, 873, 958] # broken met or boss fiber
apExclude = [151, 773] # broken met or ap fiber

pt = calibration.positionerTable.reset_index()
wc = calibration.wokCoords.reset_index()
pt = pt.merge(wc, on="holeID")
hasAp = numpy.array(["Apogee" in x for x in pt.holeType.to_numpy()])
positionerIDs = pt.positionerID.to_numpy()
positionersBoss = numpy.array(list(set(positionerIDs)-set(bossExclude)))
positionersApogee = numpy.array(list(set(positionerIDs[hasAp])-set(apExclude)))


def getImgFile(imgNum):
    mjd = 59741
    imgStr = ("%i"%imgNum).zfill(4)
    imgFile = "/Volumes/futa/osu/lco/%i/proc-fimg-fvclab-%s.fits"%(mjd, imgStr)
    return imgFile


def extractCentroids(
    fvcImgData,
    centroidMinNpix=5,
    backgroundSigma=1.5,
    simpleSigma=1,
    simpleBoxSize=19,
):

    # rough bias/bg subtract
    imbias = numpy.median(fvcImgData, axis=0)
    imbias = numpy.outer(
        numpy.ones(fvcImgData.shape[0], dtype=numpy.float32),
        imbias
    )
    im = fvcImgData - imbias

    bkg = sep.Background(im) #fvcImgData)
    bkg_image = bkg.back()

    data_sub = im - bkg_image

    # data_sub[data_sub < 10000] = 0

    # t1 = time.time()
    # objects = sep.extract(
    #     data_sub,
    #     backgroundSigma,
    #     err=bkg.globalrms,
    # )
    # # print("sep extract took", time.time()-t1)
    # # print("sep found %i sources"%len(objects))

    # # get rid of obvious bogus detections
    # objects = objects[objects["npix"] > centroidMinNpix]
    # # remove detections near edges of chip
    # # (causes issues for unlucky winpos detections)
    # objects = objects[objects["x"] > 500]
    # objects = objects[objects["x"] < 7500]
    # objects = objects[objects["y"] > 30]
    # objects = objects[objects["y"] < 5970]


    off = 1022  # trim the LR edges refinexy needs square
    im = data_sub[:, off:-off].copy()

    # xSimple, ySimple = refinexy(
    #     im, objects["x"]-off, objects["y"],
    #     psf_sigma=simpleSigma, cutout=simpleBoxSize
    # )

    xSimple, ySimple, flux = simplexy(
        im, psf_sigma=2, dlim=0,
        plim=100, saddle=0, maxper=5000
    )



    xSimple = xSimple + off

    x = list(xSimple)
    y = list(ySimple)
    t = ["simple"] * len(x)

    daofind = DAOStarFinder(fwhm=5, threshold=100)
    sources = daofind(data_sub)

    x.extend(list(sources["xcentroid"]))
    y.extend(list(sources["ycentroid"]))
    t.extend(["dao"]*len(sources))

    # import pdb; pdb.set_trace()

    print("simple found", len(xSimple), "sources")
    print("dao found", len(sources))

    df = pandas.DataFrame({
        "x": x,
        "y": y,
        "type": t
        })

    # objects = pandas.DataFrame(objects)

    # objects["xSimple"] = xSimple
    # objects["ySimple"] = ySimple

    return df


def compileData():

    firstImg = 80
    lastImg = 82 #229
    configID = 0

    _fcm = []
    _cnts = []
    _ptm = []

    for imgNum in range(firstImg, lastImg+1):
        imgFile = getImgFile(imgNum)

        ff = fits.open(imgFile)

        fbiMet = bool(ff[1].header["LED1"])
        fbiAp = bool(ff[1].header["LED3"])
        fbiBoss = bool(ff[1].header["LED4"])
        exptime = ff[1].header["exptime"]

        print("imgnum", imgNum)
        if not True in [fbiAp, fbiBoss]:
            # only metrology on, this must be
            # a new configuration
            configID += 1

        fcm = fitsTableToPandas(ff["FIDUCIALCOORDSMEAS"].data)
        cnts = extractCentroids(ff[1].data)
        ptm = fitsTableToPandas(ff["POSITIONERTABLEMEAS"].data)

        for _table in [fcm, cnts, ptm]:
            _table["imgNum"] = imgNum
            _table["exptime"] = exptime
            _table["sciOn"] = fbiAp and fbiBoss
            _table["imgScale"] = ff[1].header["FVC_SCL"]
            _table["configID"] = configID

        _fcm.append(fcm)
        _cnts.append(cnts)
        _ptm.append(ptm)

    fcm = pandas.concat(_fcm)
    cnts = pandas.concat(_cnts)
    ptm = pandas.concat(_ptm)

    fcm.to_csv("fcm.csv")
    cnts.to_csv("cnts.csv")
    ptm.to_csv("ptm.csv")


def assocCentroids():
    fcm = pandas.read_csv("fcm.csv")
    cnts = pandas.read_csv("cnts.csv")
    ptm = pandas.read_csv("ptm.csv")
    configIDs = fcm.configID.to_numpy()

    for robotID in positionerIDs:
        # check if this robot has a boss or ap
        if robotID != 494:
            continue
        print("processing robot", robotID)
        hasBoss = robotID in positionersBoss
        hasAp = robotID in positionersApogee
        if not (hasAp or hasBoss):
            print("skipping robot", robotID, "nothing to measure")
            continue
        for configID in configIDs[:1]:
            _ptm = ptm[
                (ptm.configID==configID) &
                (ptm.positionerID==robotID)
            ]
            metOnly = _ptm[_ptm.sciOn == False]
            xyMet = metOnly[["x", "y"]].to_numpy()
            if bool(metOnly.wokErrWarn.iloc[0]):
                print("skipping robot", robotID, "config", configID, "met wok warn")
                continue  # this doesn't happen ever i guess
            for exptime in [1,2]:
                _cnts = cnts[
                    (cnts.configID == configID) &
                    (cnts.exptime == exptime) &
                    (cnts.sciOn == True)
                ]
                imgNum = list(set(_cnts.imgNum))
                assert len(imgNum) == 1
                imgNum = imgNum[0]



                xyCents = _cnts[_cnts.type=="dao"][["x", "y"]].to_numpy()
                xyCentsSimp = _cnts[_cnts.type=="simple"][["x", "y"]].to_numpy()
                dxyCents = xyCents - xyMet
                drCents = numpy.linalg.norm(dxyCents, axis=1)
                asort = numpy.argsort(drCents)
                drCents = drCents[asort][:3]
                dxyCents = dxyCents[asort][:3]

                nexpect = numpy.sum([hasBoss, hasAp])

                imgData = fits.open(getImgFile(imgNum))[1].data
                plt.figure(figsize=(8,8))
                plt.title("robot: %i  nexpect: %i  exptime: %.1f"%(robotID, nexpect, exptime))
                plt.imshow(imgData)
                plt.plot(xyCents[:,0], xyCents[:,1], 'd', ms=5, mfc='none', mec="tab:orange")
                plt.plot(xyCentsSimp[:,0], xyCentsSimp[:,1], 'o', ms=5, mfc='none', mec="tab:cyan")
                plt.show()
                # plt.xlim([xyMet[0][0]-200, xyMet[0][0]+200])
                # plt.ylim([xyMet[0][1]-200, xyMet[0][1]+200])
                # plt.show()



                # plt.figure()
                # plt.title("robot: %i  nexpect: %i  exptime: %.1f"%(robotID, nexpect, exptime))
                # plt.plot(dxyCents[:,0], dxyCents[:,1], '.k')
                # plt.xlim([-10,10])
                # plt.ylim([-10,10])
                # plt.axis("equal")
            plt.show()


            import pdb; pdb.set_trace()

            # import pdb; pdb.set_trace()



if __name__ == "__main__":
    compileData()
    assocCentroids()
