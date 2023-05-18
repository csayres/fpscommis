import pandas
import glob
from skimage.transform import SimilarityTransform, EuclideanTransform
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from coordio.defaults import calibration, POSITIONER_HEIGHT
from coordio.conv import tangentToGuide, guideToTangent, tangentToWok


def fitData():
    csvFiles = sorted(glob.glob("vecs*.csv"))
    df = pandas.concat([pandas.read_csv(x) for x in csvFiles])

    mjdList = sorted(list(set(df.mjd)))
    dfList = []
    nIter = 0
    for mjd in mjdList:
        _df = df[df.mjd==mjd]
        imgList = sorted(list(set(_df.imgNum)))
        for imgNum in imgList:
            print("on img num", imgNum)
            _ddf = _df[_df.imgNum==imgNum].copy()
            xyExpect = _ddf[["xWokExpect", "yWokExpect"]].to_numpy()
            xyMeas = _ddf[["xWokMeas", "yWokMeas"]].to_numpy()
            st = SimilarityTransform()
            st.estimate(xyMeas, xyExpect) # remove pointing error
            xyMeasFit = st(xyMeas)
            dxyWok = xyMeasFit - xyExpect
            _ddf["xMeasFit"] = xyMeasFit[:,0]
            _ddf["yMeasFit"] = xyMeasFit[:,1]
            _ddf["dxFit"] = dxyWok[:,0]
            _ddf["dyFit"] = dxyWok[:,1]
            _ddf["scaleFit"] = st.scale
            _ddf["rotFit"] = st.rotation
            _ddf["txFit"] = st.translation[0]
            _ddf["tyFit"] = st.translation[1]
            for gfaID in range(1,7):
                _dddf = _ddf[_ddf.gfaID==gfaID].copy()

                # finally compute offsets after trans/rot/scale global
                # is removed
                xyExpect = _dddf[["xWokExpect", "yWokExpect"]].to_numpy()
                xyMeas = _dddf[["xMeasFit", "yMeasFit"]].to_numpy()
                st = EuclideanTransform()
                st.estimate(xyExpect, xyMeas)
                xyFit = st(xyExpect)
                dxyWok = xyMeas - xyFit
                _dddf["xFitLocal"] = xyFit[:,0]
                _dddf["yFitLocal"] = xyFit[:,1]
                _dddf["dxFitLocal"] = dxyWok[:,0]
                _dddf["dyFitLocal"] = dxyWok[:,1]
                _dddf["rotFitLocal"] = st.rotation
                _dddf["txFitLocal"] = st.translation[0]
                _dddf["tyFitLocal"] = st.translation[1]

                dfList.append(_dddf)
            # nIter += 1
        #     if nIter > 15:
        #         break
        # if nIter > 15:
        #     break

    df = pandas.concat(dfList)


    df.to_csv("init_fits.csv", index=False)


def plot():
    df = pandas.read_csv("init_fits.csv")
    # drGlobal = numpy.sqrt(df.dxFit**2+df.dyFit**2)
    # drLocal = numpy.sqrt(df.dxFitLocal**2+df.dyFitLocal**2)

    # plt.figure()
    # plt.hist(drGlobal,bins=1000)

    # plt.figure()
    # plt.hist(drLocal,bins=1000)

    # plt.figure()
    # plt.hist(df.paCen.to_numpy(),bins=1000)

    # plt.show()
    # import pdb; pdb.set_trace()

    # dfMean = df.groupby(["xWokExpect", "yWokExpect", "xPix", "yPix"]).mean().reset_index()

    # for gfaID in range(1,7):
    #     _df = dfMean[dfMean.gfaID==gfaID]
    #     plt.figure(figsize=(6,6))
    #     plt.quiver(_df.xWokExpect,_df.yWokExpect,_df.dxFit,_df.dyFit,angles="xy",units="xy", width=0.01, scale=0.005)
    #     plt.axis("equal")
    #     plt.title("GFA global %i"%(gfaID))

    #     plt.figure(figsize=(6,6))
    #     plt.quiver(_df.xWokExpect,_df.yWokExpect,_df.dxFitLocal,_df.dyFitLocal,angles="xy",units="xy", width=0.01, scale=0.005)
    #     plt.axis("equal")
    #     plt.title("GFA local %i"%(gfaID))

    dfMeanFit = df.groupby(["gfaID"]).mean().reset_index()

    dr = numpy.sqrt(dfMeanFit.txFitLocal**2+dfMeanFit.tyFitLocal**2)
    dtheta = numpy.arctan2(dfMeanFit.tyFitLocal,dfMeanFit.txFitLocal)

    dfMeanFit["dr"] = dr
    dfMeanFit["dtheta"] = dtheta
    dfMeanFit["drot"] = numpy.degrees(dfMeanFit.rotFitLocal)

    plt.figure()
    sns.scatterplot(x="dr", y="dtheta", hue="gfaID", data=dfMeanFit)

    plt.figure()
    plt.plot(dfMeanFit.gfaID, dfMeanFit.drot, '.k')
    plt.show()

    dfMeanGFA = df.groupby(["gfaID", "xWokExpect", "yWokExpect", "xPix", "yPix"]).mean().reset_index()

    for gfaID in range(1,7):
        _df = dfMeanGFA[dfMeanGFA.gfaID==gfaID]
        plt.figure(figsize=(8,8))
        plt.quiver(_df.xWokExpect,_df.yWokExpect,_df.dxFitLocal,_df.dyFitLocal, width=0.1,scale=0.01)
        plt.axis("equal")
        plt.title("GFA %i"%gfaID)

    plt.show()
    import pdb; pdb.set_trace()

    # plt.show()
    # import pdb; pdb.set_trace()


    # # down sample just one measurement per gfa per img
    # # import pdb; pdb.set_trace()
    # ds = df.groupby(["gfaID", "imgNum", "mjd"]).first().reset_index()
    # ds.to_csv("ds.csv", index=False)

    # gfaNum = 2
    # df = df[df.gfaID==gfaNum]

    mjdList = sorted(list(set(df.mjd)))
    # df = df[df.mjd==mjdList[0]]
    # configList = sorted(list(set(df.configID)))
    # df = df[df.configID==configList[0]]
    # df.to_csv("init_fits_one_mjd.csv", index=False)
    # import pdb; pdb.set_trace()

    for mjd in mjdList:
        _df = df[df.mjd==mjd]
        imgList = sorted(list(set(_df.imgNum)))
        for imgNum in imgList:
            _ddf = _df[_df.imgNum==imgNum]
            x = _ddf.xWokExpect.to_numpy()
            y = _ddf.yWokExpect.to_numpy()
            dx0 = _ddf.dxFitGlobal.to_numpy()
            dy0 = _ddf.dyFitGlobal.to_numpy()

            plt.figure()
            plt.hist(numpy.sqrt(dx0**2+dy0**2))

            dx1 = _ddf.dxFitLocal.to_numpy()
            dy1 = _ddf.dyFitLocal.to_numpy()

            plt.figure()
            plt.hist(numpy.sqrt(dx1**2+dy1**2))

            dx2 = _ddf.dxFitLocal2.to_numpy()
            dy2 = _ddf.dyFitLocal2.to_numpy()

            plt.figure()
            plt.hist(numpy.sqrt(dx2**2+dy2**2))

            plt.figure(figsize=(8,8))
            plt.quiver(x,y,dx0,dy0,angles="xy",units="xy", width=0.1,scale=.01)
            plt.axis("equal")

            plt.figure(figsize=(8,8))
            plt.quiver(x,y,dx1,dy1,angles="xy",units="xy", width=0.1, scale=.01)
            plt.axis("equal")

            plt.figure(figsize=(8,8))
            plt.quiver(x,y,dx2,dy2,angles="xy",units="xy", width=0.1, scale=.01)
            plt.axis("equal")
            plt.show()
            import pdb; pdb.set_trace()


    ds = pandas.read_csv("ds.csv")

    plt.figure()
    plt.plot(numpy.degrees(ds.rotFitGlobal),ds.scaleFitGlobal,'.k')
    plt.xlabel("rotation")
    plt.ylabel("scale")


    plt.figure()
    plt.plot(ds.txFitGlobal,ds.tyFitGlobal,'.k')
    plt.xlabel("dx")
    plt.ylabel("dy")
    plt.axis('equal')
    plt.grid("on")

    for gfaID in range(1,7):
        _df = ds[ds.gfaID==gfaID]
        plt.figure(figsize=(5,5))
        plt.plot(numpy.degrees(_df.rotFitLocal),_df.scaleFitLocal,'.k')
        plt.plot(numpy.degrees(_df.rotFitLocal2),_df.scaleFitLocal2,'.r')
        plt.ylim([.997,1.005])
        plt.xlim([-0.4,0.4])
        plt.xlabel("rotation")
        plt.ylabel("scale")
        plt.grid("on")
        plt.title("GFA %i"%gfaID)


        plt.figure(figsize=(5,5))
        plt.plot(_df.txFitLocal,_df.tyFitLocal,'.k')
        plt.plot(_df.txFitLocal2,_df.tyFitLocal2,'.r')
        plt.ylim([-1,1])
        plt.xlim([-1,1])
        plt.xlabel("dx")
        plt.ylabel("dy")
        # plt.axis('equal')
        plt.grid("on")
        plt.title("GFA %i"%gfaID)


def updateConfig():

    rotMat = numpy.array([
        [numpy.cos(numpy.radians(90)), -numpy.sin(numpy.radians(90))],
        [numpy.sin(numpy.radians(90)), numpy.cos(numpy.radians(90))]
    ])
    # df = pandas.read_csv("init_fits.csv")
    # df = df.groupby(["gfaID"]).mean().reset_index()
    # df.to_csv("mean_fits.csv", index=False)

    df = pandas.read_csv("mean_fits.csv")
    gfaCoords = calibration.gfaCoords.reset_index()


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

    _xyCen = []
    _xyX = []

    # import pdb; pdb.set_trace()


    for gfaID in range(1,7):
        _df = df[df.gfaID==gfaID]
        _gfaCoords = gfaCoords[gfaCoords.id==gfaID]
        b = _gfaCoords[["xWok", "yWok", "zWok"]].to_numpy()
        i = _gfaCoords[["ix", "iy", "iz"]].to_numpy()
        j = _gfaCoords[["jx", "jy", "jz"]].to_numpy()
        k = _gfaCoords[["kx", "ky", "kz"]].to_numpy()

        chipCenPix = numpy.array(tangentToGuide(0,0))
        chipXPix = chipCenPix + numpy.array([800, 0])
        chipCenTang = guideToTangent(chipCenPix[0], chipCenPix[1])
        chipXTang = guideToTangent(chipXPix[0], chipXPix[1])
        chipCenWok = tangentToWok(chipCenTang[0], chipCenTang[1], 0, b, i, j, k)
        chipXWok = tangentToWok(chipXTang[0], chipXTang[1], 0, b, i, j, k)

        chipCenWok = numpy.array([chipCenWok[0], chipCenWok[1]])
        chipXWok = numpy.array([chipXWok[0], chipXWok[1]])

        _xyCen.append(chipCenWok)
        _xyX.append(chipXWok)

        et = EuclideanTransform(
            translation=[float(_df.txFitLocal), float(_df.tyFitLocal)],
            rotation = float(_df.rotFitLocal)
        )

        newCoords = et([chipCenWok, chipXWok])
        newIHat = (newCoords[1] - newCoords[0])/numpy.linalg.norm(newCoords[1] - newCoords[0])
        newJHat = rotMat @ newIHat

        _site.append("APO")
        _id.append(gfaID)
        _xWok.append(newCoords[0][0])
        _yWok.append(newCoords[0][1])
        _zWok.append(0)
        _ix.append(newIHat[0])
        _iy.append(newIHat[1])
        _iz.append(0)
        _jx.append(newJHat[0])
        _jy.append(newJHat[1])
        _jz.append(0)
        _kx.append(0)
        _ky.append(0)
        _kz.append(1)

    d = {}
    d["site"] = _site
    d["id"] = _id
    d["xWok"] = _xWok
    d["yWok"] = _yWok
    d["zWok"] = _zWok
    d["ix"] = _ix
    d["iy"] = _iy
    d["iz"] = _iz
    d["jx"] = _jx
    d["jy"] = _jy
    d["jz"] = _jz
    d["kx"] = _kx
    d["ky"] = _ky
    d["kz"] = _kz
    gfaCoordsNew1 = pandas.DataFrame(d)

    xyCen = numpy.array(_xyCen)
    xyX = numpy.array(_xyX)

    # now attempt to rotate gfa coords such that cherno offsets are 0,0,0
    gfaCoordsNew2 = gfaCoordsNew1.copy()

    paRotRad = numpy.radians(420/3600.)
    paRotMat = numpy.array([
        [numpy.cos(paRotRad), -numpy.sin(paRotRad)],
        [numpy.sin(paRotRad), numpy.cos(paRotRad)]
    ])

    xyCenRot = (paRotMat @ xyCen.T).T
    xyXRot = (paRotMat @ xyX.T).T
    iHatRot = xyXRot - xyCenRot
    iHatRot = numpy.array([x/numpy.linalg.norm(x) for x in iHatRot])
    jHatRot = (rotMat @ iHatRot.T).T

    gfaCoordsNew2["xWok"] = xyCenRot[:,0]
    gfaCoordsNew2["yWok"] = xyCenRot[:,0]
    gfaCoordsNew2["ix"] = iHatRot[:,0]
    gfaCoordsNew2["iy"] = iHatRot[:,1]
    gfaCoordsNew2["jx"] = jHatRot[:,0]
    gfaCoordsNew2["jy"] = jHatRot[:,1]

    # try other direction just in case
    gfaCoordsNew3 = gfaCoordsNew1.copy()

    paRotRad = numpy.radians(-420/3600.)
    paRotMat = numpy.array([
        [numpy.cos(paRotRad), -numpy.sin(paRotRad)],
        [numpy.sin(paRotRad), numpy.cos(paRotRad)]
    ])

    xyCenRot = (paRotMat @ xyCen.T).T
    xyXRot = (paRotMat @ xyX.T).T
    iHatRot = xyXRot - xyCenRot
    iHatRot = numpy.array([x/numpy.linalg.norm(x) for x in iHatRot])
    jHatRot = (rotMat @ iHatRot.T).T

    gfaCoordsNew3["xWok"] = xyCenRot[:,0]
    gfaCoordsNew3["yWok"] = xyCenRot[:,0]
    gfaCoordsNew3["ix"] = iHatRot[:,0]
    gfaCoordsNew3["iy"] = iHatRot[:,1]
    gfaCoordsNew3["jx"] = jHatRot[:,0]
    gfaCoordsNew3["jy"] = jHatRot[:,1]

    gfaCoordsNew1.to_csv("gfaCoordsNew1.csv")
    gfaCoordsNew2.to_csv("gfaCoordsNew2.csv")
    gfaCoordsNew3.to_csv("gfaCoordsNew3.csv")

    # import pdb; pdb.set_trace()
    dx = gfaCoords.xWok-gfaCoordsNew1.xWok
    dy = gfaCoords.yWok-gfaCoordsNew1.yWok
    dr = numpy.sqrt(dx**2+dy**2)
    print(dr)







if __name__ == "__main__":
    # fitData()
    # plot()
    updateConfig()

    plt.show()
