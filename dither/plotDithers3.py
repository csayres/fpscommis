import pandas
import matplotlib.pyplot as plt
import numpy
from skimage.transform import SimilarityTransform
import seaborn as sns


def plotDither(mjd, design, camera, plot=True):
    df = pandas.read_csv("ap-boss-dithers-latest.csv")
    df["MJD"] = df.MJD_x
    df["design_id"] = df.design_id_x

    # import pdb; pdb.set_trace()

    df = df[df.MJD==mjd]
    df = df[df.design_id==design]
    df = df[df.camera == camera]

    if len(df) == 0:
        return

    parentConfig = list(set(df.parent_configuration))
    parentConfig.remove(-999)
    assert len(parentConfig)==1
    parentConfig = parentConfig[0]

    base = df[df.configuration_id==parentConfig]
    base = base[base.fvc==False]

    # import pdb; pdb.set_trace()

    df = df[df.fvc==True]

    # import pdb; pdb.set_trace()

    sciFibers = df[(df.isSky == False)] # & (df.activeFiber == True)]
    sciFibers = sciFibers[sciFibers.spectroflux > 0]
    sciFibers["synthmag"] = 2.5*numpy.log10(sciFibers.spectroflux)
    base["synthmag"] = 2.5*numpy.log10(base.spectroflux)
    # import pdb; pdb.set_trace()
    # base = sciFibers[df.isParent == True]

    # normalize flux in scifibers
    # import pdb; pdb.set_trace()
    # fg = pandas.groupby()

    if plot:
        plt.figure()
        plt.plot(sciFibers.hmag, sciFibers.synthmag, '.k')
        plt.plot(base.hmag, base.synthmag, 'xr')
        plt.title("%i - %i - %s"%(mjd, design, camera))
        plt.xlim([8,14])

        plt.figure(figsize=(9,9))
        sns.scatterplot(x="xwok", y="ywok", size="synthmag", style="fvc", data=sciFibers)
        plt.plot(base.xwok, base.ywok, 'xr')


    fiberIds = list(set(sciFibers.fiberId))
    xTest = []
    yTest = []
    xMeas = []
    yMeas = []
    for fiberId in sorted(fiberIds):
        _df = sciFibers[sciFibers.fiberId == fiberId]
        _base = base[base.fiberId == fiberId]

        bottomFlux = _df.nsmallest(4, "spectroflux")
        fluxThresh = numpy.mean(bottomFlux.spectroflux) + 2*numpy.std(bottomFlux.spectroflux)

        keep = _df.copy()
        keep = keep[keep.spectroflux > fluxThresh]
        keep["fluxnorm"] = keep.spectroflux / numpy.sum(keep.spectroflux)

        mx = numpy.sum(keep.fluxnorm*keep.xwok)
        my = numpy.sum(keep.fluxnorm*keep.ywok)

        # import pdb; pdb.set_trace()

        # dont know why this happens sometimes
        if numpy.abs(mx-float(_base.xwok)) > 0.2:
            continue

        # plt.figure()
        # sns.scatterplot()


        xMeas.append(mx)
        yMeas.append(my)

        xTest.append(float(_base.xwok))
        yTest.append(float(_base.ywok))


    xTest = numpy.array(xTest)
    yTest = numpy.array(yTest)
    xMeas = numpy.array(xMeas)
    yMeas = numpy.array(yMeas)

    dx = xTest - xMeas
    dy = yTest - yMeas

    # dxy = numpy.sqrt(dx**2+dy**2)
    # keep = dxy*1000 < 30

    if plot:
        plt.figure(figsize=(9,9))
        plt.quiver(xMeas, yMeas, dx, dy, angles="xy")
        plt.title("%i - %i - %s"%(mjd, design, camera))
        plt.xlabel("xwok (mm)")
        plt.ylabel("ywok (mm)")

        plt.figure()
        plt.hist(numpy.sqrt(dx**2+dy**2), bins=100)
        plt.xlim([0, 0.2])

    return pandas.DataFrame({
        "xStar": xMeas,
        "yStar": yMeas,
        "xFiber": xTest,
        "yFiber": yTest

    })
    # plt.show()


if __name__ == "__main__":

    df = pandas.read_csv("ap-boss-dithers-latest.csv")
    df["MJD"] = df.MJD_x
    df["design_id"] = df.design_id_x

    mjds = list(set(df.MJD))
    mjdDesign = []
    for mjd in mjds:
        _df = df[df.MJD==mjd]
        designs = list(set(_df.design_id))
        for design in designs:
            _df2 = _df[_df.design_id==design]
            if True in _df2.apogeeAssigned:
                camera = "APOGEE"
                print("camera apogee")
            else:
                camera = "b1"
            mjdDesign.append([mjd, design, camera])

    # import pdb; pdb.set_trace()


    nofvcList = []
    fvcList = []
    for mjd, design, camera in mjdDesign:
        # nofvcList.append(plotDither(mjd, design, plot=True))
        fvcList.append(plotDither(mjd, design, camera, plot=True))
        # plt.show()
        # import pdb; pdb.set_trace()


    # nofvc = pandas.concat(nofvcList)
    fvc = pandas.concat(fvcList)


    for vers in [fvc]: #[nofvc, fvc]:
        print("iter\n\n----------------")
        xyStar = vers[["xStar", "yStar"]].to_numpy() # star location
        xyFiber = vers[["xFiber", "yFiber"]].to_numpy() # star location
        dxy = xyFiber - xyStar # points from star to fiber
        nofvcModel = SimilarityTransform()
        nofvcModel.estimate(xyStar, xyFiber)
        print("translation", nofvcModel.translation)
        print("rotation", nofvcModel.rotation)
        print("rotation arcsec", numpy.degrees(nofvcModel.rotation)*3600)
        print("scale", nofvcModel.scale)
        xyModel = nofvcModel(xyStar)
        resids = xyFiber - xyModel
        plt.figure(figsize=(9,9))
        plt.title("meas")
        plt.quiver(xyStar[:,0], xyStar[:,1], dxy[:,0]*5, dxy[:,1]*5, color="black", units="xy", angles="xy", width=1)
        plt.figure(figsize=(9,9))
        plt.title("model")
        plt.quiver(xyStar[:,0], xyStar[:,1], resids[:,0]*5, resids[:,1]*5, color="black", units="xy", angles="xy", width=1)

        errxy = numpy.linalg.norm(dxy, axis=1)
        errresid = numpy.linalg.norm(resids, axis=1)
        plt.figure()
        plt.hist(errxy*1000, histtype="step", color='b', alpha=1, bins=50)
        plt.hist(errresid*1000, histtype="step", color='r', alpha=1, bins=50)
        plt.xlim([0,200])
        print("median errs", numpy.median(errxy)*1000, numpy.median(errresid)*1000)


    plt.show()

    # vers = plotDither(59587,35899)
    # print("iter\n\n----------------")
    # xyStar = vers[["xStar", "yStar"]].to_numpy() # star location
    # xyFiber = vers[["xFiber", "yFiber"]].to_numpy() # star location
    # dxy = xyFiber - xyStar # points from star to fiber
    # nofvcModel = SimilarityTransform()
    # nofvcModel.estimate(xyStar, xyFiber)
    # print("translation", nofvcModel.translation)
    # print("rotation", nofvcModel.rotation)
    # print("rotation arcsec", numpy.degrees(nofvcModel.rotation)*3600)
    # print("scale", nofvcModel.scale)
    # xyModel = nofvcModel(xyStar)
    # resids = xyFiber - xyModel

    # plt.show()



    # x = fvc[:,2]
    # y = fvc[:,3]
    # dx = fvc[:,0] - x
    # dy = fvc[:,1] - y
    # plt.figure(figsize=(9,9))
    # plt.quiver(x, y, dx*5, dy*5, color="red", units="xy", angles="xy", width=1)
    # plt.show()



    # plotDither(59575, 35945, True)
    # plotDither(59575, 35945, False)

    # plotDither(59575, 35929, True)
    # plotDither(59575, 35929, False)

    # plotDither(59578, 35939, True)
    # plotDither(59578, 35939, False)

    # plotDither(59583, 35929, True)
    # plotDither(59583, 35929, False)

    # # ra dec offset?
    # plotDither(59584, 35913, True)
    # plotDither(59584, 35913, False)

    # plotDither(59585, 35899, True)
    # plotDither(59585, 35899, False)

    # plotDither(59585, 36093, True)
    # plotDither(59585, 36093, False)

    # plotDither(59585, 36101, True)
    # plotDither(59585, 36101, False)

    # plt.show()

