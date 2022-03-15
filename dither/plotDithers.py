import pandas
pandas.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import numpy
from skimage.transform import SimilarityTransform
import seaborn as sns


def plotDither(mjd, design, fvc=False, plot=True):
    df = pandas.read_csv("dither-%i-%i-ap.csv"%(mjd, design))
    print("exposures", list(set(df.expNum)))
    df = df[df.fvc==fvc]
    if len(list(set(df.configuration_id)))==1:
        print("skipping, only one configuration!")
        return pandas.DataFrame({
            "xStar": [],
            "yStar": [],
            "xFiber": [],
            "yFiber": []
        })

    sciFibers = df[(df.isSky == False)] # & (df.activeFiber == True)]
    # sciFibers = sciFibers[sciFibers.flux > 0]
    sciFibers["synthmag"] = 2.5*numpy.log10(sciFibers.flux)
    # import pdb; pdb.set_trace()
    base = sciFibers[df.isParent == True]


    fiberIds = list(set(sciFibers.fiberId))
    xTest = []
    yTest = []
    xMeas = []
    yMeas = []

    sciFibers["fluxnorm"] = 1
    for fiberId in sorted(fiberIds):
        _df = sciFibers[sciFibers.fiberId == fiberId]
        _base = base[base.fiberId == fiberId]

        bottomFlux = _df.nsmallest(4, "flux")
        fluxThresh = numpy.mean(bottomFlux.flux) + 2*numpy.std(bottomFlux.flux)

        keep = _df.copy()
        keep = keep[keep.flux > fluxThresh]
        keep["fluxnorm"] = keep.flux / numpy.sum(keep.flux)

        if len(keep) == 0:
            print("no fibers above background? skipping", fiberId)
            continue

        for expNum in keep.expNum:
            fn = keep[keep.expNum==expNum]["fluxnorm"]
            _idx = (sciFibers.expNum==expNum) & (sciFibers.fiberId==fiberId)
            sciFibers.at[_idx, "fluxnorm"] = fn
            # import pdb; pdb.set_trace()

        mx = numpy.sum(keep.fluxnorm*keep.xwok)
        my = numpy.sum(keep.fluxnorm*keep.ywok)

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
        plt.figure()
        plt.plot(sciFibers.hmag, sciFibers.synthmag, '.k')
        plt.plot(base.hmag, base.synthmag, 'xr')
        plt.title("%i - %i - fvc=%s"%(mjd, design, str(fvc)))
        plt.xlim([8,14])

        plt.figure(figsize=(10,10))
        sns.scatterplot(x="xwok", y="ywok", hue="fluxnorm", size="fluxnorm", data=sciFibers)
        plt.plot(base.xwok, base.ywok, 'xr')
        plt.axis("equal")


        plt.figure(figsize=(10,10))
        plt.quiver(xMeas, yMeas, dx, dy, angles="xy")
        plt.title("APOGEE MJD=%i Design=%i FVC=%s"%(mjd, design, str(fvc)))
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

    mjdDesign = [
        [59595, 35897],
        [59595, 35947],
        [59596, 35887],
        [59596, 35919],
    ]

    nofvcList = []
    fvcList = []
    for mjd, design in mjdDesign:
        # nofvcList.append(plotDither(mjd, design, plot=True))
        fvcList.append(plotDither(mjd, design, fvc=True, plot=True))
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
        plt.figure(figsize=(10,10))
        plt.title("meas")
        plt.quiver(xyStar[:,0], xyStar[:,1], dxy[:,0]*5, dxy[:,1]*5, color="black", units="xy", angles="xy", width=1)
        plt.figure(figsize=(10,10))
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
    # plt.figure(figsize=(10,10))
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

