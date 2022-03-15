import pandas
import matplotlib.pyplot as plt
import numpy
from skimage.transform import SimilarityTransform
import seaborn as sns


def plotDither(mjd, design, plot=True):
    df = pandas.read_csv("guideOff-%i-%i-ap.csv"%(mjd, design))
    df = df[(df.expNum==40250010) | (df.expNum==40250011) | (df.expNum==40250012)]

    sciFibers = df[(df.isSky == False)] # & (df.activeFiber == True)]
    sciFibers = sciFibers[sciFibers.activeFiber == True]
    sciFibers["synthmag"] = 2.5*numpy.log10(sciFibers.flux)
    # sciFibers["logx"] = numpy.log10(sciFibers.hmag)
    # base = sciFibers[df.isParent == True]
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()

    cmap = sns.color_palette("hls",3)

    plt.figure(figsize=(14,10))
    sns.lineplot(x="hmag", y="synthmag", hue="expNum", data=sciFibers, alpha=0.7, lw=2, palette=cmap)
    # ax = plt.gca()
    # ax.set(xscale="log")
    # plt.plot(sciFibers.hmag, sciFibers.synthmag, '.k')
    plt.title("%i - %i"%(mjd, design))
    plt.xlim([8,14])

    # fiberIds = list(set(sciFibers.fiberId))

    # xTest = []
    # yTest = []
    # xMeas = []
    # yMeas = []
    # for fiberId in sorted(fiberIds):
    #     _df = sciFibers[sciFibers.fiberId == fiberId]
    #     _base = base[base.fiberId == fiberId]

    #     bottomFlux = _df.nsmallest(4, "flux")
    #     fluxThresh = numpy.mean(bottomFlux.flux) + 2*numpy.std(bottomFlux.flux)

    #     keep = _df.copy()
    #     keep = keep[keep.flux > fluxThresh]
    #     keep["fluxnorm"] = keep.flux / numpy.sum(keep.flux)

    #     mx = numpy.sum(keep.fluxnorm*keep.xFocal)
    #     my = numpy.sum(keep.fluxnorm*keep.yFocal)

    #     # dont know why this happens sometimes
    #     if numpy.abs(mx-float(_base.xFocal)) > 0.2:
    #         continue


    #     xMeas.append(mx)
    #     yMeas.append(my)

    #     xTest.append(float(_base.xFocal))
    #     yTest.append(float(_base.yFocal))


    # xTest = numpy.array(xTest)
    # yTest = numpy.array(yTest)
    # xMeas = numpy.array(xMeas)
    # yMeas = numpy.array(yMeas)

    # dx = xMeas - xTest
    # dy = yMeas - yTest

    # if plot:
    #     plt.figure(figsize=(10,10))
    #     plt.quiver(xMeas, yMeas, -dx, -dy, angles="xy")
    #     plt.title("APOGEE MJD=%i Design=%i FVC=%s"%(mjd, design, str(fvc)))
    #     plt.xlabel("xFocal (mm)")
    #     plt.ylabel("yFocal (mm)")

    #     plt.figure()
    #     plt.hist(numpy.sqrt(dx**2+dy**2), bins=100)
    #     plt.xlim([0, 0.2])

    # return pandas.DataFrame({
    #     "xStar": xMeas,
    #     "yStar": yMeas,
    #     "xFiber": xTest,
    #     "yFiber": yTest

    # })
    # plt.show()


if __name__ == "__main__":
    plotDither(59587, 35899)

    # mjdDesign = [
    #     # [59575, 35945],
    #     [59575, 35929],
    #     [59583, 35929],
    #     [59584, 35913],
    #     [59585, 35899],
    #     [59585, 36093],
    #     [59585, 36101],
    #     [59586, 35889],
    # ]

    # nofvcList = []
    # fvcList = []
    # for mjd, design in mjdDesign:
    #     nofvcList.append(plotDither(mjd, design, plot=False))
    #     fvcList.append(plotDither(mjd, design, fvc=True, plot=False))
    #     # plt.show()
    #     # import pdb; pdb.set_trace()


    # nofvc = pandas.concat(nofvcList)
    # fvc = pandas.concat(fvcList)


    # for vers in [nofvc, fvc]:
    #     print("iter\n\n----------------")
    #     xyStar = vers[["xStar", "yStar"]].to_numpy() # star location
    #     xyFiber = vers[["xFiber", "yFiber"]].to_numpy() # star location
    #     dxy = xyFiber - xyStar # points from star to fiber
    #     nofvcModel = SimilarityTransform()
    #     nofvcModel.estimate(xyStar, xyFiber)
    #     print("translation", nofvcModel.translation)
    #     print("rotation", nofvcModel.rotation)
    #     print("rotation arcsec", numpy.degrees(nofvcModel.rotation)*3600)
    #     print("scale", nofvcModel.scale)
    #     xyModel = nofvcModel(xyStar)
    #     resids = xyFiber - xyModel
    #     plt.figure(figsize=(10,10))
    #     plt.title("meas")
    #     plt.quiver(xyStar[:,0], xyStar[:,1], dxy[:,0]*5, dxy[:,1]*5, color="black", units="xy", angles="xy", width=1)
    #     plt.figure(figsize=(10,10))
    #     plt.title("model")
    #     plt.quiver(xyStar[:,0], xyStar[:,1], resids[:,0]*5, resids[:,1]*5, color="black", units="xy", angles="xy", width=1)

    #     errxy = numpy.linalg.norm(dxy, axis=1)
    #     errresid = numpy.linalg.norm(resids, axis=1)
    #     plt.figure()
    #     plt.hist(errxy*1000, histtype="step", color='b', alpha=1, bins=50)
    #     plt.hist(errresid*1000, histtype="step", color='r', alpha=1, bins=50)
    #     plt.xlim([0,200])
    #     print("median errs", numpy.median(errxy)*1000, numpy.median(errresid)*1000)


    plt.show()




