import numpy
from coordio import defaults
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import time
from scipy.optimize import minimize
from coordio.transforms import positionerToWok

pt = defaults.calibration.positionerTable.reset_index()
wc = defaults.calibration.wokCoords.reset_index()
jt = pt.merge(wc, on="holeID")

def forwardModel(x, positionerID, alphaDeg, betaDeg):
    xBeta, la, alphaOffDeg, betaOffDeg, dx, dy = x
    yBeta = 0 # by definition for metrology fiber

    _jt = jt[jt.positionerID==positionerID]
    b = numpy.array([_jt[x] for x in ["xWok", "yWok", "zWok"]]).flatten()
    iHat = numpy.array([_jt[x] for x in ["ix", "iy", "iz"]]).flatten()
    jHat = numpy.array([_jt[x] for x in ["jx", "jy", "jz"]]).flatten()
    kHat = numpy.array([_jt[x] for x in ["kx", "ky", "kz"]]).flatten()



    xw, yw, zw = positionerToWok(
        alphaDeg, betaDeg,
        xBeta, yBeta, la,
        alphaOffDeg, betaOffDeg,
        dx, dy, b, iHat, jHat, kHat
    )

    return xw, yw

def minimizeMe(x, positionerID, alphaDeg, betaDeg, xWok, yWok):
    xw, yw = forwardModel(x, positionerID, alphaDeg, betaDeg)
    return numpy.sum((xw-xWok)**2 + (yw-yWok)**2)


def fitCalibs(positionerTableIn, positionerTableMeas):
    x0 = numpy.array([
        defaults.MET_BETA_XY[0], defaults.ALPHA_LEN,
        0, 0, 0, 0
    ])
    positionerIDs = positionerTableIn.positionerID.to_numpy()
    _xBeta = []
    _la = []
    _alphaOffDeg = []
    _betaOffDeg = []
    _dx = []
    _dy = []
    for positionerID in positionerIDs:

        print("calibrating positioner", positionerID)
        _df = positionerTableMeas[positionerTableMeas.positionerID==positionerID]
        args = (
            positionerID,
            _df.alphaReport.to_numpy(),
            _df.betaReport.to_numpy(),
            _df.xWokMeasMetrology.to_numpy(),
            _df.yWokMeasMetrology.to_numpy()
        )
        tstart = time.time()
        out = minimize(minimizeMe, x0, args, method="Powell")
        xBeta, la, alphaOffDeg, betaOffDeg, dx, dy = out.x
        _xBeta.append(xBeta)
        _la.append(la)
        _alphaOffDeg.append(alphaOffDeg)
        _betaOffDeg.append(betaOffDeg)
        _dx.append(dx)
        _dy.append(dy)
        tend = time.time()
        print("took %.2f"%(tend-tstart))
    _xBeta = numpy.array(_xBeta)
    _la = numpy.array(_la)
    _alphaOffDeg = numpy.array(_alphaOffDeg)
    _betaOffDeg = numpy.array(_betaOffDeg)
    _dx = numpy.array(_dx)
    _dy = numpy.array(_dy)

    positionerTableOut = positionerTableIn.copy()
    positionerTableOut["metX"] = _xBeta
    positionerTableOut["alphaArmLen"] = _la
    positionerTableOut["alphaOffset"] = _alphaOffDeg
    positionerTableOut["betaOffset"] = _betaOffDeg
    positionerTableOut["dx"] = _dx
    positionerTableOut["dy"] = _dy

    return positionerTableOut

def plotCalib(pt_meas, title="", positionerTable=None):
    pt_meas = pt_meas.copy()

    if positionerTable is None:
        xExpect = pt_meas.xWokReportMetrology
        yExpect = pt_meas.yWokReportMetrology

    else:
        xExpect = []
        yExpect = []
        for ii, row_meas in pt_meas.iterrows():
            row_pt = positionerTable[positionerTable.positionerID==row_meas.positionerID]
            x = row_pt[["metX", "alphaArmLen", "alphaOffset", "betaOffset", "dx", "dy"]].to_numpy().squeeze()
            # import pdb; pdb.set_trace()
            xw,yw = forwardModel(x, int(row_pt.positionerID), float(row_meas.alphaReport), float(row_meas.betaReport))
            xExpect.append(xw)
            yExpect.append(yw)
            # import pdb; pdb.set_trace()

    xExpect = numpy.array(xExpect)
    yExpect = numpy.array(yExpect)
    xMeas = pt_meas.xWokMeasMetrology
    yMeas = pt_meas.yWokMeasMetrology
    dx = xExpect - xMeas
    dy = yExpect - yMeas
    dr = numpy.sqrt(dx**2+dy**2)

    rms = numpy.sqrt(numpy.mean(dr**2))*1000
    rmsStr = " RMS: %.2f um"%rms
    title = title + rmsStr

    plt.figure(figsize=(13,8))
    sns.boxplot(x=pt_meas.positionerID, y=dr)


    plt.title(title)

    plt.figure(figsize=(8,8))
    plt.quiver(xExpect, yExpect, dx, dy ,angles="xy", units="xy", scale=.1, width=1)
    plt.title(title)


if __name__ == "__main__":
    ptMeas = pandas.read_csv("octCalib.csv")

    ptMeas["xErr"] = ptMeas.xWokReportMetrology - ptMeas.xWokMeasMetrology
    ptMeas["yErr"] = ptMeas.yWokReportMetrology - ptMeas.yWokMeasMetrology
    ptMeas["rErr"] = numpy.sqrt(ptMeas.xErr**2+ptMeas.yErr**2)

    ptMeas = ptMeas[ptMeas.rErr < 1]

    plotCalib(ptMeas)

    # plt.figure()
    # plt.hist(ptMeas.rErr, bins=numpy.linspace(0,2,1000))
    # plt.show()
    # import pdb; pdb.set_trace()

    # ptOut = fitCalibs(pt, ptMeas)
    # ptOut.to_csv("positionerTable_new.csv")

    ptOut = pandas.read_csv("positionerTable_new.csv", index_col=0)
    plotCalib(ptMeas, positionerTable=ptOut)

    plt.figure()
    ptMeas["is985"] = ptMeas.positionerID==985
    sns.histplot(x="peak", hue="is985", data=ptMeas, stat="density", element="step")

    newRobots = [1231,710,920,776,440,983,1144,56,877,1370,515,1017,898,1111,1092,792,1256,204,751,615,916,478,117,423,535]
    print(len(newRobots))

    ptMerge = pt.merge(ptOut, on="positionerID", suffixes=(None, "_new"))
    ptMerge["isNew"] = ptMerge.positionerID.isin(newRobots)
    ptMerge = ptMerge.merge(wc, on="holeID", suffixes=(None, "_wc"))

    plt.figure()
    plt.hist(ptMerge.alphaOffset, label="alpha old")
    plt.hist(ptMerge.betaOffset, label="beta old")
    plt.hist(ptMerge.alphaOffset_new, label="alpha new")
    plt.hist(ptMerge.betaOffset_new, label="beta new")
    plt.legend()

    plt.figure()
    plt.quiver(ptMerge.xWok, ptMerge.yWok, ptMerge.dx, ptMerge.dy, color="red", angles="xy", units="xy")
    plt.quiver(ptMerge.xWok, ptMerge.yWok, ptMerge.dx_new, ptMerge.dy_new, color="black", angles="xy", units="xy")
    _df = ptMerge[ptMerge.isNew]
    plt.plot(_df.xWok, _df.yWok, 'o', ms=10, mfc="none", mec="red")

    plt.figure()
    ptMerge["dxMet"] = ptMerge.metX_new - ptMerge.metX
    sns.histplot(x="dxMet", hue="isNew", element="step", data=ptMerge)

    # sort by positioner ID
    ptMerge = ptMerge.sort_values("positionerID").reset_index(drop=True)
    ptOut = ptOut.sort_values("positionerID").reset_index(drop=True)

    ptOut["apX"] = ptMerge.dxMet + ptOut.apX
    ptOut["bossX"] = ptMerge.dxMet + ptOut.bossX

    ptOut.to_csv("positionerTable_metCalib.csv")

    # import pdb; pdb.set_trace()


    # _df = ptMeas[ptMeas.positionerID==985]
    # plt.plot(_df.imgNum, _df.flux)

    # plt.show()

