import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
from functools import partial
from scipy.optimize import minimize

do_plot = True

df = pandas.read_csv("jj.csv")

# normalize flux by flux in sky fibers
# import pdb; pdb.set_trace()
meanSky = numpy.mean(df[df.isSky]["flux"])
sciFibers = df[df.isSky==False]
sciFibers["synthmag"] = 2.5*numpy.log10(sciFibers.flux)
# print("minflux", numpy.min(sciFibers.flux))
base = sciFibers[df.isParent==True]


_df = sciFibers.groupby(["fiberId"])[["hmag", "synthmag"]].max()
_df.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
_df = _df.dropna()
_df = _df[_df.hmag > -2]
# plt.plot(_df.hmag, _df.synthmag, 'o', fillstyle="none", markeredgecolor="b")

# fit the enveloope
X = numpy.ones((len(_df), 2))
X[:,1] = _df.hmag.to_numpy()
coeff = numpy.linalg.lstsq(X, _df.synthmag.to_numpy())[0]


def fit(_hmag):
    return coeff[0] + _hmag * coeff[1]


def norm(_synthmag, _hmag):
    n = fit(_hmag)
    return _synthmag / n


def makePlot(df, y):
    plt.figure()
    plt.plot(df.hmag, df[y], '.k')
    # plt.hist(df.flux, bins=1000)
    _base = sciFibers[df.isParent==True]
    plt.plot(_base.hmag, _base[y], 'xr')
    plt.xlim([8,14])
    _df = df.groupby(["fiberId"])[["hmag", y]].max()
    _df.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
    _df = _df.dropna()
    _df = _df[_df.hmag > -2]
    plt.plot(_df.hmag, _df[y], 'o', fillstyle="none", markeredgecolor="b")


if do_plot:
    makePlot(sciFibers, "synthmag")
    plt.plot(_df.hmag, fit(_df.hmag), ':r')


sciFibers["fluxnorm"] = norm(sciFibers.synthmag, sciFibers.hmag)

sciFibers.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)
sciFibers = sciFibers.dropna(subset=["fluxnorm"])
sciFibers = sciFibers[sciFibers.hmag > -2]
#

if do_plot:
    makePlot(sciFibers, "fluxnorm")


# plt.show()
# import pdb; pdb.set_trace()

sciFibers["fluxnorm2"] = sciFibers.fluxnorm**10

# cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
# import pdb; pdb.set_trace()

if do_plot:
    plt.figure(figsize=(12,12))
    sns.scatterplot(x="xFocal", y="yFocal", hue="fluxnorm2", size="fluxnorm2", data=sciFibers)
    ax = plt.gca()
    ax.plot(base.xFocal, base.yFocal, 'xr')

fiberIds = list(set(sciFibers.fiberId))


def gaussModel(x, y, xc, yc, std, amp, dcOff):
    return amp*numpy.exp(-1*((x-xc)**2 + (y-yc)**2)/(2*std**2)) + dcOff


def gaussMin(p, xData, yData, fluxData, std, amp, dcOff):
    xc,yc = p
    err = fluxData - gaussModel(xData, yData, xc, yc, std, amp, dcOff)
    return numpy.sum(err**2)

# fiberIds = [23, 39, 57, 67, 78]
xTest = []
yTest = []
xMeas = []
yMeas = []


for fiberId in sorted(fiberIds):
    print("on fiberID", fiberId)
    _df = sciFibers[sciFibers.fiberId == fiberId]
    _base = base[base.fiberId == fiberId].iloc[0]
    # sort by flux
    topFlux = _df.nlargest(1, "flux")

    bottomFlux = _df.nsmallest(4, "flux")
    fluxThresh = numpy.mean(bottomFlux.flux) + 2*numpy.std(bottomFlux.flux)

    keep = _df.copy()
    keep = keep[keep.flux > fluxThresh]
    keep["fluxnorm"] = keep.flux / numpy.sum(keep.flux)

    mx = numpy.sum(keep.fluxnorm*keep.xFocal)
    my = numpy.sum(keep.fluxnorm*keep.yFocal)

    # print("sum norm", numpy.sum(keep.fluxnorm))

    # print("got", len(keep), "fibers")
    # plt.figure()
    # sns.scatterplot(x="xFocal", y="yFocal", hue="flux", size="flux", data=keep)
    # plt.plot(_base.xFocal, _base.yFocal, 'xb')
    # plt.plot(mx, my, 'o', fillstyle="none", markeredgecolor="r")
    # plt.show()
    if numpy.abs(mx-float(_base.xFocal)) > 1:
        continue


    xMeas.append(mx)
    yMeas.append(my)

    xTest.append(float(_base.xFocal))
    yTest.append(float(_base.yFocal))

xTest = numpy.array(xTest)
yTest = numpy.array(yTest)
xMeas = numpy.array(xMeas)
yMeas = numpy.array(yMeas)
dx = xMeas - xTest
dy = yMeas - yTest
plt.figure(figsize=(10,10))
plt.quiver(xMeas, yMeas, -dx, -dy, angles="xy")
plt.title("APOGEE MJD=59584 Design=35913")
plt.xlabel("xFocal (mm)")
plt.ylabel("yFocal (mm)")

plt.figure()
plt.hist(numpy.sqrt(dx**2+dy**2))
plt.show()

# plt.figure()
# plt.hist(stds, bins=100)
# plt.show()

if do_plot:
    plt.show()

