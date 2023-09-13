import numpy
import pandas
from astropy.io import fits
import seaborn as sns
import matplotlib.pyplot as plt

from coordio.utils import fitsTableToPandas

MJD = 59807
baseDir = "/Volumes/futa/lco/data/fcam/%i"%MJD
# configid : [minImg, maxImg] map
cmap = {
    1: [7,116],
    2: [117,226],
    3: [227,336]
}

def getImgFile(imgNum):
    imgNumStr = str(imgNum).zfill(4)
    return baseDir + "/proc-fimg-fvc1s-%s.fits"%imgNumStr


def extractData(configid, imgNum):
    ff = fits.open(getImgFile(imgNum))
    ptm = fitsTableToPandas(ff["POSITIONERTABLEMEAS"].data)
    trans = numpy.array([ff[1].header["FVC_TRAX"], ff[1].header["FVC_TRAY"]])
    scale = ff[1].header["FVC_SCL"]
    rot = ff[1].header["FVC_ROT"]
    ipa = numpy.around(ff[1].header["IPA"], decimals=1)

    ptm["config"] = configid
    ptm["imgNum"] = imgNum
    ptm["transx"] = trans[0]
    ptm["transy"] = trans[1]
    ptm["rot"] = numpy.radians(rot)
    ptm["scale"] = scale
    ptm["ipa"] = ipa

    return ptm.reset_index()


def compileData():
    dfList = []
    for config, (minImg, maxImg) in cmap.items():
        for imgNum in range(minImg, maxImg+1):
            print("on img", imgNum)
            dfList.append(extractData(config, imgNum))

    df = pandas.concat(dfList)
    df.to_csv("rawMeasAll.csv")

def plotData():
    df = pandas.read_csv("rawMeasAll.csv")
    df_med = df.groupby(["config", "positionerID"]).median().reset_index()
    df_med = df_med[["config", "positionerID", "flux", "peak"]]
    df = df.merge(df_med, on=["config", "positionerID"], suffixes=(None, "_m"))
    df["fluxPerc"] = df.flux/df.flux_m

    plt.figure()
    plt.hist(df.fluxPerc,bins=200)

    plt.figure(figsize=(8,8))
    plt.scatter(df.x, df.y, c=df.fluxPerc, s=0.2, vmin=0.9, vmax=1)
    plt.axis("equal")
    plt.colorbar()

    plt.show()

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    # compileData()
    plotData()

