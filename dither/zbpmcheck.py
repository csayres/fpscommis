import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from coordio.utils import fitsTableToPandas
from coordio.transforms import FVCTransformAPO
import numpy
from processDither import parseConfSummary
import pandas
import seaborn as sns


def getFiles():
    configID = 5478
    files = sorted(glob.glob("proc*.fits"))
    confsum = parseConfSummary("confSummary-5478.par")
    confsum["positionerID"] = confsum["positionerId"]

    confsumBoss = confsum[confsum.fiberType=="BOSS"][["positionerID", "fiberId", "activeFiber", "xwok", "ywok", "alpha", "beta"]]
    confsumMetrology = confsum[confsum.fiberType=="METROLOGY"][["positionerID", "fiberId", "xwok", "ywok"]]

    confsumCut = confsumBoss.merge(confsumMetrology, on="positionerID", suffixes=("_confBOSS", "_confMet"))


    confsumF = parseConfSummary("confSummaryF-5478.par")
    confsumF["positionerID"] = confsumF["positionerId"]




    bossFiberID = 195 # robot 501

    # confrow = confsum[confsum.fiberId==bossFiberID]
    # positionerID = int(confrow["positionerId"])

    nudgeCounter = 0
    zbplusCounter = 0
    zbminusCounter = 0

    dfList = []

    for f in files:
        ff = fits.open(f)
        centtype = ff[1].header["FVC_CNTT"]
        if ff[1].header["CONFIGID"] != configID:
            continue
        if centtype == "nudge":
            nudgeCounter += 1
            ii = nudgeCounter
        if centtype == "zbplus":
            zbplusCounter += 1
            ii = zbplusCounter
        if centtype == "zbminus":
            zbminusCounter += 1
            ii = zbminusCounter
        pa = fitsTableToPandas(ff["POSANGLES"].data)
        IPA = ff[1].header["IPA"]
        imgData = ff[1].data

        ft = FVCTransformAPO(
            imgData,
            pa,
            IPA
        )
        ft.extractCentroids()
        ft.fit(centType=centtype)

        df = ft.positionerTableMeas
        df["fvcIter"] = ii
        df["centType"] = centtype

        dfList.append(df)

    df = pandas.concat(dfList)
    df = df.merge(confsumCut, on="positionerID")

    df.to_csv("config5478.csv")

# getFiles()

df = pandas.read_csv("config5478.csv")

dxBoss = df.xWokMeasBOSS - df.xwok_confBOSS
dyBoss = df.yWokMeasBOSS - df.ywok_confBOSS
drBoss = numpy.sqrt(dxBoss**2+dyBoss*2)
df["dxBoss"] = dxBoss
df["dyBoss"] = dyBoss
df["drBoss"] = drBoss

dxMet = df.xWokMeasMetrology - df.xwok_confMet
dyMet = df.yWokMeasMetrology - df.ywok_confMet
drMet = numpy.sqrt(dxMet**2+dyMet*2)
df["dxMet"] = dxMet
df["dyMet"] = dyMet
df["drMet"] = drMet

# for centType in ["nudge", "zbplus", "zbminus"]:
#     _df = df[(df.centType==centType) & (df.fvcIter==2) & (df.activeFiber==True)]
#     # import pdb; pdb.set_trace()

#     # fiberBoss = _df.fiberId_confBOSS.to_numpy()
#     # positionerID = _df.positionerID.to_numpy()
#     # xBoss = _df.dxBoss.to_numpy()
#     # yBoss = _df.dyBoss.to_numpy()
#     # drBoss = _df.drBoss.to_numpy()
#     # drMet = _df.drMet.to_numpy()
#     # asBoss = numpy.argsort(drBoss)[::-1]



#     plt.figure()
#     rmsBoss = numpy.sqrt(numpy.mean(_df.drMet**2))*1000
#     per90Boss = numpy.nanpercentile(_df.drMet, 90)*1000
#     sns.scatterplot(x="dxMet", y="dyMet", data=_df)
#     plt.title(centType + " BOSS\nrms=%.1f  perc90=%.1f"%(rmsBoss, per90Boss))
#     # for ii in range(3):
#     #     ind = asBoss[ii]
#     #     plt.text(xBoss[ind], yBoss[ind], str(fiberBoss[ind]))
#     plt.axis("equal")


# plt.show()

for centType in ["nudge", "zbplus", "zbminus"]:
    _df = df[df.centType==centType]
    iter1 = _df[_df.fvcIter==1]
    iter2 = _df[_df.fvcIter==2]
    plt.figure()
    plt.plot(iter1.dxMet, iter1.dyMet, '.', color="tab:orange")
    plt.plot(iter2.dxMet, iter2.dyMet, '.', color="tab:blue")
    plt.show()



import pdb; pdb.set_trace()



