import glob
import pandas
import time
import numpy
import seaborn as sns
import matplotlib.pyplot as plt

csvFiles = glob.glob("/Volumes/futa/utah/rotData/fvc_*.csv")
boxSize = 3
wpSig = 0.7

dfList = []
t1 = time.time()
for ff in csvFiles:
    df = pandas.read_csv(ff)
    dfList.append(df)

df = pandas.concat(dfList)
df["_alt"] = numpy.round_(df.alt, decimals=1)
df = df[(df.wpSig==0.7) & (df.boxSize==3) & (df._alt==70.0)]

# import pdb; pdb.set_trace()

# stdPos = df.groupby(["positionerID", "rotpos", "_alt"]).std().reset_index()
# meanPos = df.groupby(["positionerID", "rotpos", "_alt"]).mean().reset_index()
# meanPos["stdx"] = stdPos.xWinpos * 120
# meanPos["stdy"] = stdPos.yWinpos * 120

stdPos = df.groupby(["positionerID", "rotpos"]).std().reset_index()
meanPos = df.groupby(["positionerID", "rotpos"]).mean().reset_index()
meanPos["stdx"] = stdPos.xWokMeasMetrology * 1000
meanPos["stdy"] = stdPos.yWokMeasMetrology * 1000


meanPos = meanPos[meanPos.stdx < 2000]
meanPos = meanPos[meanPos.stdy < 2000]



meanPos["ab"] = meanPos.a / meanPos.b

plt.figure()
sns.histplot(meanPos, x="stdx", y="stdy", cbar=True)
plt.xlim([0, 50])
plt.ylim([0, 50])
plt.xlabel("std x (microns)")
plt.ylabel("std y (microns)")
# plt.title("std xy centroid (microns)")

plt.figure()
sns.kdeplot(data=meanPos, x="stdx", y="stdy", fill=True, levels=10)
plt.xlim([0, 50])
plt.ylim([0, 50])
plt.xlabel("std x (microns)")
plt.ylabel("std y (microns)")

plt.figure()
sns.violinplot(data=meanPos, x="rotpos", y="stdx") #fill=True, levels=10)
# plt.xlim([0, 50])
plt.ylim([0, 15])
plt.xlabel("rotpos")
plt.ylabel("std x (microns)")

plt.figure()
sns.violinplot(data=meanPos, x="rotpos", y="stdy") #fill=True, levels=10)
# plt.xlim([0, 15])
plt.ylim([0, 15])
plt.xlabel("rotpos")
plt.ylabel("std y (microns)")
# plt.title("std xy centroid (microns)")


# plt.figure()
# sns.histplot(stdPos, x="x", y="y", cbar=True)
# plt.xlim([0, 0.3])
# plt.ylim([0, 0.3])

# print("took", time.time()-t1)


# plt.figure(figsize=(13,13))
# sns.scatterplot(x="xWinpos", y="yWinpos", hue="stdx", data=meanPos, palette="mako_r", alpha=1, s=5)
# plt.xlabel("x microns")
# plt.ylabel("y microns")
# plt.axis("equal")

# plt.figure(figsize=(13,13))
# sns.scatterplot(x="xWinpos", y="yWinpos", hue="stdy", data=meanPos, palette="mako_r", alpha=1, s=5)
# plt.xlabel("x microns")
# plt.ylabel("y microns")
# plt.axis("equal")

# plt.figure()
# sns.pairplot(
#     data=meanPos,
#     y_vars=["stdx", "stdy"],
#     x_vars=["ab", "a", "b", "flux", "cflux", "peak", "cpeak", "imgNum", "rotpos", "x2", "y2", "xy", "positionerID"],
#     plot_kws={"alpha": 0.04, "facecolor":'k', "edgecolor":"k", "s": 1}
# )

plt.show()
import pdb; pdb.set_trace()