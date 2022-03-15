import pandas
import glob
import time
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.transform import SimilarityTransform

# csvs = glob.glob("utah/fvc*.csv")

# dfList = []

# for csv in csvs:
#     print("processing", csv)
#     dfList.append(pandas.read_csv(csv))

# dfAll = pandas.concat(dfList)
# dfAll.to_csv("utah/fvcAll.csv")
alt = 70
boxsize = 3
sigma = 0.7

tstart = time.time()
df = pandas.read_csv("utah/fvcAll.csv", engine="c")
df = df[(df.rotpos < 185) | (df.rotpos > 188.5)]
# import pdb; pdb.set_trace()
df["alt"] = numpy.round(df["alt"])
df = df[df.alt==alt]
df = df[df.wpSig==sigma]
df = df[df.boxSize==boxsize]

rotposs = list(set(df.rotpos))
df_std = df.groupby(["rotpos", "positionerID", "boxSize"]).std().reset_index()
df_mean = df.groupby(["rotpos", "positionerID", "boxSize"]).mean().reset_index()

df_std["meanWokX"] = df_mean.xWokReportMetrology
df_std["meanWokY"] = df_mean.yWokReportMetrology
df_std["meanCCDX"] = df_mean.x
df_std["meanCCDY"] = df_mean.y
df_std["meanWinposX"] = df_mean.xWinpos
df_std["meanWinposY"] = df_mean.yWinpos
df_std["meanRotpos"] = df_mean.rotpos
df_std["meanBoxsize"] = df_mean.boxSize

# xyOrig = df_mean[["x", "y"]].to_numpy()
# xyNew = df_mean[["xWinpos", "yWinpos"]].to_numpy()
# st = SimilarityTransform()
# st.estimate(xyOrig, xyNew)
# xyFit = st(xyOrig)
# print(st.translation, numpy.degrees(st.rotation), st.scale)

# df_std["dx"] = df_std.meanWinposX - xyFit[:,0]
# df_std["dy"] = df_std.meanWinposY - xyFit[:,1]
# df_std["dr"] = numpy.sqrt(df_std.dx**2+df_std.dy**2)

# df_std_r = df_std[df_std.dr > 0.2]

# plt.figure()
# plt.quiver(df_std_r.meanCCDX, df_std_r.meanCCDY, df_std_r.dx, df_std_r.dy, angles="xy", scale_units="xy")
# plt.axis("equal")

# plt.figure()
# plt.quiver(df_std.meanCCDX, df_std.meanCCDY, df_std.dx, df_std.dy, angles="xy", scale_units="xy")
# plt.axis("equal")


# plt.show()


xyOrig = df[["x", "y"]].to_numpy()
xyNew = df[["xWinpos", "yWinpos"]].to_numpy()
st = SimilarityTransform()
st.estimate(xyOrig, xyNew)
xyFit = st(xyOrig)
print(st.translation, numpy.degrees(st.rotation), st.scale)

df["dx"] = df.xWinpos - xyFit[:,0]
df["dy"] = df.yWinpos - xyFit[:,1]
df["dr"] = numpy.sqrt(df.dx**2+df.dy**2)

df_r = df[df.dr > 0.2]


plt.figure()
plt.quiver(df_r.x, df_r.y, df_r.dx, df_r.dy, angles="xy", scale_units="xy")
plt.axis("equal")

plt.figure()
plt.quiver(df.x, df.y, df.dx, df.dy, angles="xy", scale_units="xy")
plt.axis("equal")


plt.show()


# plt.figure()
# sns.lineplot(x="meanRotpos", y="xWinpos", data=df_std, label="winpos")
# sns.lineplot(x="meanRotpos", y="x", data=df_std, label="norm")
# plt.legend()

# plt.figure()
# sns.lineplot(x="meanRotpos", y="yWinpos", data=df_std)
# sns.lineplot(x="meanRotpos", y="y", data=df_std)



# plt.figure()
# sns.lineplot(x="meanRotpos", y="yWinpos", style="meanBoxsize", data=df_std)

# plt.show()

# sns.scatterplot(x="meanCCDX", y="meanCCDY", size="xWinpos", alpha=0.5, hue="xWinpos", style="boxSize", data=df_std)
# plt.show()

# seems like a bad regime arouond 185 degrees?
# df_std = df_std[(df_std.rotpos < 185) | (df_std.rotpos > 188.5)]

# import pdb; pdb.set_trace()
# plt.figure(figsize=(8,8))
# sns.scatterplot(x="meanWokX", y="meanWokY", size="xWokMeasMetrology", hue="xWokMeasMetrology", hue_norm=(0,0.02), palette="mako", alpha=0.5, data=df_std)
# plt.axis("equal")

# plt.figure(figsize=(8,8))
# sns.scatterplot(x="meanWokX", y="meanWokY", size="yWokMeasMetrology", hue="yWokMeasMetrology", hue_norm=(0,0.02), palette="mako", alpha=0.5, data=df_std)
# plt.axis("equal")

# plt.figure(figsize=(8,8))
# sns.scatterplot(x="meanCCDX", y="meanCCDY", size="xWokMeasMetrology", hue="xWokMeasMetrology", hue_norm=(0,0.02), palette="mako", alpha=0.5, data=df_std)
# plt.axis("equal")

# plt.figure(figsize=(8,8))
# sns.scatterplot(x="meanCCDX", y="meanCCDY", size="yWokMeasMetrology", hue="yWokMeasMetrology", hue_norm=(0,0.02), palette="mako", alpha=0.5, data=df_std)
# plt.axis("equal")

# plt.figure()
# sns.lineplot(x="rotpos", y="xWokMeasMetrology", ci="sd", data=df_std)

# plt.figure()
# sns.lineplot(x="rotpos", y="yWokMeasMetrology", ci="sd", data=df_std)

# plt.figure()
# plt.hist(df_std.xWokMeasMetrology*1000, bins=numpy.arange(30))
# plt.xlabel("x std")
# plt.figure()
# plt.hist(df_std.yWokMeasMetrology*1000, bins=numpy.arange(30))
# plt.xlabel("y std")
# plt.show()
