import pandas
import glob
import time
import numpy
import seaborn as sns
import matplotlib.pyplot as plt

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
df["alt"] = numpy.round(df["alt"])
df = df[df.alt==alt]
df = df[df.wpSig==sigma]
df = df[df.boxSize==boxsize]

rotposs = list(set(df.rotpos))
df_std = df.groupby(["rotpos", "positionerID"]).std().reset_index()
df_mean = df.groupby(["rotpos", "positionerID"]).mean().reset_index()

df_std["meanWokX"] = df_mean.xWokReportMetrology
df_std["meanWokY"] = df_mean.yWokReportMetrology
df_std["meanCCDX"] = df_mean.x
df_std["meanCCDY"] = df_mean.y

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

plt.figure()
sns.lineplot(x="rotpos", y="xWokMeasMetrology", ci="sd", data=df_std)

plt.figure()
sns.lineplot(x="rotpos", y="yWokMeasMetrology", ci="sd", data=df_std)

plt.figure()
plt.hist(df_std.xWokMeasMetrology*1000, bins=numpy.arange(30))
plt.xlabel("x std")
plt.figure()
plt.hist(df_std.yWokMeasMetrology*1000, bins=numpy.arange(30))
plt.xlabel("y std")
plt.show()
