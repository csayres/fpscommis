import glob
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy

badRobots = [54, 184, 463, 608, 1042, 1136, 1182]


files = glob.glob("utahdata/histData/*.csv")

dfs = []
for f in files:
    df = pandas.read_csv(f)
    dfs.append(df)

df = pandas.concat(dfs)
df["datetime"] = pandas.to_datetime(df.date)
df["date"] = pandas.to_datetime(df.datetime).dt.date
df["alphaErr"] = (df.alphaReport - df.alphaMeas)
df["betaErr"] = (df.betaReport - df.betaMeas)


df["rErr"] = numpy.sqrt((df.xWokReportMetrology - df.xWokMeasMetrology)**2 + (df.yWokReportMetrology - df.yWokMeasMetrology)**2)
df["rTrans"] = numpy.sqrt((df.xtrans**2)+(df.ytrans)**2)
df["badRobots"] = df["positionerID"].isin(badRobots)

# import pdb; pdb.set_trace()

# df.to_csv("centAll.csv")

# df = pandas.read_csv("centAll.csv")

# df = df[df.rErr < 1.5]

df = df[df.boxSize != 5]
df = df[df.useWinpos == True]
# df =df[df.rErr > 0.5]

# import pdb; pdb.set_trace()

# filter by unfolded arms
# df = df[df.betaMeas < 160]

# import pdb; pdb.set_trace()

df = df.sort_values("datetime")
print("len before", len(df))
df = df.groupby(["positionerID", "configid", "mjd", "useWinpos", "boxSize"]).head(1)
df = df.groupby(["configid", "mjd", "useWinpos", "boxSize"]).median().reset_index()


print("len after", len(df))

# handle alpha wrapping
alphaErr = []
for ae in df.alphaErr.to_numpy():
    if ae > 180:
        alphaErr.append(-1*(ae-360))
    elif ae < -180:
        alphaErr.append(-1*(ae+360))
    else:
        alphaErr.append(ae)
df["alphaErr"] = alphaErr
df["rErrCut"] = df.rErr < 0.04
df = df[df.temp > -100]

# plt.hist(df.scale, bins=1000)
# plt.show()

#create datetime column

# plt.figure()
# sns.histplot(x="alphaErr", data=df, hue="useWinpos", bins=500)

# plt.figure()
# sns.histplot(x="betaErr", data=df, hue="useWinpos", bins=500)

# plt.figure()
# sns.histplot(x="rErr", data=df, hue="useWinpos", bins=500)


# plt.figure(figsize=(13,8))
# sns.lineplot(x="mjd", y="alphaErr", hue="positionerID", style="badRobots", size="badRobots", ci=None, markers=True, alpha=0.5, data=df)


# plt.figure(figsize=(13,8))
# sns.lineplot(x="mjd", y="betaErr", hue="positionerID", style="badRobots", size="badRobots", ci=None, markers=True, alpha=0.5, data=df)


# plt.figure(figsize=(13,8))
# sns.lineplot(x="mjd", y="rErr", hue="positionerID", style="badRobots", size="badRobots", ci=None, markers=True, alpha=0.5, data=df)

# plt.show()

# plt.figure(figsize=(13,13))
# vs = ["rErr", "temp", "scale", "rTrans", "xtrans", "ytrans", "fvcRot", "fiducialRMS", "ipa", "errxy", "flux", "alt", 'ZB_00', 'ZB_01', 'ZB_02', 'ZB_03', 'ZB_04', 'ZB_05', 'ZB_06', 'ZB_09', 'ZB_20', 'ZB_28', 'ZB_29']

vs = ["rErrCut", "scale", "xtrans", "ytrans", "ipa", "alt", 'ZB_00', 'ZB_01', 'ZB_02', 'ZB_03', 'ZB_04'] #, 'ZB_05', 'ZB_06', 'ZB_09'] #'ZB_20', 'ZB_28', 'ZB_29']

xvars = ["scale", "xtrans", "ytrans", "fvcRot", 'ZB_00', 'ZB_01', 'ZB_02', 'ZB_03', 'ZB_04', 'ZB_05', 'ZB_06', 'ZB_09', 'ZB_20', 'ZB_28', 'ZB_29']
yvars = ["ipa", "alt", "temp", "flux"]

# _df = df[vs]

sns.pairplot(df, x_vars=xvars, y_vars=yvars, height=1.2, hue="rErrCut", plot_kws={"alpha": 0.5, "s":1})



plt.show()


# for v in vs:
#     plt.figure(figsize=(13,13))
#     plt.plot(df[v], df.rErr, '.k', markersize=2, alpha=1)
#     plt.ylim([0,0.2])
#     plt.xlabel(v)
#     plt.ylabel("rErr")

# plt.show()

# sns.pairplot(
#     df,
#     x_vars=["scale", "rTrans", "fvcRot", "fiducialRMS", "ipa", "errxy", "flux", "alt"],
#     y_vars=["rErr", "alphaErr", "betaErr"],
#     plot_kws={"alpha": 0.3, "s": 3, "color": "black"}
#     )

# # plt.figure(figsize=(13,13))

# sns.pairplot(
#     df,
#     x_vars=['ZB_00', 'ZB_01', 'ZB_02', 'ZB_03', 'ZB_04', 'ZB_05', 'ZB_06', 'ZB_09', 'ZB_20', 'ZB_28', 'ZB_29'],
#     y_vars=["rErr", "alphaErr", "betaErr"],
#     plot_kws={"alpha": 0.3, "s": 3, "color": "black"}
# )

# plt.show()



# import pdb; pdb.set_trace()

