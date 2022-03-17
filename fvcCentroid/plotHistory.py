import glob
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy

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

# df = df[df.rErr < 1.5]

df = df[df.boxSize != 5]

# import pdb; pdb.set_trace()

# filter by unfolded arms
# df = df[df.betaMeas < 160]

# import pdb; pdb.set_trace()

df = df.sort_values("datetime")
df = df.groupby(["positionerID", "configid", "mjd", "useWinpos", "boxSize"]).head(1)

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

#create datetime column

plt.figure()
sns.histplot(x="alphaErr", data=df, hue="useWinpos", bins=500)

plt.figure()
sns.histplot(x="betaErr", data=df, hue="useWinpos", bins=500)

plt.figure()
sns.histplot(x="rErr", data=df, hue="useWinpos", bins=500)


plt.figure(figsize=(13,8))
sns.lineplot(x="datetime", y="alphaErr", hue="useWinpos", style="positionerID", ci=None, markers=True, alpha=0.05, data=df)


plt.figure(figsize=(13,8))
sns.lineplot(x="datetime", y="betaErr", hue="useWinpos", style="positionerID", ci=None, markers=True, alpha=0.05, data=df)


plt.figure(figsize=(13,8))
sns.lineplot(x="datetime", y="rErr", hue="useWinpos", style="positionerID", ci=None, markers=True, alpha=0.05, data=df)

plt.show()


import pdb; pdb.set_trace()