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

import pdb; pdb.set_trace()

# filter by unfolded arms
# df = df[df.betaMeas < 160]

# df = df.groupby(["datetime", "positionerID", ""]).mean().reset_index()

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
plt.hist(df.alphaErr, bins=500)

plt.figure()
plt.hist(df.betaErr, bins=500)


plt.figure(figsize=(13,8))
sns.scatterplot(x="datetime", y="alphaErr", ci=None, markers=True, alpha=0.1, data=df)


plt.figure(figsize=(13,8))
sns.scatterplot(x="datetime", y="betaErr", ci=None, markers=True, alpha=0.1, data=df)


plt.figure(figsize=(13,8))
sns.scatterplot(x="datetime", y="rErr", ci=None, markers=True, alpha=0.1, data=df)

plt.show()


import pdb; pdb.set_trace()