import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy


std = pandas.read_csv("std.csv") # includes rot
mean = pandas.read_csv("mean.csv") # includes rot

std_rotmarg = pandas.read_csv("std_rotmarg.csv") # does not include rot
mean_rotmarg = pandas.read_csv("mean_rotmarg.csv") # does not include rot

std["xWokMeasMetUM"] = std.xWokMeasMetrology * 1000
std["yWokMeasMetUM"] = std.yWokMeasMetrology * 1000
std_rotmarg["xWokMeasMetUM"] = std_rotmarg.xWokMeasMetrology * 1000
std_rotmarg["yWokMeasMetUM"] = std_rotmarg.yWokMeasMetrology * 1000

std = std[std.xWokMeasMetUM < 1000]
std = std[std.yWokMeasMetUM < 1000]
std_rotmarg = std_rotmarg[std_rotmarg.xWokMeasMetUM < 1000]
std_rotmarg = std_rotmarg[std_rotmarg.yWokMeasMetUM < 1000]

std = std[std.polidName != "desi"]
std_rotmarg = std_rotmarg[std_rotmarg.polidName != "desi"]
# std = std[std.polidName != "z4"]

# confids = [4113, 4114]

for ii, df in enumerate([std, std_rotmarg]):
    # for confid in confids:
    #     df = std[_df.configid==confid]
    plt.figure(figsize=(10,10))
    sns.kdeplot(data=df, x="xWokMeasMetUM", y="yWokMeasMetUM", hue="polidName", levels=20)
    if ii == 0:
        plt.title("stdev at same rot angle")

    if ii == 1:
        plt.title("stdev across all rot angles")
    plt.xlim([0,30])
    plt.ylim([0,30])


    # sns.kdeplot(data=df, x="yWokMeasMetUM", hue="polidName")
    # sns.scatterplot(data=df, x="xWokMeasMetUM", y="yWokMeasMetUM", hue="polidName", alpha=0.5)
    # plt.xlim([0,50])
    # plt.ylim([0,50])
    # plt.title(str(confid))

    # plt.figure()
    # sns.boxplot(data=df[df.polidName=="nom"], x="rotpos", y="xWokMeasMetUM")

    # plt.figure()
    # sns.boxplot(data=df[df.polidName=="nom"], x="rotpos", y="yWokMeasMetUM")


_df = mean[mean.polidName=="nom"]
_df["dx_trans"] = _df.xtrans - numpy.mean(_df.xtrans)
_df["dy_trans"] = _df.ytrans - numpy.mean(_df.ytrans)

plt.figure()
sns.scatterplot(data=_df, x="rotpos", y="dx_trans")

plt.figure()
sns.scatterplot(data=_df, x="rotpos", y="dy_trans")

plt.figure()
sns.scatterplot(data=_df, x="dx_trans", y="dy_trans", hue="rotpos")
plt.axis("equal")


plt.figure()
sns.scatterplot(data=_df, x="temp", y="scale", hue="configid")
# plt.axis("equal")

plt.show()

import pdb; pdb.set_trace()