import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy

# fit coeffs by polidName
polidSet = {
    "nom": [0, 1, 2, 3, 4, 5, 6, 9, 20, 28, 29],
    "desi": [0,1,2,3,4,5,6,9,20,27,28,29,30],
    "all": list(range(33)),
}


if False:

    def fiberCircle(units, ax):
        if units=="mm":
            r = 0.060
        else: # pixels
            r = 0.50
        theta = numpy.linspace(0, 2*numpy.pi, 1000)
        x = r * numpy.cos(theta)
        y = r * numpy.sin(theta)
        ax.plot(x,y,'--', color="tab:cyan", label="fiber core")


    def fiberDia(units, ax):
        if units=="mm":
            r = 0.060
        else: # pixels
            r = 0.50
        ax.plot.axvline(-r, label="fiber core")
        ax.plot.axvline(r, label="fiber core")



    # following csvs generated by
    # ~/fpscommis/fvc/fvcCentroidRotReduce.py at Utah
    # raw data fcam/59661 image numberse 26-619
    # 11 fvc images at rot angles at 15 degree increments


    # grouped by positionerID, configid, rotpos, polidName
    std = pandas.read_csv("std.csv")
    mean = pandas.read_csv("mean.csv")

    df1 = mean.merge(std, on=["positionerID", "configid", "polidName", "rotpos"], suffixes=(None, "_std"))

    # grouped by positionerID, configid, rotpos, polidName
    std_rotmarg = pandas.read_csv("std_rotmarg.csv")
    mean_rotmarg = pandas.read_csv("mean_rotmarg.csv")

    rotmarg = mean_rotmarg.merge(std_rotmarg, on=["positionerID", "configid", "polidName"], suffixes=(None, "_std"))

    # grouped by positionerID, configid, rotpos, polidName, nexp
    # nexp is number of images used to average
    # exposures are drawn randomly from the set of 11 at each rotator position
    # random draws are repeated 6 times at each rotator position

    std_stacked = pandas.read_csv("std_stacked.csv")
    mean_stacked = pandas.read_csv("mean_stacked.csv")

    stacked = mean_stacked.merge(std_stacked, on=["positionerID", "configid", "polidName", "nexp"], suffixes=(None, "_std"))

    # find configID/rot positions with bogus detections
    # tossOut = std[std.yWinpos > 10]
    # for ii, row in tossOut.iterrows():
    #     print(row.configid, row.rotpos, row.positionerID)
    # # fraction thrown out
    # print("throw out fraction", len(tossOut)/len(std))
    # two robots seem to be the problem (985 and 1300)
    # throw them out from all data

    df1 = df1[~df1.positionerID.isin([985, 1300])]
    rotmarg = rotmarg[~rotmarg.positionerID.isin([985, 1300])]
    stacked = stacked[~stacked.positionerID.isin([985, 1300])]


    ########### CENTROIDING ##################
    # scatter in raw centroiding
    df = df1[df1.polidName == "nom"]
    # throw out a a few problematic data points to make autoplotting better
    df = df[df.x2_std < 1]
    df = df[df.cflux_std < 5000]


    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWinpos_std", y="yWinpos_std", color="tab:cyan", data=df, levels=10)
    plt.plot(df.xWinpos_std, df.yWinpos_std, '.k', markersize=5, alpha=0.2, zorder=1)
    plt.xlabel("std x (pixels)")
    plt.ylabel("std y (pixels)")

    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWinpos_std", y="x2_std", color="tab:cyan", data=df, levels=10)
    plt.plot(df.xWinpos_std, df.x2_std, '.k', markersize=5, alpha=0.2, zorder=1)
    plt.xlabel("std x (pixels)")
    plt.ylabel("std x2 (pixels)")


    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWinpos_std", y="y2_std", color="tab:cyan", data=df, levels=10)
    plt.plot(df.xWinpos_std, df.y2_std, '.k', markersize=5, alpha=0.2, zorder=1)
    plt.xlabel("std x (pixels)")
    plt.ylabel("std y2 (pixels)")

    plt.figure(figsize=(10,10))
    sns.kdeplot(x="yWinpos_std", y="x2_std", color="tab:cyan", data=df, levels=10)
    plt.plot(df.xWinpos_std, df.x2_std, '.k', markersize=5, alpha=0.2, zorder=1)
    plt.xlabel("std y (pixels)")
    plt.ylabel("std x2 (pixels)")


    plt.figure(figsize=(10,10))
    sns.kdeplot(x="yWinpos_std", y="y2_std", color="tab:cyan", data=df, levels=10)
    plt.plot(df.xWinpos_std, df.y2_std, '.k', markersize=5, alpha=0.2, zorder=1)
    plt.xlabel("std y (pixels)")
    plt.ylabel("std y2 (pixels)")


    sns.pairplot(
        x_vars=["flux_std", "cflux_std", "peak_std", "cpeak_std", "x2_std", "y2_std", "xy_std"],
        y_vars=["xWinpos_std", "yWinpos_std"],
        data=df,
        plot_kws={"alpha": 0.1, "linewidth": 0, "color": "black", "s": 1}
    )

    # scatter as a function of CCD position
    plt.figure(figsize=(10,10))
    sns.scatterplot(x="xWinpos", y="yWinpos", hue="xWinpos_std", data=df, alpha=1, linewidth=0, s=8)
    plt.xlabel("x CCD (pixels)")
    plt.ylabel("y CCD (pixels)")
    plt.axis("equal")

    plt.figure(figsize=(10,10))
    sns.scatterplot(x="xWinpos", y="yWinpos", hue="yWinpos_std", data=df, alpha=1, linewidth=0, s=8)
    plt.xlabel("x CCD (pixels)")
    plt.ylabel("y CCD (pixels)")
    plt.axis("equal")

    # scatter as a function of rotated CCD position
    plt.figure(figsize=(10,10))
    sns.scatterplot(x="xWinposRot", y="yWinposRot", hue="xWinpos_std", data=df, alpha=1, linewidth=0, s=8)
    plt.xlabel("x CCD (rot, pixels)")
    plt.ylabel("y CCD (rot, pixels)")
    plt.axis("equal")

    plt.figure(figsize=(10,10))
    sns.scatterplot(x="xWinposRot", y="yWinposRot", hue="yWinpos_std", data=df, alpha=1, linewidth=0, s=8)
    plt.xlabel("x CCD (rot, pixels)")
    plt.ylabel("y CCD (rot, pixels)")
    plt.axis("equal")


    # morphology vs CCD
    plt.figure(figsize=(10,10))
    sns.scatterplot(x="xWinpos", y="yWinpos", hue="x2", data=df, alpha=1, linewidth=0, s=8)
    plt.xlabel("x CCD (pixels)")
    plt.ylabel("y CCD (pixels)")
    plt.axis("equal")

    plt.figure(figsize=(10,10))
    sns.scatterplot(x="xWinpos", y="yWinpos", hue="y2", data=df, alpha=1, linewidth=0, s=8)
    plt.xlabel("x CCD (pixels)")
    plt.ylabel("y CCD (pixels)")
    plt.axis("equal")


    plt.figure(figsize=(10,10))
    sns.scatterplot(x="x2", y="y2", hue="xWinpos_std", data=df, alpha=1, linewidth=0, s=3)
    plt.xlabel("x2 (pixels)")
    plt.ylabel("y2 (pixels)")
    plt.axis("equal")

    plt.figure(figsize=(10,10))
    sns.scatterplot(x="x2", y="y2", hue="yWinpos_std", data=df, alpha=1, linewidth=0, s=3)
    plt.xlabel("x2 (pixels)")
    plt.ylabel("y2 (pixels)")
    plt.axis("equal")


    # centroid location after averaging
    #this takes 40 minutes to run!!!
    df = stacked[stacked.polidName=="nom"]
    for nexp in [2,3,4,5]:
        print("nexp", nexp)
        _df = df[df.nexp==nexp]

        plt.figure(figsize=(10,10))
        sns.kdeplot(x="xWinpos_std", y="yWinpos_std", color="tab:cyan", data=_df, levels=10)
        plt.plot(_df.xWinpos_std, _df.yWinpos_std, '.k', markersize=5, alpha=0.1, zorder=1)
        plt.xlabel("std x (pixels)")
        plt.ylabel("std y (pixels)")
        plt.xlim([0, 0.5])
        plt.ylim([0, 0.5])
        plt.title("nexp %i"%nexp)


    ##### looking at xy fit position ##########
    df = rotmarg[rotmarg.polidName=="nom"]
    df2 = df1[df1.polidName=="nom"]

    # import pdb; pdb.set_trace()

    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWokMeasMetrology_std", y="yWokMeasMetrology_std", color="tab:cyan", data=df, levels=10)
    plt.plot(df.xWokMeasMetrology_std, df.yWokMeasMetrology_std, '.k', markersize=10, alpha=1, zorder=1)
    plt.xlabel("std x (mm)")
    plt.ylabel("std y (mm)")
    plt.xlim([0, 0.04])
    plt.ylim([0, 0.04])
    plt.title("metrology: all rot angles")

    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWokMeasBOSS_std", y="yWokMeasBOSS_std", color="tab:cyan", data=df, levels=10)
    plt.plot(df.xWokMeasBOSS_std, df.yWokMeasBOSS_std, '.k', markersize=10, alpha=1, zorder=1)
    plt.xlabel("std x (mm)")
    plt.ylabel("std y (mm)")
    plt.xlim([0, 0.04])
    plt.ylim([0, 0.04])
    plt.title("BOSS: all rot angles")

    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWokMeasAPOGEE_std", y="yWokMeasAPOGEE_std", color="tab:cyan", data=df, levels=10)
    plt.plot(df.xWokMeasAPOGEE_std, df.yWokMeasAPOGEE_std, '.k', markersize=10, alpha=1, zorder=1)
    plt.xlabel("std x (mm)")
    plt.ylabel("std y (mm)")
    plt.xlim([0, 0.04])
    plt.ylim([0, 0.04])
    plt.title("APOGEE: all rot angles")

    ########### same rot angle

    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWokMeasMetrology_std", y="yWokMeasMetrology_std", color="tab:cyan", data=df2, levels=10)
    plt.plot(df2.xWokMeasMetrology_std, df2.yWokMeasMetrology_std, '.k', markersize=10, alpha=1, zorder=1)
    plt.xlabel("std x (mm)")
    plt.ylabel("std y (mm)")
    plt.xlim([0, 0.04])
    plt.ylim([0, 0.04])
    plt.title("metrology: same rot angle")

    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWokMeasBOSS_std", y="yWokMeasBOSS_std", color="tab:cyan", data=df2, levels=10)
    plt.plot(df2.xWokMeasBOSS_std, df2.yWokMeasBOSS_std, '.k', markersize=10, alpha=1, zorder=1)
    plt.xlabel("std x (mm)")
    plt.ylabel("std y (mm)")
    plt.xlim([0, 0.04])
    plt.ylim([0, 0.04])
    plt.title("BOSS: same rot angle")

    plt.figure(figsize=(10,10))
    sns.kdeplot(x="xWokMeasAPOGEE_std", y="yWokMeasAPOGEE_std", color="tab:cyan", data=df2, levels=10)
    plt.plot(df2.xWokMeasAPOGEE_std, df2.yWokMeasAPOGEE_std, '.k', markersize=10, alpha=1, zorder=1)
    plt.xlabel("std x (mm)")
    plt.ylabel("std y (mm)")
    plt.xlim([0, 0.04])
    plt.ylim([0, 0.04])
    plt.title("APOGEE: same rot angle")

    df = stacked[stacked.polidName=="nom"]
    # this takes forever to run (40 mins?)
    for nexp in [2,3,4,5]:
        print("nexp", nexp)
        _df = df[df.nexp==nexp]
        plt.figure(figsize=(10,10))
        sns.kdeplot(x="xWokMeasMetrology_std", y="yWokMeasMetrology_std", color="tab:cyan", data=_df, levels=10)
        plt.plot(_df.xWokMeasMetrology_std, _df.yWokMeasMetrology_std, '.k', markersize=10, alpha=1, zorder=1)
        plt.xlabel("std x (mm)")
        plt.ylabel("std y (mm)")
        plt.xlim([0, 0.04])
        plt.ylim([0, 0.04])
        plt.title("metrology: same rot angle, nexp %i"%nexp)

    # look at mean xy values, and try to find a rotator angle that is closest to that?
    df = df1[df1.polidName == "nom"]
    # throw out a a few problematic data points to make autoplotting better
    # df = df[df.x2_std < 1]
    # df = df[df.cflux_std < 5000]
    rotposList = list(set(df.rotpos))
    rotpos = []
    dx = []
    dy = []
    for _rotpos in rotposList:
        _df = df[df.rotpos==_rotpos]
        _dfm = _df.merge(mean_rotmarg[mean_rotmarg.polidName=="nom"], on=["positionerID", "configid"], suffixes=(None, "_mean"))
        _dx = _dfm["xWokMeasMetrology"] - _dfm["xWokMeasMetrology_mean"]
        _dy = _dfm["yWokMeasMetrology"] - _dfm["yWokMeasMetrology_mean"]
        dx.extend(list(_dx))
        dy.extend(list(_dy))
        rotpos.extend([_rotpos]*len(_dx))

    dd = pandas.DataFrame({
        "rotpos": rotpos,
        "dx": dx,
        "dy": dy
    })

    dd["dr"] = numpy.sqrt(dd.dx**2 + dd.dy**2)

    plt.figure()
    plt.plot(dd.rotpos, dd.dr, '.k')

    plt.figure()
    sns.boxplot(x="rotpos", y="dr", data=dd)


# overwrite df1
df = pandas.read_csv("demean.csv")
print(set(df.configid))
import pdb; pdb.set_trace()
# df = df[~df.positionerID.isin([985, 1300])]
df["rWokMeasMetrology"] = numpy.linalg.norm(df[["xWokMeasMetrology", "yWokMeasMetrology"]].to_numpy(), axis=1)

plt.figure(figsize=(10,10))
sns.jointplot(x=df.xWokMeasMetrology, y=df.yWokMeasMetrology, hue=df.configid, alpha=1)
# plt.plot(df.xWokMeasMetrology.to_numpy(), df.yWokMeasMetrology.to_numpy(), '.k', alpha=1) #0.01)
# plt.xlim([-0.03, 0.03])
# plt.ylim([-0.03, 0.03])
# plt.axis("equal")
plt.title("demean")

# plt.figure()
sns.jointplot(x=df.rotpos, y=df.rWokMeasMetrology, hue=df.positionerID, alpha=1)



# df = pandas.read_csv("demd.csv")
# plt.figure(figsize=(10,10))
# plt.plot(df.xWokMeasMetrology.to_numpy(), df.yWokMeasMetrology.to_numpy(), '.k', alpha=0.01)
# # plt.xlim([-0.03, 0.03])
# # plt.ylim([-0.03, 0.03])
# # plt.axis("equal")
# plt.title("demedian")


plt.show()
