import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy

# gfa1timeOff = 37*24*60*60
df = pandas.read_csv("ptData.csv")
df = df[df.gfaNum==1]
# df["dateObsMJD"] = df.dateObsMJD + gfa1timeOff
keep = numpy.abs(df.tccAz.to_numpy()-df.actAz.to_numpy()) < 1 #0.5
_df = df.iloc[keep]
_df = _df[_df.grms != -999]

df = _df.copy()
df["dAz"] = df.actAz - df.tccAz
df["dAlt"] = df.actAlt - df.tccAlt

df.to_csv("ptDataClean.csv", index=False)

def plotSkyPoints():
    theta = numpy.radians(df.tccAz)
    r = 90 - df.tccAlt
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location("S")
    c = ax.scatter(theta, r, alpha=1, s=5, c=df.dAz.to_numpy())
    altLabels = ax.get_yticklabels()
    newLabels = []
    for altLabel in altLabels:
        tick = int(altLabel.get_text())
        tick = 90 - tick
        tick = str(tick) + "$^o$"
        altLabel.set_text(tick)
        newLabels.append(altLabel)
    ax.set_yticklabels(newLabels)
    plt.colorbar(c)
    plt.title("Az Err (deg)")

    azLabels = ax.get_xticklabels()
    directions = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
    newLabels = []
    for azLabel, direction in zip(azLabels, directions):
        tick = azLabel.get_text()
        tick = tick + " (%s)"%direction
        azLabel.set_text(tick)
        newLabels.append(azLabel)
    ax.set_xticklabels(newLabels)


    theta = numpy.radians(df.tccAz)
    r = 90 - df.tccAlt
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location("S")
    c = ax.scatter(theta, r, alpha=1, s=5, c=df.dAlt.to_numpy())
    altLabels = ax.get_yticklabels()
    newLabels = []
    for altLabel in altLabels:
        tick = int(altLabel.get_text())
        tick = 90 - tick
        tick = str(tick) + "$^o$"
        altLabel.set_text(tick)
        newLabels.append(altLabel)
    ax.set_yticklabels(newLabels)
    plt.colorbar(c)
    plt.title("Alt Err (deg)")

    azLabels = ax.get_xticklabels()
    directions = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
    newLabels = []
    for azLabel, direction in zip(azLabels, directions):
        tick = azLabel.get_text()
        tick = tick + " (%s)"%direction
        azLabel.set_text(tick)
        newLabels.append(azLabel)
    ax.set_xticklabels(newLabels)


    plt.show()

def fixTime():

    gfa1 = df[df.gfaNum==1]
    mg = df.merge(gfa1, on=["mjd", "imgNum"], suffixes=(None, "_1"))
    mg["dt1"] = (mg.dateObsMJD - mg.dateObsMJD_1)*24*60*60

    sns.lineplot(x="dateObsMJD", y="dt1", hue="gfaNum", style="gfaNum", data=mg)
    plt.show()

    mg["dateObsMJD"]
    # import pdb; pdb.set_trace()

def filterData():
    gfa1 = df[df.gfaNum==1]
    mg = df.merge(gfa1, on=["mjd", "imgNum"], suffixes=(None, "_1"))
    mg["dateObsMJD"] = mg.dateObsMJD_1
    _df = mg[[df.columns]]
    import pdb; pdb.set_trace()

def dAz():
    dAz = df.tccAz-df.actAz
    plt.figure()
    plt.hist(dAz, bins=200)
    plt.show()

def grms():
    plt.figure()
    plt.hist(df.grms, bins=200)
    plt.show()


if __name__ == "__main__":
    plotSkyPoints()
    # grms()
    # plotTimeOffset()
    # filterData()
    # df2 = df.groupby(["mjd", "imgNum"]).count().reset_index()
    # import pdb; pdb.set_trace()
# sns.scatterplot(x="tccAlt", y="tccAz", hue="mjd", data=df)