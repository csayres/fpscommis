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

tstart = time.time()
df = pandas.read_csv("utah/fvcAll.csv", engine="c")
df["alt"] = numpy.round(df["alt"])
df = df[df.alt==70]

rotposs = list(set(df.rotpos))
sigmas = sorted(list(set(df.wpSig)))
boxSizes = list(set(df.boxSize))

df_nom = df[df.rotpos==135.4]
df_nom_mean = df_nom.groupby(["rotpos", "wpSig", "boxSize", "positionerID"]).mean().reset_index()
df_rest = df[df.rotpos!=135.4]
df_rest_mean = df_rest.groupby(["rotpos", "wpSig", "boxSize", "positionerID"]).mean().reset_index()

_rotpos = []
_sig = []
_box = []
_meanErr = []
_medianErr = []
_stdErr = []
_05err = []
_25err = []
_75err = []
_90err = []
_95err = []


dfList = []

for rotpos in list(set(df_rest.rotpos)):
    gotOld = False
    for sigma in sigmas:
        for box in boxSizes:
            if sigma==0.0:
                if gotOld:
                    continue
                gotOld = True
                box = 3

            _df1 = df_nom_mean[(df_nom_mean.wpSig==sigma) & (df_nom_mean.boxSize==box)]
            _df2 = df_rest_mean[
                (df_rest_mean.rotpos==rotpos) & \
                (df_rest_mean.wpSig==sigma) & \
                (df_rest_mean.boxSize==box)
            ]

            dxyMet = _df1[["xWokMeasMetrology", "yWokMeasMetrology"]].to_numpy() - \
                    _df2[["xWokMeasMetrology", "yWokMeasMetrology"]].to_numpy()


            err = numpy.linalg.norm(dxyMet, axis=1)
            _meanErr.append(numpy.mean(err))
            _medianErr.append(numpy.median(err))
            _stdErr.append(numpy.std(err))
            _05err.append(numpy.quantile(err, 0.05))
            _25err.append(numpy.quantile(err, 0.25))
            _75err.append(numpy.quantile(err, 0.75))
            _90err.append(numpy.quantile(err, 0.90))
            _95err.append(numpy.quantile(err, 0.95))
            _rotpos.append(rotpos)
            _sig.append(sigma)
            _box.append(box)

df = pandas.DataFrame({
    "rotpos":_rotpos,
    "sig":_sig,
    "box":_box,
    "meanErr":_meanErr,
    "medianErr":_medianErr,
    "stdErr":_stdErr,
    "05err":_05err,
    "25err":_25err,
    "75err":_75err,
    "90err":_90err,
    "95err":_95err,
})


olds = df[df.sig==0]

news = df[(df.sig==0.7) & (df.box==3)]

# plt.figure()
# oldRot = olds.rotpos.to_numpy()
# oldMed = olds.medianErr.to_numpy()
# old05 = olds["05err"].to_numpy()
# old95 = old["95err"].to_numpy()


plt.figure()
sns.lineplot(x="rotpos", y="95err", hue="box", style="sig", data=df)

plt.figure()
sns.lineplot(x="rotpos", y="medianErr", hue="box", style="sig", data=df)
plt.show()






# print(set(df.rotpos))
# print("took %.1f secs to load"%(time.time()-tstart))
