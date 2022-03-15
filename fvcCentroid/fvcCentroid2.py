import glob
from astropy.io import fits
import sep
import numpy
import pandas
from coordio.transforms import FVCTransformAPO
from coordio.utils import fitsTableToPandas
from coordio.defaults import calibration
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse


fvcFiles = glob.glob("/Volumes/futa/apo/data/fcam/59605/proc*.fits")


def dataGen():

    ftList = []
    for file in fvcFiles:
        print("on file", file)
        ff = fits.open(file)
        # import pdb; pdb.set_trace()
        IPA = ff[1].header["IPA"]
        LED = ff[1].header["LED1"]
        ROTPOS = ff[1].header["ROTPOS"]
        positionerCoords = fitsTableToPandas(ff[7].data)

        imgNum = int(file.split("-")[-1].split(".")[0])
        if LED < 5.5:
            ff.close()
            continue

        fvcTF = FVCTransformAPO(
            ff[1].data,
            positionerCoords,
            IPA,
            file
        )
        fvcTF.extractCentroids()
        fvcTF.fit()
        # print(fvcTF.getMetadata())
        ft = fvcTF.positionerTableMeas
        ft["rotpos"] = ROTPOS
        ft["imgnum"] = imgNum
        ftList.append(ft)
        # import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()

    ft = pandas.concat(ftList)
    ft.to_csv("centData.csv")

dataGen()

ft = pandas.read_csv("centData.csv")
ft["roundness"] = numpy.sqrt(ft.x2**2+ft.y2**2)



mean = ft.groupby(["rotpos", "positionerID"]).mean().reset_index()



plt.figure(figsize=(8,8))
sns.scatterplot(x="x", y="y", hue="xy", data=ft, palette="vlag")
plt.axis("equal")

plt.figure(figsize=(8,8))
sns.scatterplot(x="x", y="y", hue="y2", data=ft)
plt.axis("equal")

plt.figure(figsize=(8,8))
sns.scatterplot(x="x", y="y", hue="x2", data=ft)
plt.axis("equal")

# ax = plt.gca()
# for ii, row in mean.iterrows():
#     e = Ellipse(xy=(row['xCCD'], row['yCCD']),
#                 width=6*row['a'],
#                 height=6*row['b'],
#                 angle=row['theta'] * 180. / numpy.pi)
#     e.set_facecolor('none')
#     e.set_edgecolor('red')
#     ax.add_artist(e)


plt.show()


_ft = ft[ft.rotpos.isin([135.4, 225.4])]


_ft135 = _ft.loc[_ft.rotpos==135.4]

# import pdb; pdb.set_trace()

x = _ft135.xWokExpectMetrology.to_numpy()
y = _ft135.yWokExpectMetrology.to_numpy()

import pdb; pdb.set_trace()

dx = _ft135.xWokMeasMetrology.to_numpy() - x
dy = _ft135.yWokMeasMetrology.to_numpy() - y
err = numpy.sqrt(dx**2+dy**2)*1000

plt.figure()
plt.quiver(x[err<500],y[err<500],dx[err<500],dy[err<500],angles="xy", units="xy", scale=1/200)
plt.axis("equal")



pos0 = mean.loc[mean.rotpos == 135.4]
pos1 = mean.loc[mean.rotpos == 225.4]

dx = pos0.xWokMeasMetrology.to_numpy() - pos1.xWokMeasMetrology.to_numpy()
dy = pos0.yWokMeasMetrology.to_numpy() - pos1.yWokMeasMetrology.to_numpy()
err = numpy.sqrt(dx**2+dy**2)*1000
x = pos0.xWokMeasMetrology.to_numpy()
y = pos0.yWokMeasMetrology.to_numpy()

plt.figure()
plt.quiver(x[err<500],y[err<500],dx[err<500],dy[err<500],angles="xy", units="xy", scale=1/200)
plt.axis("equal")

plt.figure()
plt.hist(err[err<500])

print("rms err", numpy.sqrt(numpy.mean(err)))

plt.show()

# import pdb; pdb.set_trace()

# sns.scatterplot(x="xWokMetMeas", y="yWokMetMeas", hue="rotpos", data=_ft)
# plt.axis("equal")
# plt.show()

    # import pdb; pdb.set_trace()

    # out =



    # import pdb; pdb.set_trace()


