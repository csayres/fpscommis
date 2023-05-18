import numpy
numpy.random.seed(0)
from coordio.defaults import calibration
from coordio.transforms import xyWokFiberFromPositioner
import matplotlib.pyplot as plt
import pandas

wokErr = 10/1000 # microns in mm
pointErr = 0.1 * 0.06 # arcsec in mm
addNoise = False

positionerTable = calibration.positionerTable.reset_index()
wokCoords = calibration.wokCoords.reset_index()
fullTable = positionerTable.merge(wokCoords, on="holeID")

fullTableErr = fullTable.copy()

avgBossX = numpy.mean(fullTable.bossX)
avgBossY = numpy.mean(fullTable.bossY)

avgApX = numpy.mean(fullTable.apX)
avgApY = numpy.mean(fullTable.apY)

fullTableErr["bossX"] = avgBossX
fullTableErr["bossY"] = avgBossY

fullTableErr["apX"] = avgApX
fullTableErr["apY"] = avgApY

dAp = numpy.sqrt((fullTable.apX-avgApX)**2+(fullTable.apY-avgApY)**2)
dBoss = numpy.sqrt((fullTable.bossX-avgBossX)**2+(fullTable.bossY-avgBossY)**2)

plt.figure()
plt.hist(dAp*1000, bins=numpy.linspace(0,200,10))

plt.figure()
plt.hist(dBoss*1000, bins=numpy.linspace(0,200,10))

# plt.show()
# import pdb; pdb.set_trace()

# simulate real positions in wok space
# 10 random locations per positioner
nIter = 20
alphas = numpy.random.uniform(0,360, size=(nIter, len(fullTable)))
betas = numpy.random.uniform(0,180, size=(nIter, len(fullTable)))

realizations = []
ii = 0
for alpha, beta in zip(alphas, betas):

    for table in [fullTable, fullTableErr]:
        table["alphaSim"] = alpha
        table["betaSim"] = beta
        xyWokFiberFromPositioner(table, angleType="Sim")

    # add noise to fake measurement
    xnoise = numpy.random.normal(scale=pointErr) + numpy.random.normal(scale=wokErr, size=len(positionerTable))
    ynoise = numpy.random.normal(scale=pointErr) + numpy.random.normal(scale=wokErr, size=len(positionerTable))
    for fiber in ["APOGEE", "BOSS"]:
        if addNoise:
            fullTableErr["xWokSim"+fiber] += xnoise
            fullTableErr["yWokSim"+fiber] += ynoise
        dx = fullTable["xWokSim"+fiber] - fullTableErr["xWokSim"+fiber]
        dy = fullTable["yWokSim"+fiber] - fullTableErr["yWokSim"+fiber]
        fullTableErr["dxWokSim"+fiber] = dx
        fullTableErr["dyWokSim"+fiber] = dy
        fullTableErr["bossXTrue"] = fullTable["bossX"]
        fullTableErr["bossYTrue"] = fullTable["bossY"]
        fullTableErr["apXTrue"] = fullTable["apX"]
        fullTableErr["apYTrue"] = fullTable["apY"]

        # rotate dxyWoks into beta arm frame
        alphaOff = fullTableErr.alphaOffset.to_numpy()
        betaOff = fullTableErr.betaOffset.to_numpy()
        totalRot = numpy.radians(alpha+beta+alphaOff+betaOff-90)

        dxbeta = numpy.cos(totalRot)*dx + numpy.sin(totalRot)*dy
        dybeta = -numpy.sin(totalRot)*dx + numpy.cos(totalRot)*dy
        fullTableErr["dxBetaSim"+fiber] = dxbeta
        fullTableErr["dyBetaSim"+fiber] = dybeta
    fullTableErr["dxBetaTrueBOSS"] = -1*(fullTableErr.bossX - fullTable.bossX)
    fullTableErr["dyBetaTrueBOSS"] = -1*(fullTableErr.bossY - fullTable.bossY)
    fullTableErr["dxBetaTrueAPOGEE"] = -1*(fullTableErr.apX - fullTable.apX)
    fullTableErr["dyBetaTrueAPOGEE"] = -1*(fullTableErr.apY - fullTable.apY)

    fullTableErr["iter"] = ii
    ii += 1
    realizations.append(fullTableErr.copy())

dfAll = pandas.concat(realizations)
positionerIDs = list(set(dfAll.positionerID))
for positionerID in positionerIDs[:10]:
    _df = dfAll[dfAll.positionerID==positionerID]
    plt.figure(figsize=(5,5))
    plt.plot(_df.dxBetaSimBOSS, _df.dyBetaSimBOSS, '.', alpha=0.2, color="tab:blue")
    plt.plot(_df.dxBetaTrueBOSS, _df.dyBetaTrueBOSS, 'x', color="tab:blue")
    plt.plot(_df.dxBetaSimAPOGEE, _df.dyBetaSimAPOGEE, '.', alpha=0.2, color="tab:red")
    plt.plot(_df.dxBetaTrueAPOGEE, _df.dyBetaTrueAPOGEE, 'x', color="tab:red")
    plt.xlim([-.1,.1])
    plt.ylim([-.1,.1])
    plt.grid("on")
    # plt.axis("equal")

plt.show()


import pdb; pdb.set_trace()




# simulate fiber offsets derived from dithers,
# test how well the "truth" is recoverd