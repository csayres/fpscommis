import numpy
from coordio.defaults import calibration
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

"""
This takes measurements with LTC and Microscopy to solve for scale/orientaiton
on microscopy measurements for Apogee+Boss robots, then applies a simple fit
to update the Boss robots without LTC apogee measurements.

not done here, but probably a good idea incase a robot ever gets plugged into
apogee later, adjust apogee fiber positions based on met pos in a similar way.

"""

measDir = "/Volumes/futa/fibermeas/2022-01-24T17:30:34.427885/"

pt = calibration.positionerTable.reset_index()
wc = calibration.wokCoords.reset_index()
fa = calibration.fiberAssignments.reset_index()
pt = pt.merge(wc, on="holeID")

pt = pt.merge(fa, on="holeID")
pt["positionerID"] = pt.positionerID_x

# import pdb; pdb.set_trace()

allCentroids = pandas.read_csv(measDir + "allCentroids.csv")
allCentroids["positionerID"] = [int(x.strip("P")) for x in allCentroids.robotID.to_numpy()]

# measuredRobots = set(allCentroids.positionerID)
activeRobots = set(pt.positionerID)


micDF = allCentroids[allCentroids.positionerID.isin(activeRobots)]


# reformat the positioner table
keys = list(pt.columns)
_keys = ["metX", "metY", "apX", "apY", "bossX", "bossY"]
for _key in _keys:
    keys.remove(_key)

d = {}
for key in keys:
    d[key] = []
d["x"] = []
d["y"] = []
d["fiberID"] = []

for ii, row in pt.iterrows():
    for key in keys:
        d[key].append(row[key])

    d["x"].append(row.metX)
    d["y"].append(row.metY)
    d["fiberID"].append("Metrology")

    for key in keys:
        d[key].append(row[key])

    d["x"].append(row.bossX)
    d["y"].append(row.bossY)
    d["fiberID"].append("BOSS")

    for key in keys:
        d[key].append(row[key])

    d["x"].append(row.apX)
    d["y"].append(row.apY)
    d["fiberID"].append("Apogee")

ptDF = pandas.DataFrame(d)

### first just look at robots we have measured with the ltc
ptDF_ltc = ptDF[ptDF.holeType=="ApogeeBoss"]
ltcRobots = list(set(ptDF_ltc.positionerID))

micDF_ltc = micDF[micDF.positionerID.isin(ltcRobots)]

_positionerID = []
_bossX = []
_bossY = []
for _robot in ltcRobots:
    _mic = micDF_ltc[micDF_ltc.positionerID == _robot]
    _pt = ptDF_ltc[ptDF_ltc.positionerID == _robot]
    if len(_mic) == 0:
        print("no meas for robot", _robot)
        continue

    # solve the linear transform to new coords
    _micMet = _mic[(_mic.fiberID=="Metrology")]
    _micMetX = float(_micMet.xBetaMeasMM)
    _micMetY = float(_micMet.yBetaMeasMM)

    _micAp = _mic[(_mic.fiberID=="Apogee")]
    _micApX = float(_micAp.xBetaMeasMM)
    _micApY = float(_micAp.yBetaMeasMM)

    _micBoss = _mic[(_mic.fiberID=="BOSS")]
    _micBossX = float(_micBoss.xBetaMeasMM)
    _micBossY = float(_micBoss.yBetaMeasMM)


    _ptMet = _pt[_pt.fiberID=="Metrology"]
    _ptMetX = float(_ptMet.x)
    _ptMetY = float(_ptMet.y)

    _ptAp = _pt[_pt.fiberID=="Apogee"]
    _ptApX = float(_ptAp.x)
    _ptApY = float(_ptAp.y)

    xa = (_ptApX - _ptMetX) / (_micApX - _micMetX)
    xb = _ptApX - xa * _micApX

    ya = (_ptApY - _ptMetY) / (_micApY - _micMetY)
    yb = _ptApY - ya * _micApY

    _positionerID.append(_robot)
    _bossX.append(xa*_micBossX + xb)
    _bossY.append(ya*_micBossY + yb)

    # import pdb; pdb.set_trace()


newMeasDF = pandas.DataFrame({
    "positionerID": _positionerID,
    "bossXNew": _bossX,
    "bossYNew": _bossY
    })


# look at differences
_pt = pt[pt.positionerID.isin(_positionerID)]

_pt = _pt.merge(newMeasDF, on="positionerID")

print("len updates", len(_positionerID))

plt.figure(figsize=(10,10))
plt.plot(_pt.bossX, _pt.bossY, '.k', alpha=0.2, markersize=10, label="old positions")
plt.plot(_pt.bossXNew, _pt.bossYNew, '.r', alpha=0.2, markersize=10, label="new positions")
plt.xlabel("x beta (mm)")
plt.ylabel("y beta (mm)")
plt.legend()



_pt["dxBoss"] = _pt.bossX - _pt.bossXNew
_pt["dyBoss"] = _pt.bossY - _pt.bossYNew
_pt["dr"] = numpy.sqrt(_pt.dxBoss**2 + _pt.dyBoss**2)

modeledBossY = numpy.median(newMeasDF.bossYNew)


plt.figure()
plt.hist(_pt.dr*1000)
plt.xlabel("dr (um)")

# plt.show()

plt.figure()
sns.scatterplot(x="bossXNew", y="bossYNew", hue="metX", data=_pt)

mx = _pt.metX.to_numpy()
bx = _pt.bossXNew.to_numpy()

X = numpy.ones((len(mx), 2))
X[:,1] = mx

coeff = numpy.linalg.lstsq(X, bx)[0]


def predBossX(metX):
    return coeff[0] + coeff[1]*metX


plt.figure()
sns.scatterplot(x="metX", y="bossXNew", data=_pt)
plt.plot(_pt.metX, predBossX(_pt.metX), 'r')
# plt.show()
plt.figure()
plt.hist((_pt.bossYNew - modeledBossY)*1000, bins=50)
plt.grid("on")


# finally update boss fiber locations
pt = calibration.positionerTable.reset_index()
positioners = list(pt.positionerID)
_bossX = []
_bossY = []
_fiberID = []
_positionerID = []
_fit = []

for positioner in positioners:
    xx = pt[pt.positionerID==positioner]
    yy = _pt[_pt.positionerID==positioner]
    fid = fa[fa.holeID == xx.holeID.values[0]]["BOSSFiber"]
    _fiberID.append(int(fid))
    _positionerID.append(positioner)

    if len(yy)==0:
        # print("no existing meas for", positioner, "modeling instead")
        _bossY.append(modeledBossY)
        _bossX.append(predBossX(float(xx.metX)))
        _fit.append(True)
    else:
        # print("existing meas for", positioner)
        _bossX.append(float(yy.bossXNew))
        _bossY.append(float(yy.bossYNew))
        _fit.append(False)


pt["bossX"] = _bossX
pt["bossY"] = _bossY

plt.figure(figsize=(10,10))
plt.plot(pt.metX, pt.metY, '.k', alpha=0.1)
plt.plot(pt.apX, pt.apY, '.k', alpha=0.1)
plt.plot(pt.bossX, pt.bossY, '.k', alpha=0.1)
plt.axis("equal")


plt.figure(figsize=(10,10))
plt.plot(pt.apX-pt.metX, pt.apY, '.k', alpha=0.1)
plt.plot(pt.bossX-pt.metX, pt.bossY, '.r', alpha=0.1)
plt.axis("equal")

plt.figure(figsize=(10,10))
plt.plot(pt.apX-pt.metX, pt.apY-numpy.median(pt.apY), '.k', alpha=0.1)
plt.plot(pt.bossX-pt.metX, pt.bossY-numpy.median(pt.bossY), '.r', alpha=0.1)
plt.axis("equal")

plt.show()

pt.to_csv("positionerTable_bossUpdate.csv")


# create table for yue
df = pandas.DataFrame({
    "positionerID": _positionerID,
    "bossFiberID": _fiberID,
    "bossX": _bossX,
    "bossY": _bossY,
    "fit": _fit

    })

df.to_csv("bossFiberMeas.csv", index=False)

# import pdb; pdb.set_trace()



