import pandas
from coordio.defaults import calibration
import matplotlib.pyplot as plt


fpiIDs = [75,225]

# pt = calibration.positionerTable.reset_index()

fa = calibration.fiberAssignments.reset_index()
fa2 = fa[fa["APOGEEFiber"].notna()]
holeIDs1 = fa2.holeID.to_numpy()


wc = calibration.wokCoords.reset_index()
import pdb; pdb.set_trace()
wc2 = wc[wc.holeType=="ApogeeBoss"]
holeIDs2 = wc2.holeID.to_numpy()

xWok = wc.xWok.to_numpy()
yWok = wc.yWok.to_numpy()
holeIDs = wc.holeID.to_numpy()

plt.figure(figsize=(8,8))
for x,y,h in zip(xWok,yWok,holeIDs):
    if h in holeIDs1 and h in holeIDs2:
        plt.plot(x,y,'.k')
        continue
    elif h in holeIDs1:
        color="green"
    elif h in holeIDs2:
        color="red"
    else:
        continue

    plt.text(x,y,h,color=color, va="center",ha="center")

plt.xlim([-350,350])
plt.ylim([-350,350])
plt.show()

