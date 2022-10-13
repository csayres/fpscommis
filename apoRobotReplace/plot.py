from coordio.defaults import calibration
import numpy
import matplotlib.pyplot as plt

pt = calibration.positionerTable.reset_index()
wc = calibration.wokCoords.reset_index()
ft = pt.merge(wc, on="holeID")

# permDisable = numpy.array([54, 184, 344, 444, 608, 612, 1042, 1093, 1136, 1182])
# maybeBad = numpy.array([902,1064,1053])

permDisable = numpy.array([54, 184, 344, 444, 608, 612, 1042, 1093, 1182, 1053, 1136])
maybeBad = numpy.array([902, 1064, 437, 1201])

# collided = numpy.array([
#    [40,          1],
#    [52,          1],
#    [78,          2],
#   [115,          1],
#   [146,          1],
#   [160,          5],
#   [184,         72],
#   [208,          5],
#   [238,          1],
#   [260,          1],
#   [264,          2],
#   [310,          1],
#   [344,         17],
#   [372,          1],
#   [396,          1],
#   [422,          1],
#   [432,          1],
#   [444,         36],
#   [450,          1],
#   [466,          1],
#   [493,          1],
#   [507,          1],
#   [544,          1],
#   [565,         25],
#   [588,         19],
#   [608,         69],
#   [612,         42],
#   [627,          1],
#   [664,         11],
#   [693,          1],
#   [706,          2],
#   [891,          2],
#   [902,         25],
#   [995,          3],
#  [1042,         10],
#  [1053,         11],
#  [1064,         12],
#  [1093,         48],
#  [1136,         45],
#  [1182,         46],
#  [1272,          1],
#  [1285,          1],
# ])

plt.figure(figsize=(10,10))
plt.plot(ft.xWok, ft.yWok, 'o', ms=15, mfc="none", mec="black")

_df = ft[ft.hexCol==1]
_xs = _df.xWok
_ys = _df.yWok
_r = _df.hexRow
for x,y,r in zip(_xs,_ys,_r):
    plt.text(x-40,y,"%i"%r, ha="right", va="center")

_df = ft[ft.positionerID.isin(permDisable)]

def plotIndiv(_df, color):
    _xs = _df.xWok
    _ys = _df.yWok
    _holeID = _df.holeID
    _pID = _df.positionerID
    plt.plot(_xs,_ys,"o", color=color, ms=14)

    for _x, _y, _h, _p in zip(_xs,_ys,_holeID,_pID):
        strP = "P"+("%i"%_p).zfill(4)
        text="%s\n%s"%(strP, _h)
        plt.text(_x,_y,text,ha="center",va="center",fontsize=4)

plotIndiv(_df, "red")

_df = ft[ft.positionerID.isin(maybeBad)]
plotIndiv(_df,"gray")


plt.axis("equal")
plt.xlabel("x wok")
plt.ylabel("y wok")
plt.savefig("badrobots.pdf")
